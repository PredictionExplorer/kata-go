#!/usr/bin/env python3
"""Crash-safe, content-verified publication of self-play model exports."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import errno
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)


MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
DEFAULT_REQUIRED_FILES = ("model.bin.gz", "model.ckpt")
EXPORT_CONTRACT = "katago-hardened-candidate-publication-v2"


class ExportError(RuntimeError):
    """Publication error with a stable code for automation."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    def run(
        self, argv: Sequence[str], *, timeout: Optional[float] = None
    ) -> CommandResult: ...


class SubprocessCommandRunner:
    """Run argv directly. Shell interpretation is intentionally unavailable."""

    def run(
        self, argv: Sequence[str], *, timeout: Optional[float] = None
    ) -> CommandResult:
        _validate_argv(argv)
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


@dataclass(frozen=True)
class ExportRequest:
    source_dir: Path
    destination_root: Path
    candidate_name: str
    model_name: str
    export_command: Tuple[str, ...]
    clean_checkpoint_command: Tuple[str, ...]
    model_probe_command: Tuple[str, ...] = ()
    required_files: Tuple[str, ...] = DEFAULT_REQUIRED_FILES
    source_checkpoint_name: str = "model.ckpt"
    uncompressed_model_name: str = "model.bin"
    compressed_model_name: str = "model.bin.gz"
    command_timeout_seconds: Optional[float] = None
    unsafe_allow_unprobed_for_tests: bool = False

    def normalized(self) -> "ExportRequest":
        return dataclasses.replace(
            self,
            source_dir=Path(self.source_dir),
            destination_root=Path(self.destination_root),
            export_command=tuple(self.export_command),
            clean_checkpoint_command=tuple(self.clean_checkpoint_command),
            model_probe_command=tuple(self.model_probe_command),
            required_files=tuple(self.required_files),
        )


@dataclass(frozen=True)
class PublicationResult:
    final_dir: Path
    manifest_sha256: str
    idempotent: bool
    source_checkpoint_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finalDir": str(self.final_dir),
            "manifestSha256": self.manifest_sha256,
            "idempotent": self.idempotent,
            "sourceCheckpointSha256": self.source_checkpoint_sha256,
        }


class HardenedExporter:
    def __init__(
        self,
        *,
        command_runner: Optional[CommandRunner] = None,
    ) -> None:
        self.runner = command_runner or SubprocessCommandRunner()

    def publish(self, request: ExportRequest) -> PublicationResult:
        request = request.normalized()
        self._validate_request(request)

        source_checkpoint = request.source_dir / request.source_checkpoint_name
        source_snapshot = _stable_file_snapshot(source_checkpoint)
        if source_snapshot is None:
            raise ExportError(
                "source_changed",
                "Source checkpoint changed while it was being hashed",
                details={"path": str(source_checkpoint)},
            )
        source_checkpoint_sha256, source_checkpoint_size = source_snapshot

        request.destination_root.mkdir(parents=True, exist_ok=True)
        final_dir = request.destination_root / request.candidate_name
        request_fingerprint = _request_fingerprint(request)
        if final_dir.exists() or final_dir.is_symlink():
            manifest_sha256 = self._assert_existing_request(
                final_dir,
                request=request,
                request_fingerprint=request_fingerprint,
                source_checkpoint_sha256=source_checkpoint_sha256,
                source_checkpoint_size=source_checkpoint_size,
            )
            return PublicationResult(
                final_dir=final_dir,
                manifest_sha256=manifest_sha256,
                idempotent=True,
                source_checkpoint_sha256=source_checkpoint_sha256,
            )

        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{request.candidate_name}.",
                suffix=".partial",
                dir=str(request.destination_root),
            )
        )

        try:
            placeholders = {
                "source_dir": str(request.source_dir),
                "source_checkpoint": str(source_checkpoint),
                "partial_dir": str(staging_dir),
                "destination_root": str(request.destination_root),
                "candidate_name": request.candidate_name,
                "model_name": request.model_name,
                "model_bin": str(staging_dir / request.uncompressed_model_name),
                "model_file": str(staging_dir / request.compressed_model_name),
                "cleaned_checkpoint": str(staging_dir / "model.ckpt"),
            }
            self._run_command(
                request.export_command,
                placeholders,
                request.command_timeout_seconds,
                "model_export_failed",
            )
            self._run_command(
                request.clean_checkpoint_command,
                placeholders,
                request.command_timeout_seconds,
                "checkpoint_clean_failed",
            )

            # export_model_pytorch.py writes a diagnostic log containing the
            # unique staging path. It is transient, not a model artifact, and
            # would otherwise make retries produce different manifests.
            transient_log = staging_dir / "log.txt"
            if transient_log.exists():
                if transient_log.is_symlink() or not transient_log.is_file():
                    raise ExportError(
                        "invalid_export_output",
                        "Transient export log is not a regular file",
                        details={"path": str(transient_log)},
                    )
                transient_log.unlink()

            uncompressed_model = staging_dir / request.uncompressed_model_name
            compressed_model = staging_dir / request.compressed_model_name
            _require_regular_file(uncompressed_model, "uncompressed model")
            if compressed_model.exists():
                raise ExportError(
                    "unexpected_export_output",
                    "Exporter produced the reserved compressed model path",
                    details={"path": str(compressed_model)},
                )
            _deterministic_gzip(uncompressed_model, compressed_model)
            uncompressed_model.unlink()

            self._validate_required_files(staging_dir, request.required_files)
            if request.model_probe_command:
                self._run_command(
                    request.model_probe_command,
                    placeholders,
                    request.command_timeout_seconds,
                    "model_probe_failed",
                )
                # A probe must not replace or delete required artifacts.
                self._validate_required_files(staging_dir, request.required_files)

            if _stable_file_snapshot(source_checkpoint) != source_snapshot:
                raise ExportError(
                    "source_changed",
                    "Source checkpoint changed during export; refusing publication",
                    details={"path": str(source_checkpoint)},
                )

            files = _build_file_manifest(staging_dir)
            manifest: Dict[str, Any] = {
                "schemaVersion": MANIFEST_SCHEMA_VERSION,
                "exportContract": EXPORT_CONTRACT,
                "requestFingerprintSha256": request_fingerprint,
                "modelProbePassed": bool(request.model_probe_command),
                "candidateName": request.candidate_name,
                "modelName": request.model_name,
                "sourceCheckpoint": {
                    "name": request.source_checkpoint_name,
                    "sha256": source_checkpoint_sha256,
                    "size": source_checkpoint_size,
                },
                "files": files,
            }
            manifest_bytes = _canonical_json(manifest)
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            manifest_path = staging_dir / MANIFEST_FILENAME
            _write_new_file(manifest_path, manifest_bytes)

            _fsync_tree(staging_dir)
            _fsync_directory(request.destination_root)

            if final_dir.exists() or final_dir.is_symlink():
                self._assert_identical_publication(
                    final_dir, manifest_bytes, manifest_sha256
                )
                return PublicationResult(
                    final_dir=final_dir,
                    manifest_sha256=manifest_sha256,
                    idempotent=True,
                    source_checkpoint_sha256=source_checkpoint_sha256,
                )

            try:
                os.rename(str(staging_dir), str(final_dir))
            except OSError as exc:
                # Another publisher may have won the same-name race. Only an
                # exact, valid manifest is an idempotent outcome.
                if (
                    exc.errno
                    not in (
                        errno.EEXIST,
                        errno.ENOTEMPTY,
                        errno.EACCES,
                    )
                    or not final_dir.exists()
                ):
                    raise
                self._assert_identical_publication(
                    final_dir, manifest_bytes, manifest_sha256
                )
                return PublicationResult(
                    final_dir=final_dir,
                    manifest_sha256=manifest_sha256,
                    idempotent=True,
                    source_checkpoint_sha256=source_checkpoint_sha256,
                )

            _fsync_directory(request.destination_root)
            staging_dir = final_dir
            return PublicationResult(
                final_dir=final_dir,
                manifest_sha256=manifest_sha256,
                idempotent=False,
                source_checkpoint_sha256=source_checkpoint_sha256,
            )
        except ExportError:
            raise
        except BaseException as exc:
            raise ExportError(
                "publication_failed",
                f"Could not publish model: {exc}",
                details={
                    "candidateName": request.candidate_name,
                    "destinationRoot": str(request.destination_root),
                },
            ) from exc
        finally:
            # After a successful rename staging_dir equals final_dir and must
            # remain. Every other path is a uniquely named .partial directory.
            if staging_dir != final_dir and staging_dir.exists():
                shutil.rmtree(staging_dir)
                with contextlib.suppress(OSError):
                    _fsync_directory(request.destination_root)

    def _assert_existing_request(
        self,
        final_dir: Path,
        *,
        request: ExportRequest,
        request_fingerprint: str,
        source_checkpoint_sha256: str,
        source_checkpoint_size: int,
    ) -> str:
        """Fast idempotent retry without re-running expensive model export."""

        if final_dir.is_symlink() or not final_dir.is_dir():
            raise ExportError(
                "name_collision",
                "Candidate publication path is not a regular directory",
                details={"path": str(final_dir)},
            )
        manifest_path = final_dir / MANIFEST_FILENAME
        _require_regular_file(manifest_path, "published manifest")
        manifest_bytes = manifest_path.read_bytes()
        try:
            manifest = json.loads(manifest_bytes)
        except json.JSONDecodeError as exc:
            raise ExportError(
                "name_collision",
                "Existing candidate manifest is invalid JSON",
                details={"path": str(manifest_path)},
            ) from exc
        if not isinstance(manifest, Mapping):
            raise ExportError(
                "name_collision",
                "Existing candidate manifest root is not an object",
            )
        canonical_manifest = _canonical_json(manifest)
        if manifest_bytes != canonical_manifest:
            raise ExportError(
                "name_collision",
                "Existing candidate manifest is not canonical",
                details={"path": str(manifest_path)},
            )
        _verify_manifest_files(final_dir, manifest)

        expected_identity = {
            "schemaVersion": MANIFEST_SCHEMA_VERSION,
            "exportContract": EXPORT_CONTRACT,
            "requestFingerprintSha256": request_fingerprint,
            "modelProbePassed": bool(request.model_probe_command),
            "candidateName": request.candidate_name,
            "modelName": request.model_name,
        }
        conflicts = [
            key
            for key, expected in expected_identity.items()
            if manifest.get(key) != expected
        ]
        source = manifest.get("sourceCheckpoint")
        if not isinstance(source, Mapping):
            conflicts.append("sourceCheckpoint")
        else:
            if source.get("name") != request.source_checkpoint_name:
                conflicts.append("sourceCheckpoint.name")
            if source.get("sha256") != source_checkpoint_sha256:
                conflicts.append("sourceCheckpoint.sha256")
            if source.get("size") != source_checkpoint_size:
                conflicts.append("sourceCheckpoint.size")
        if conflicts:
            raise ExportError(
                "name_collision",
                "Candidate name already exists for a different source or export contract",
                details={"path": str(final_dir), "conflicts": sorted(set(conflicts))},
            )
        return hashlib.sha256(canonical_manifest).hexdigest()

    def _validate_request(self, request: ExportRequest) -> None:
        _validate_leaf_name(request.candidate_name, "candidate_name")
        if not request.model_name or any(
            ord(character) < 32 for character in request.model_name
        ):
            raise ExportError(
                "invalid_request",
                "model_name must be non-empty and contain no control characters",
            )
        _validate_leaf_name(request.source_checkpoint_name, "source_checkpoint_name")
        _validate_leaf_name(request.uncompressed_model_name, "uncompressed_model_name")
        _validate_leaf_name(request.compressed_model_name, "compressed_model_name")
        _validate_argv(request.export_command)
        _validate_argv(request.clean_checkpoint_command)
        if not isinstance(request.unsafe_allow_unprobed_for_tests, bool):
            raise ExportError(
                "invalid_request",
                "unsafe_allow_unprobed_for_tests must be a boolean",
            )
        if not request.model_probe_command:
            if not request.unsafe_allow_unprobed_for_tests:
                raise ExportError(
                    "model_probe_required",
                    "A model-load and finite-output probe is required for publication",
                )
        else:
            _validate_argv(request.model_probe_command)
        if not request.required_files:
            raise ExportError("invalid_request", "required_files must not be empty")
        for relative_name in request.required_files:
            _validate_relative_artifact_path(relative_name)
        mandatory_files = {
            request.compressed_model_name,
            "model.ckpt",
        }
        if not mandatory_files.issubset(set(request.required_files)):
            raise ExportError(
                "invalid_request",
                "required_files must include the compressed model and cleaned checkpoint",
                details={"mandatoryFiles": sorted(mandatory_files)},
            )
        if (
            request.command_timeout_seconds is not None
            and request.command_timeout_seconds <= 0
        ):
            raise ExportError(
                "invalid_request",
                "command_timeout_seconds must be positive when configured",
            )

        if request.source_dir.is_symlink() or not request.source_dir.is_dir():
            raise ExportError(
                "source_missing",
                "Source candidate directory is missing or is a symlink",
                details={"sourceDir": str(request.source_dir)},
            )
        source_checkpoint = request.source_dir / request.source_checkpoint_name
        _require_regular_file(source_checkpoint, "source checkpoint")
        if request.destination_root.is_symlink() or (
            request.destination_root.exists() and not request.destination_root.is_dir()
        ):
            raise ExportError(
                "invalid_destination",
                "Destination root is not a regular directory",
                details={"destinationRoot": str(request.destination_root)},
            )

    def _run_command(
        self,
        template: Sequence[str],
        placeholders: Mapping[str, str],
        timeout: Optional[float],
        error_code: str,
    ) -> None:
        try:
            argv = [part.format_map(placeholders) for part in template]
        except KeyError as exc:
            raise ExportError(
                "unknown_command_placeholder",
                f"Unknown command placeholder: {exc}",
            ) from exc
        _validate_argv(argv)
        try:
            result = self.runner.run(argv, timeout=timeout)
        except ExportError:
            raise
        except BaseException as exc:
            raise ExportError(
                error_code,
                f"Command could not be executed: {exc}",
                details={"argv": argv},
            ) from exc
        if result.returncode != 0:
            raise ExportError(
                error_code,
                "Command returned a non-zero status",
                details={
                    "argv": argv,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
            )

    def _validate_required_files(
        self, root: Path, required_files: Sequence[str]
    ) -> None:
        for relative_name in required_files:
            path = root / relative_name
            _require_regular_file(path, f"required artifact {relative_name}")

    def _assert_identical_publication(
        self,
        final_dir: Path,
        expected_manifest_bytes: bytes,
        expected_manifest_sha256: str,
    ) -> None:
        if final_dir.is_symlink() or not final_dir.is_dir():
            raise ExportError(
                "name_collision",
                "Candidate publication path is not a regular directory",
                details={"path": str(final_dir)},
            )
        manifest_path = final_dir / MANIFEST_FILENAME
        _require_regular_file(manifest_path, "published manifest")
        actual_manifest_bytes = manifest_path.read_bytes()
        try:
            actual_manifest = json.loads(actual_manifest_bytes)
        except json.JSONDecodeError as exc:
            raise ExportError(
                "name_collision",
                "Existing candidate manifest is invalid JSON",
                details={"path": str(manifest_path)},
            ) from exc
        if not isinstance(actual_manifest, Mapping):
            raise ExportError(
                "name_collision",
                "Existing candidate manifest root is not an object",
            )
        canonical_actual = _canonical_json(actual_manifest)
        actual_manifest_sha256 = hashlib.sha256(canonical_actual).hexdigest()
        if actual_manifest_bytes != canonical_actual:
            raise ExportError(
                "name_collision",
                "Existing candidate manifest is not canonical",
                details={"path": str(manifest_path)},
            )
        _verify_manifest_files(final_dir, actual_manifest)
        if (
            actual_manifest_sha256 != expected_manifest_sha256
            or canonical_actual != expected_manifest_bytes
        ):
            raise ExportError(
                "name_collision",
                "Candidate name already exists with different content",
                details={
                    "path": str(final_dir),
                    "existingManifestSha256": actual_manifest_sha256,
                    "newManifestSha256": expected_manifest_sha256,
                },
            )


def existing_katago_export_request(
    *,
    source_dir: Path,
    destination_root: Path,
    candidate_name: str,
    model_name: str,
    python_executable: str,
    export_script: Path,
    clean_script: Path,
    model_probe_command: Sequence[str] = (),
    command_timeout_seconds: Optional[float] = None,
) -> ExportRequest:
    """Build argv templates for KataGo's existing Python export scripts."""

    return ExportRequest(
        source_dir=source_dir,
        destination_root=destination_root,
        candidate_name=candidate_name,
        model_name=model_name,
        export_command=(
            python_executable,
            str(export_script),
            "-checkpoint",
            "{source_checkpoint}",
            "-export-dir",
            "{partial_dir}",
            "-model-name",
            "{model_name}",
            "-filename-prefix",
            "model",
            "-use-swa",
        ),
        clean_checkpoint_command=(
            python_executable,
            str(clean_script),
            "-checkpoint",
            "{source_checkpoint}",
            "-output",
            "{cleaned_checkpoint}",
        ),
        model_probe_command=tuple(model_probe_command),
        command_timeout_seconds=command_timeout_seconds,
    )


def _request_fingerprint(request: ExportRequest) -> str:
    payload = {
        "exportContract": EXPORT_CONTRACT,
        "modelName": request.model_name,
        "sourceCheckpointName": request.source_checkpoint_name,
        "uncompressedModelName": request.uncompressed_model_name,
        "compressedModelName": request.compressed_model_name,
        "requiredFiles": list(request.required_files),
        "exportCommand": list(request.export_command),
        "cleanCheckpointCommand": list(request.clean_checkpoint_command),
        "modelProbeCommand": list(request.model_probe_command),
        "unsafeAllowUnprobedForTests": request.unsafe_allow_unprobed_for_tests,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _validate_argv(argv: Sequence[str]) -> None:
    if not argv or not all(isinstance(part, str) and part for part in argv):
        raise ExportError(
            "invalid_command",
            "Commands must be non-empty JSON-style argv arrays",
        )


def _validate_leaf_name(value: str, field_name: str) -> None:
    if (
        not value
        or value in (".", "..")
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or value.endswith(".partial")
        or any(ord(character) < 32 for character in value)
    ):
        raise ExportError(
            "invalid_request",
            f"{field_name} must be a safe single path component",
            details={field_name: value},
        )


def _validate_relative_artifact_path(value: str) -> None:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or "\\" in value
        or value == MANIFEST_FILENAME
    ):
        raise ExportError(
            "invalid_request",
            "Required artifact paths must be safe relative paths",
            details={"path": value},
        )


def _require_regular_file(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ExportError(
            "missing_required_file",
            f"{description} is missing or is not a regular file",
            details={"path": str(path)},
        )


def _deterministic_gzip(source: Path, destination: Path) -> None:
    _require_regular_file(source, "uncompressed model")
    with source.open("rb") as input_file, destination.open("xb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_output,
            compresslevel=9,
            mtime=0,
        ) as compressed:
            shutil.copyfileobj(input_file, compressed, length=1024 * 1024)
        raw_output.flush()
        os.fsync(raw_output.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_snapshot(path: Path) -> Optional[Tuple[str, int]]:
    try:
        before = path.stat()
        digest = _sha256_file(path)
        after = path.stat()
    except OSError:
        return None
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        return None
    return digest, after.st_size


def _iter_regular_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for directory, directory_names, file_names in os.walk(root):
        directory_names.sort()
        file_names.sort()
        directory_path = Path(directory)
        for directory_name in directory_names:
            child = directory_path / directory_name
            if child.is_symlink():
                raise ExportError(
                    "invalid_export_output",
                    "Export output contains a symlinked directory",
                    details={"path": str(child)},
                )
        for file_name in file_names:
            path = directory_path / file_name
            if path.name == MANIFEST_FILENAME:
                continue
            _require_regular_file(path, "export artifact")
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _build_file_manifest(root: Path) -> List[Dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in _iter_regular_files(root)
    ]


def _verify_manifest_files(root: Path, manifest: Mapping[str, Any]) -> None:
    files_value = manifest.get("files")
    if not isinstance(files_value, list):
        raise ExportError(
            "name_collision", "Existing manifest files field is not an array"
        )
    expected_paths: List[str] = []
    for entry in files_value:
        if not isinstance(entry, Mapping):
            raise ExportError(
                "name_collision", "Existing manifest has an invalid file entry"
            )
        path_value = entry.get("path")
        sha_value = entry.get("sha256")
        size_value = entry.get("size")
        if (
            not isinstance(path_value, str)
            or not isinstance(sha_value, str)
            or isinstance(size_value, bool)
            or not isinstance(size_value, int)
        ):
            raise ExportError(
                "name_collision", "Existing manifest file metadata is invalid"
            )
        _validate_relative_artifact_path(path_value)
        path = root / path_value
        _require_regular_file(path, "published artifact")
        if path.stat().st_size != size_value or _sha256_file(path) != sha_value:
            raise ExportError(
                "name_collision",
                "Existing publication does not match its manifest",
                details={"path": str(path)},
            )
        expected_paths.append(path_value)

    if expected_paths != sorted(expected_paths) or len(expected_paths) != len(
        set(expected_paths)
    ):
        raise ExportError(
            "name_collision",
            "Existing manifest file entries are not unique and sorted",
        )
    actual_paths = [
        path.relative_to(root).as_posix() for path in _iter_regular_files(root)
    ]
    if actual_paths != expected_paths:
        raise ExportError(
            "name_collision",
            "Existing publication has unmanifested or missing files",
            details={
                "manifestPaths": expected_paths,
                "actualPaths": actual_paths,
            },
        )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new_file(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _fsync_tree(root: Path) -> None:
    directories: List[Path] = []
    for directory, directory_names, file_names in os.walk(root):
        directory_names.sort()
        file_names.sort()
        directory_path = Path(directory)
        directories.append(directory_path)
        for file_name in file_names:
            path = directory_path / file_name
            _require_regular_file(path, "export artifact")
            with path.open("rb") as source:
                os.fsync(source.fileno())
    for directory in reversed(directories):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parse_argv_json(value: Optional[str], option_name: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ExportError(
            "invalid_command",
            f"{option_name} is not valid JSON: {exc}",
        ) from exc
    if not isinstance(parsed, list):
        raise ExportError("invalid_command", f"{option_name} must be a JSON argv array")
    _validate_argv(parsed)
    return tuple(parsed)


def _build_cli_request(args: argparse.Namespace) -> ExportRequest:
    probe_command = _parse_argv_json(
        args.model_probe_command_json, "--model-probe-command-json"
    )
    required_files = tuple(
        dict.fromkeys(DEFAULT_REQUIRED_FILES + tuple(args.required_file or ()))
    )
    if args.export_command_json is not None or args.clean_command_json is not None:
        if args.export_command_json is None or args.clean_command_json is None:
            raise ExportError(
                "invalid_command",
                "--export-command-json and --clean-command-json must be supplied together",
            )
        return ExportRequest(
            source_dir=args.source_dir,
            destination_root=args.destination_root,
            candidate_name=args.candidate_name,
            model_name=args.model_name,
            export_command=_parse_argv_json(
                args.export_command_json, "--export-command-json"
            ),
            clean_checkpoint_command=_parse_argv_json(
                args.clean_command_json, "--clean-command-json"
            ),
            model_probe_command=probe_command,
            required_files=required_files,
            command_timeout_seconds=args.command_timeout_seconds,
            unsafe_allow_unprobed_for_tests=args.unsafe_allow_unprobed_for_tests,
        )

    python_root = Path(__file__).resolve().parent.parent
    export_script = args.export_script or (python_root / "export_model_pytorch.py")
    clean_script = args.clean_script or (python_root / "clean_checkpoint.py")
    request = existing_katago_export_request(
        source_dir=args.source_dir,
        destination_root=args.destination_root,
        candidate_name=args.candidate_name,
        model_name=args.model_name,
        python_executable=args.python_executable,
        export_script=export_script,
        clean_script=clean_script,
        model_probe_command=probe_command,
        command_timeout_seconds=args.command_timeout_seconds,
    )
    return dataclasses.replace(
        request,
        required_files=required_files,
        unsafe_allow_unprobed_for_tests=args.unsafe_allow_unprobed_for_tests,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--destination-root", required=True, type=Path)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--export-script", type=Path)
    parser.add_argument("--clean-script", type=Path)
    parser.add_argument("--export-command-json")
    parser.add_argument("--clean-command-json")
    parser.add_argument("--model-probe-command-json")
    parser.add_argument(
        "--unsafe-allow-unprobed-for-tests",
        action="store_true",
        help="TEST ONLY: permit publication without a model probe",
    )
    parser.add_argument("--required-file", action="append")
    parser.add_argument("--command-timeout-seconds", type=float)
    args = parser.parse_args(argv)

    try:
        request = _build_cli_request(args)
        result = HardenedExporter().publish(request)
        sys.stdout.buffer.write(_canonical_json(result.to_dict()))
        return 0
    except ExportError as exc:
        sys.stderr.buffer.write(_canonical_json(exc.to_dict()))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
