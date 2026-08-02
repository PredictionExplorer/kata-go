"""Pure helpers for production training cadence and validation telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


VALIDATION_MANIFEST_CONTRACT = "katago-fixed-validation-manifest-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(1024 * 1024)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validation_inventory(directory: Path):
    root = Path(directory)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("fixed validation directory must be absolute and non-symlink")
    nested = [
        path
        for path in root.rglob("*.npz")
        if path.parent != root
    ]
    if nested:
        raise ValueError(
            f"fixed validation NPZ files must be top-level: {nested[0]}"
        )
    files = []
    for path in sorted(root.glob("*.npz")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"fixed validation input is not a regular file: {path}")
        metadata = path.stat()
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": metadata.st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise ValueError("fixed validation directory contains no NPZ files")
    return files


def build_validation_manifest(directory: Path, output: Path) -> Mapping[str, Any]:
    source = Path(directory)
    files = _validation_inventory(source)
    root = source.resolve()
    target = Path(output)
    if not target.is_absolute() or target.is_symlink():
        raise ValueError("fixed validation manifest path must be absolute and non-symlink")
    manifest = {
        "schema_version": 1,
        "contract": VALIDATION_MANIFEST_CONTRACT,
        "directory": str(root),
        "files": files,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    data = (_canonical_json(manifest) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.read_bytes() != data:
            raise ValueError("existing fixed validation manifest conflicts")
        return manifest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != data:
                raise ValueError("fixed validation manifest publication raced")
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def validate_validation_manifest(
    directory: Path, manifest_path: Path
) -> Mapping[str, Any]:
    directory_path = Path(directory)
    inventory = _validation_inventory(directory_path)
    root = directory_path.resolve()
    source = Path(manifest_path)
    if (
        not source.is_absolute()
        or source.is_symlink()
        or not source.is_file()
    ):
        raise ValueError("fixed validation manifest must be an absolute regular file")
    data = source.read_bytes()
    try:
        manifest = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"fixed validation manifest is invalid: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("fixed validation manifest root must be an object")
    payload = dict(manifest)
    supplied_hash = payload.pop("manifest_sha256", None)
    if (
        data != (_canonical_json(manifest) + "\n").encode("utf-8")
        or manifest.get("schema_version") != 1
        or manifest.get("contract") != VALIDATION_MANIFEST_CONTRACT
        or manifest.get("directory") != str(root)
        or supplied_hash
        != hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        or manifest.get("files") != inventory
    ):
        raise ValueError("fixed validation inventory or manifest changed")
    return manifest


def export_interval_ready(
    global_step_samples, last_exported_samples, minimum_interval
):
    if minimum_interval is None:
        return True
    return global_step_samples - last_exported_samples >= minimum_interval


def validation_data_dir(current_data_dir, fixed_validation_dir):
    if fixed_validation_dir is not None:
        return fixed_validation_dir
    return os.path.join(current_data_dir, "val")


def add_validation_telemetry(
    metric_sums,
    metric_weights,
    *,
    global_step_samples,
    validation_samples,
    validation_batches,
    validation_wall_seconds,
    running_metrics,
):
    for key, value in (
        ("global_step_samples", global_step_samples),
        ("val_samples", validation_samples),
        ("val_batches", validation_batches),
        ("val_wall_seconds", validation_wall_seconds),
    ):
        metric_sums[key] = float(value)
        metric_weights[key] = 1.0
    for key in ("nsamp", "wsum"):
        if key in running_metrics["sums"] and key in running_metrics["weights"]:
            metric_sums[f"{key}_train"] = running_metrics["sums"][key]
            metric_weights[f"{key}_train"] = running_metrics["weights"][key]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-validation")
    freeze.add_argument("--directory", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_validation_manifest(args.directory, args.output)
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(_canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
