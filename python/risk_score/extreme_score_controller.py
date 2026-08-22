"""Atomically activate evaluator-approved extreme-score model snapshots."""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from katago.train.extreme_score_policy import load_extreme_score_training_policy

from risk_score.extreme_score_evaluation_run import (
    load_spec,
    verify_execution_attestation,
)
from risk_score.extreme_score_evaluator import (
    canonical_json,
    canonical_sha256,
    file_sha256,
    load_plan,
    load_report,
)
from risk_score.extreme_score_progress import load_training_progress

STATE_CONTRACT = "risk-score-extreme-accepted-model-state-v1"
TRANSACTION_CONTRACT = "risk-score-extreme-promotion-transaction-v1"
COMPLETION_CONTRACT = "risk-score-extreme-promotion-completion-v1"
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class ExtremeScoreControllerError(RuntimeError):
    """Promotion state or evaluator evidence is unsafe or inconsistent."""


def _regular_file(path: Path, role: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ExtremeScoreControllerError(
            f"{role} must be a regular non-symlink file: {source}"
        )
    return source.resolve()


def _read_only_file(path: Path, role: str) -> Path:
    source = _regular_file(path, role)
    if source.stat().st_mode & _WRITE_BITS:
        raise ExtremeScoreControllerError(f"{role} must be read-only: {source}")
    return source


def _safe_directory(path: Path, role: str) -> Path:
    target = Path(path)
    if not target.is_absolute() or target != Path(os.path.abspath(target)):
        raise ExtremeScoreControllerError(f"{role} must be absolute and normalized")
    current = Path(target.anchor)
    for part in target.parts[1:]:
        current = current / part
        if current.exists() or current.is_symlink():
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ExtremeScoreControllerError(
                    f"{role} contains a symlink or non-directory: {current}"
                )
        else:
            current.mkdir()
    return target


def _safe_subdirectory(root: Path, *parts: str) -> Path:
    current = _safe_directory(root, "controller state root")
    for part in parts:
        if not part or part in {".", ".."} or "/" in part:
            raise ExtremeScoreControllerError(
                "controller snapshot path component is unsafe"
            )
        current = current / part
        if current.exists() or current.is_symlink():
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ExtremeScoreControllerError(
                    f"controller snapshot directory is unsafe: {current}"
                )
        else:
            current.mkdir()
    return current


def _require_within(path: Path, root: Path, role: str) -> Path:
    target = Path(os.path.abspath(path))
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ExtremeScoreControllerError(
            f"{role} must be inside controller state root"
        ) from exc
    return target


def _safe_file_within(path: Path, root: Path, role: str) -> Path:
    target = _require_within(path, root, role)
    relative_parent = target.parent.relative_to(root)
    if relative_parent.parts:
        _safe_subdirectory(root, *relative_parent.parts)
    return target


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    data = (canonical_json(value) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ExtremeScoreControllerError("immutable artifact parent is unsafe")
    if target.exists() or target.is_symlink():
        source = _read_only_file(target, "immutable controller artifact")
        if source.read_bytes() != data:
            raise ExtremeScoreControllerError(
                f"immutable controller artifact conflicts: {target}"
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
        try:
            os.link(temporary, target)
        except FileExistsError:
            source = _read_only_file(target, "immutable controller artifact")
            if source.read_bytes() != data:
                raise ExtremeScoreControllerError(
                    f"immutable controller publication raced: {target}"
                )
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_projection(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    data = (canonical_json(value) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ExtremeScoreControllerError("accepted-state parent is unsafe")
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
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _load_state(path: Path) -> dict[str, Any]:
    source = _read_only_file(path, "accepted model state")
    try:
        import json

        value = json.loads(source.read_bytes())
    except (UnicodeDecodeError, ValueError) as exc:
        raise ExtremeScoreControllerError(
            f"accepted model state is invalid: {exc}"
        ) from exc
    expected = {
        "schema_version",
        "contract",
        "model",
        "artifact",
        "source_plan_sha256",
        "source_report_sha256",
        "rollback",
        "training_progress",
        "state_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ExtremeScoreControllerError("accepted model state keys differ")
    supplied = value["state_sha256"]
    payload = dict(value)
    payload.pop("state_sha256")
    if (
        value["schema_version"] != 1
        or value["contract"] != STATE_CONTRACT
        or supplied != canonical_sha256(payload)
    ):
        raise ExtremeScoreControllerError("accepted model state self-hash is invalid")
    artifact = value["artifact"]
    if not isinstance(artifact, Mapping) or set(artifact) != {"path", "file_sha256"}:
        raise ExtremeScoreControllerError(
            "accepted model artifact binding is malformed"
        )
    source_artifact = _read_only_file(Path(artifact["path"]), "accepted model artifact")
    if file_sha256(source_artifact) != artifact["file_sha256"]:
        raise ExtremeScoreControllerError("accepted model artifact changed")
    progress = value["training_progress"]
    progress_keys = {
        "schema_version",
        "contract",
        "training_policy",
        "cohort_size",
        "selected_training_samples",
        "checkpoint",
        "source_progress_path",
        "source_progress_file_sha256",
        "source_progress_sha256",
        "accepted_progress_sha256",
    }
    if (
        not isinstance(progress, Mapping)
        or set(progress) != progress_keys
        or progress.get("schema_version") != 1
        or progress.get("contract")
        != "risk-score-extreme-accepted-training-progress-v1"
    ):
        raise ExtremeScoreControllerError("accepted training progress is malformed")
    progress_payload = dict(progress)
    progress_hash = progress_payload.pop("accepted_progress_sha256")
    if progress_hash != canonical_sha256(progress_payload):
        raise ExtremeScoreControllerError(
            "accepted training progress self-hash is invalid"
        )
    checkpoint = progress.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "path",
        "file_sha256",
    }:
        raise ExtremeScoreControllerError("accepted checkpoint binding is malformed")
    checkpoint_path = _read_only_file(
        Path(checkpoint["path"]), "accepted trainer checkpoint"
    )
    if file_sha256(checkpoint_path) != checkpoint["file_sha256"]:
        raise ExtremeScoreControllerError("accepted trainer checkpoint changed")
    return value


def load_accepted_state(path: Path) -> dict[str, Any]:
    """Load and revalidate the authoritative accepted model and checkpoint."""
    return _load_state(path)


def _snapshot_model(source: Path, digest: str, state_root: Path) -> dict[str, str]:
    source_file = _regular_file(source, "model artifact")
    if file_sha256(source_file) != digest:
        raise ExtremeScoreControllerError("model artifact hash is invalid")
    destination = _safe_subdirectory(state_root, "models", digest) / "model.bin.gz"
    if destination.exists() or destination.is_symlink():
        target = _read_only_file(destination, "model snapshot")
        if file_sha256(target) != digest:
            raise ExtremeScoreControllerError("model snapshot conflicts")
        return {"path": str(target), "file_sha256": digest}
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise ExtremeScoreControllerError("model snapshot parent is unsafe")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".model.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target, source_file.open("rb") as handle:
            shutil.copyfileobj(handle, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if file_sha256(temporary) != digest:
            raise ExtremeScoreControllerError("model snapshot copy hash mismatch")
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            target = _read_only_file(destination, "model snapshot")
            if file_sha256(target) != digest:
                raise ExtremeScoreControllerError("model snapshot publication raced")
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination.resolve()), "file_sha256": digest}


def _snapshot_checkpoint(source: Path, digest: str, state_root: Path) -> dict[str, str]:
    source_file = _regular_file(source, "trainer checkpoint")
    if file_sha256(source_file) != digest:
        raise ExtremeScoreControllerError("trainer checkpoint hash is invalid")
    destination = (
        _safe_subdirectory(state_root, "checkpoints", digest) / "checkpoint.ckpt"
    )
    if destination.exists() or destination.is_symlink():
        target = _read_only_file(destination, "trainer checkpoint snapshot")
        if file_sha256(target) != digest:
            raise ExtremeScoreControllerError("trainer checkpoint snapshot conflicts")
        return {"path": str(target), "file_sha256": digest}
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".checkpoint.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target, source_file.open("rb") as handle:
            shutil.copyfileobj(handle, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if file_sha256(temporary) != digest:
            raise ExtremeScoreControllerError(
                "trainer checkpoint snapshot copy hash mismatch"
            )
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            target = _read_only_file(destination, "trainer checkpoint snapshot")
            if file_sha256(target) != digest:
                raise ExtremeScoreControllerError(
                    "trainer checkpoint snapshot publication raced"
                )
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination.resolve()), "file_sha256": digest}


def _model_source(run_spec: Mapping[str, Any], digest: str) -> Path:
    matches = [
        Path(item["path"])
        for item in run_spec["focal_models"]
        if item.get("sha256") == digest
    ]
    if len(matches) != 1:
        raise ExtremeScoreControllerError(
            f"evaluation run spec does not bind model {digest}"
        )
    return matches[0]


def _preserve_rollback(rollback: Mapping[str, Any], state_root: Path) -> dict[str, Any]:
    reference = rollback["reference_model_artifact"]
    checkpoint = rollback["trainer_checkpoint_artifact"]
    preserved = dict(rollback)
    preserved["reference_model_artifact"] = _snapshot_model(
        Path(reference["path"]), reference["file_sha256"], state_root
    )
    preserved["trainer_checkpoint_artifact"] = _snapshot_checkpoint(
        Path(checkpoint["path"]), checkpoint["file_sha256"], state_root
    )
    return preserved


def _accepted_progress(
    *,
    training_policy: Mapping[str, Any],
    cohort_size: int,
    selected_training_samples: int,
    checkpoint: Mapping[str, str],
    source_progress: Mapping[str, Any] | None,
    source_progress_path: Path | None,
    source_progress_file_sha256: str | None = None,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract": "risk-score-extreme-accepted-training-progress-v1",
        "training_policy": dict(training_policy),
        "cohort_size": cohort_size,
        "selected_training_samples": selected_training_samples,
        "checkpoint": dict(checkpoint),
        "source_progress_path": (
            str(Path(source_progress_path).resolve())
            if source_progress_path is not None
            else None
        ),
        "source_progress_file_sha256": (
            source_progress_file_sha256 if source_progress_path is not None else None
        ),
        "source_progress_sha256": (
            source_progress["progress_sha256"] if source_progress is not None else None
        ),
    }
    value["accepted_progress_sha256"] = canonical_sha256(value)
    return value


def _state(
    *,
    model: Mapping[str, Any],
    artifact: Mapping[str, str],
    plan_sha256: str | None,
    report_sha256: str | None,
    rollback: Mapping[str, Any] | None,
    training_progress: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract": STATE_CONTRACT,
        "model": dict(model),
        "artifact": dict(artifact),
        "source_plan_sha256": plan_sha256,
        "source_report_sha256": report_sha256,
        "rollback": dict(rollback) if rollback is not None else None,
        "training_progress": dict(training_progress),
    }
    value["state_sha256"] = canonical_sha256(value)
    return value


@contextmanager
def _exclusive_lock(path: Path):
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def bootstrap(
    *,
    plan_path: Path,
    run_spec_path: Path,
    state_root: Path,
    state_path: Path,
    lock_path: Path,
    training_policy_path: Path,
    plan_loader: Callable[[Path], Mapping[str, Any]] = load_plan,
    run_spec_loader: Callable[[Path], Mapping[str, Any]] = load_spec,
) -> dict[str, Any]:
    plan = dict(plan_loader(plan_path))
    run_spec = dict(run_spec_loader(run_spec_path))
    reference = plan["reference_model"]
    root = _safe_directory(Path(os.path.abspath(state_root)), "controller state root")
    state_path = _safe_file_within(state_path, root, "accepted state")
    lock_path = _safe_file_within(lock_path, root, "controller lock")
    artifact = _snapshot_model(
        _model_source(run_spec, reference["sha256"]),
        reference["sha256"],
        root,
    )
    rollback = _preserve_rollback(plan["rollback_recommendation"], root)
    policy = load_extreme_score_training_policy(training_policy_path, 1)
    progress = _accepted_progress(
        training_policy=policy,
        cohort_size=1,
        selected_training_samples=0,
        checkpoint=rollback["trainer_checkpoint_artifact"],
        source_progress=None,
        source_progress_path=None,
    )
    desired = _state(
        model=reference,
        artifact=artifact,
        plan_sha256=plan["plan_sha256"],
        report_sha256=None,
        rollback=rollback,
        training_progress=progress,
    )
    with _exclusive_lock(lock_path):
        if Path(state_path).exists() or Path(state_path).is_symlink():
            existing = _load_state(state_path)
            if existing != desired:
                raise ExtremeScoreControllerError(
                    "accepted state already exists with another identity"
                )
        else:
            _replace_projection(state_path, desired)
    return desired


def initialize_accepted_state(
    *,
    model_id: str,
    model_path: Path,
    checkpoint_path: Path,
    training_policy_path: Path,
    state_root: Path,
    state_path: Path,
    lock_path: Path,
    selected_training_samples: int = 0,
) -> dict[str, Any]:
    """Initialize the original immutable model before the first league round."""
    if not isinstance(model_id, str) or not model_id or model_id != model_id.strip():
        raise ExtremeScoreControllerError("initial model_id is invalid")
    if type(selected_training_samples) is not int or selected_training_samples < 0:
        raise ExtremeScoreControllerError(
            "initial selected_training_samples must be nonnegative"
        )
    model_source = _regular_file(model_path, "initial model")
    model = {"model_id": model_id, "sha256": file_sha256(model_source)}
    checkpoint_source = _regular_file(checkpoint_path, "initial checkpoint")
    checkpoint_digest = file_sha256(checkpoint_source)
    root = _safe_directory(Path(os.path.abspath(state_root)), "controller state root")
    state_path = _safe_file_within(state_path, root, "accepted state")
    lock_path = _safe_file_within(lock_path, root, "controller lock")
    artifact = _snapshot_model(model_source, model["sha256"], root)
    checkpoint = _snapshot_checkpoint(checkpoint_source, checkpoint_digest, root)
    policy = load_extreme_score_training_policy(training_policy_path, 1)
    progress = _accepted_progress(
        training_policy=policy,
        cohort_size=1,
        selected_training_samples=selected_training_samples,
        checkpoint=checkpoint,
        source_progress=None,
        source_progress_path=None,
    )
    desired = _state(
        model=model,
        artifact=artifact,
        plan_sha256=None,
        report_sha256=None,
        rollback={
            "action": "retain_reference",
            "reference_model": model,
            "reference_model_artifact": artifact,
            "trainer_checkpoint_artifact": checkpoint,
            "quarantine_candidate_on_failure": True,
        },
        training_progress=progress,
    )
    with _exclusive_lock(lock_path):
        if Path(state_path).exists() or Path(state_path).is_symlink():
            if _load_state(state_path) != desired:
                raise ExtremeScoreControllerError(
                    "accepted state already has another initial model"
                )
        else:
            _replace_projection(state_path, desired)
    return desired


def reconcile(
    *,
    plan_path: Path,
    report_path: Path,
    attestation_path: Path,
    run_spec_path: Path,
    training_progress_path: Path,
    state_root: Path,
    state_path: Path,
    lock_path: Path,
    plan_loader: Callable[[Path], Mapping[str, Any]] = load_plan,
    report_loader: Callable[[Path], Mapping[str, Any]] = load_report,
    run_spec_loader: Callable[[Path], Mapping[str, Any]] = load_spec,
    progress_loader: Callable[[Path], Mapping[str, Any]] = load_training_progress,
    attestation_verifier: Callable[..., Mapping[str, Any]] = (
        verify_execution_attestation
    ),
) -> dict[str, Any]:
    plan = dict(plan_loader(plan_path))
    report = dict(report_loader(report_path))
    run_spec = dict(run_spec_loader(run_spec_path))
    progress_source = _regular_file(training_progress_path, "training progress")
    progress_before = progress_source.lstat()
    training_progress = dict(progress_loader(progress_source))
    progress_file_sha256 = file_sha256(progress_source)
    progress_after = progress_source.lstat()
    progress_identity_before = (
        progress_before.st_dev,
        progress_before.st_ino,
        progress_before.st_size,
        progress_before.st_mtime_ns,
        progress_before.st_ctime_ns,
    )
    progress_identity_after = (
        progress_after.st_dev,
        progress_after.st_ino,
        progress_after.st_size,
        progress_after.st_mtime_ns,
        progress_after.st_ctime_ns,
    )
    if progress_identity_before != progress_identity_after:
        raise ExtremeScoreControllerError(
            "training progress changed while being authenticated"
        )
    if report["plan_binding"]["plan_sha256"] != plan["plan_sha256"] or report[
        "plan_binding"
    ]["file_sha256"] != file_sha256(plan_path):
        raise ExtremeScoreControllerError("report is bound to another plan")
    attestation = dict(
        attestation_verifier(
            attestation_path=attestation_path,
            spec_path=run_spec_path,
            plan_path=plan_path,
            report_path=report_path,
            plan=plan,
            report=report,
        )
    )
    reference = plan["reference_model"]
    candidate = plan["candidate_model"]
    promote = report["decision"] == "PASS" and report["promotion_recommended"] is True
    if report["promotion_recommended"] is not promote:
        raise ExtremeScoreControllerError("report promotion decision is inconsistent")
    selected = candidate if promote else reference
    root = _safe_directory(Path(os.path.abspath(state_root)), "controller state root")
    state_path = _safe_file_within(state_path, root, "accepted state")
    lock_path = _safe_file_within(lock_path, root, "controller lock")
    artifact = _snapshot_model(
        _model_source(run_spec, selected["sha256"]),
        selected["sha256"],
        root,
    )
    report_rollback = report["rollback_recommendation"]
    if (
        training_progress["checkpoint"]["file_sha256"]
        != report_rollback["trainer_checkpoint_artifact"]["file_sha256"]
    ):
        raise ExtremeScoreControllerError(
            "training progress checkpoint differs from report rollback"
        )
    rollback = _preserve_rollback(report_rollback, root)
    preserved_progress = _accepted_progress(
        training_policy=training_progress["training_policy"],
        cohort_size=training_progress["cohort_size"],
        selected_training_samples=training_progress["selected_training_samples"],
        checkpoint=rollback["trainer_checkpoint_artifact"],
        source_progress=training_progress,
        source_progress_path=progress_source,
        source_progress_file_sha256=progress_file_sha256,
    )
    desired = _state(
        model=selected,
        artifact=artifact,
        plan_sha256=plan["plan_sha256"],
        report_sha256=report["report_sha256"],
        rollback=rollback,
        training_progress=preserved_progress,
    )
    transaction = {
        "schema_version": 1,
        "contract": TRANSACTION_CONTRACT,
        "plan_path": str(Path(plan_path).resolve()),
        "plan_file_sha256": file_sha256(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "report_path": str(Path(report_path).resolve()),
        "report_file_sha256": file_sha256(report_path),
        "report_sha256": report["report_sha256"],
        "execution_attestation_path": str(Path(attestation_path).resolve()),
        "execution_attestation_file_sha256": file_sha256(attestation_path),
        "execution_attestation_sha256": attestation["attestation_sha256"],
        "decision": report["decision"],
        "promotion_recommended": promote,
        "before_model": reference,
        "after_state": desired,
    }
    transaction["transaction_sha256"] = canonical_sha256(transaction)
    transaction_path = (
        root / "transactions" / f"{transaction['transaction_sha256']}.json"
    )
    completion = {
        "schema_version": 1,
        "contract": COMPLETION_CONTRACT,
        "transaction_sha256": transaction["transaction_sha256"],
        "accepted_state_sha256": desired["state_sha256"],
    }
    completion["completion_sha256"] = canonical_sha256(completion)
    completion_path = root / "completions" / f"{transaction['transaction_sha256']}.json"

    with _exclusive_lock(lock_path):
        current = _load_state(state_path)
        current_hash = current["model"]["sha256"]
        if current_hash not in {reference["sha256"], selected["sha256"]}:
            raise ExtremeScoreControllerError(
                "accepted model is neither the plan reference nor retry target"
            )
        current_progress = current["training_progress"]
        if (
            current_progress["training_policy"]["file_sha256"]
            != training_progress["training_policy"]["file_sha256"]
        ):
            raise ExtremeScoreControllerError(
                "training progress policy differs from accepted state"
            )
        if (
            training_progress["selected_training_samples"]
            < current_progress["selected_training_samples"]
        ):
            raise ExtremeScoreControllerError(
                "training progress selected-sample count moved backward"
            )
        _publish_immutable_json(transaction_path, transaction)
        if current != desired:
            _replace_projection(state_path, desired)
        verified = _load_state(state_path)
        if verified != desired:
            raise ExtremeScoreControllerError(
                "accepted model projection failed verification"
            )
        _publish_immutable_json(completion_path, completion)
    return {
        "decision": report["decision"],
        "promotion_applied": promote,
        "accepted_state": desired,
        "transaction_path": str(transaction_path),
        "completion_path": str(completion_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--model-id", required=True)
    initialize.add_argument("--model", required=True, type=Path)
    initialize.add_argument("--checkpoint", required=True, type=Path)
    initialize.add_argument("--training-policy", required=True, type=Path)
    initialize.add_argument("--state-root", required=True, type=Path)
    initialize.add_argument("--state", required=True, type=Path)
    initialize.add_argument("--lock", required=True, type=Path)
    for name in ("bootstrap", "reconcile"):
        command = subparsers.add_parser(name)
        command.add_argument("--plan", required=True, type=Path)
        command.add_argument("--run-spec", required=True, type=Path)
        command.add_argument("--state-root", required=True, type=Path)
        command.add_argument("--state", required=True, type=Path)
        command.add_argument("--lock", required=True, type=Path)
        if name == "bootstrap":
            command.add_argument("--training-policy", required=True, type=Path)
        else:
            command.add_argument("--report", required=True, type=Path)
            command.add_argument("--attestation", required=True, type=Path)
            command.add_argument("--training-progress", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "initialize":
            value = initialize_accepted_state(
                model_id=args.model_id,
                model_path=args.model,
                checkpoint_path=args.checkpoint,
                training_policy_path=args.training_policy,
                state_root=args.state_root,
                state_path=args.state,
                lock_path=args.lock,
            )
        elif args.command == "bootstrap":
            value = bootstrap(
                plan_path=args.plan,
                run_spec_path=args.run_spec,
                state_root=args.state_root,
                state_path=args.state,
                lock_path=args.lock,
                training_policy_path=args.training_policy,
            )
        else:
            value = reconcile(
                plan_path=args.plan,
                report_path=args.report,
                attestation_path=args.attestation,
                run_spec_path=args.run_spec,
                training_progress_path=args.training_progress,
                state_root=args.state_root,
                state_path=args.state,
                lock_path=args.lock,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            canonical_json(
                {"error": {"type": type(exc).__name__, "message": str(exc)}}
            ),
            file=sys.stderr,
        )
        return 2
    print(canonical_json(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
