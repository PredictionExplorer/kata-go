import os
from pathlib import Path

import pytest
from katago.train.extreme_score_policy import load_extreme_score_training_policy
from risk_score.extreme_score_controller import (
    bootstrap,
    initialize_accepted_state,
    load_accepted_state,
    reconcile,
)
from risk_score.extreme_score_evaluator import file_sha256
from risk_score.extreme_score_league import DEFAULT_POLICY_PATH


def _fixture(tmp_path, *, decision):
    reference = tmp_path / "reference.bin.gz"
    candidate = tmp_path / "candidate.bin.gz"
    checkpoint = tmp_path / "checkpoint.ckpt"
    plan_path = tmp_path / "plan.json"
    report_path = tmp_path / "report.json"
    run_spec_path = tmp_path / "run-spec.json"
    attestation_path = tmp_path / "attestation.json"
    progress_path = tmp_path / "progress.json"
    reference.write_bytes(b"reference")
    candidate.write_bytes(b"candidate")
    checkpoint.write_bytes(b"checkpoint")
    plan_path.write_bytes(b"plan")
    report_path.write_bytes(b"report")
    run_spec_path.write_bytes(b"run-spec")
    attestation_path.write_bytes(b"attestation")
    progress_path.write_bytes(b"progress")
    reference_model = {
        "model_id": "reference-v1",
        "sha256": file_sha256(reference),
    }
    candidate_model = {
        "model_id": "candidate-v1",
        "sha256": file_sha256(candidate),
    }
    plan = {
        "plan_sha256": "1" * 64,
        "reference_model": reference_model,
        "candidate_model": candidate_model,
    }
    rollback = {
        "action": "retain_reference",
        "reference_model": reference_model,
        "reference_model_artifact": {
            "path": str(reference.resolve()),
            "file_sha256": file_sha256(reference),
        },
        "trainer_checkpoint_artifact": {
            "path": str(checkpoint.resolve()),
            "file_sha256": file_sha256(checkpoint),
        },
        "quarantine_candidate_on_failure": True,
    }
    plan["rollback_recommendation"] = rollback
    report = {
        "plan_binding": {
            "path": str(plan_path.resolve()),
            "file_sha256": file_sha256(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "execution_matrix_sha256": "2" * 64,
        },
        "decision": decision,
        "promotion_recommended": decision == "PASS",
        "rollback_recommendation": rollback,
        "report_sha256": "3" * 64 if decision == "PASS" else "4" * 64,
    }
    run_spec = {
        "focal_models": [
            {"path": str(reference.resolve()), "sha256": file_sha256(reference)},
            {"path": str(candidate.resolve()), "sha256": file_sha256(candidate)},
        ]
    }
    progress = {
        "schema_version": 1,
        "contract": "risk-score-extreme-training-progress-v1",
        "checkpoint": rollback["trainer_checkpoint_artifact"],
        "training_policy": load_extreme_score_training_policy(DEFAULT_POLICY_PATH, 1),
        "cohort_size": 1,
        "selected_training_samples": 10,
        "shuffle_provenance_sha256": "6" * 64,
        "progress_sha256": "7" * 64,
    }
    return {
        "plan_path": plan_path,
        "report_path": report_path,
        "run_spec_path": run_spec_path,
        "attestation_path": attestation_path,
        "progress_path": progress_path,
        "plan": plan,
        "report": report,
        "run_spec": run_spec,
        "progress": progress,
        "checkpoint": checkpoint,
    }


def _loaders(data):
    return {
        "plan_loader": lambda _path: data["plan"],
        "report_loader": lambda _path: data["report"],
        "run_spec_loader": lambda _path: data["run_spec"],
        "progress_loader": lambda _path: data["progress"],
        "attestation_verifier": lambda **_kwargs: {"attestation_sha256": "5" * 64},
    }


def test_pass_atomically_activates_candidate_and_is_restart_safe(tmp_path):
    data = _fixture(tmp_path, decision="PASS")
    root = tmp_path / "state-root"
    state_path = root / "accepted-current.json"
    lock_path = root / "controller.lock"
    loaders = _loaders(data)

    initial = bootstrap(
        plan_path=data["plan_path"],
        run_spec_path=data["run_spec_path"],
        state_root=root,
        state_path=state_path,
        lock_path=lock_path,
        training_policy_path=DEFAULT_POLICY_PATH,
        plan_loader=loaders["plan_loader"],
        run_spec_loader=loaders["run_spec_loader"],
    )
    assert initial["model"] == data["plan"]["reference_model"]

    result = reconcile(
        plan_path=data["plan_path"],
        report_path=data["report_path"],
        attestation_path=data["attestation_path"],
        training_progress_path=data["progress_path"],
        run_spec_path=data["run_spec_path"],
        state_root=root,
        state_path=state_path,
        lock_path=lock_path,
        **loaders,
    )
    assert result["promotion_applied"] is True
    assert result["accepted_state"]["model"] == data["plan"]["candidate_model"]
    assert (
        Path(result["accepted_state"]["artifact"]["path"]).stat().st_mode & 0o222 == 0
    )
    assert Path(result["transaction_path"]).stat().st_mode & 0o222 == 0
    assert Path(result["completion_path"]).stat().st_mode & 0o222 == 0
    rollback_checkpoint = Path(
        result["accepted_state"]["rollback"]["trainer_checkpoint_artifact"]["path"]
    )
    assert rollback_checkpoint != data["checkpoint"]
    assert rollback_checkpoint.stat().st_mode & 0o222 == 0

    resumed = reconcile(
        plan_path=data["plan_path"],
        report_path=data["report_path"],
        attestation_path=data["attestation_path"],
        training_progress_path=data["progress_path"],
        run_spec_path=data["run_spec_path"],
        state_root=root,
        state_path=state_path,
        lock_path=lock_path,
        **loaders,
    )
    assert resumed == result
    data["checkpoint"].unlink()
    assert load_accepted_state(state_path) == result["accepted_state"]


def test_nonpass_preserves_reference_but_records_decision(tmp_path):
    data = _fixture(tmp_path, decision="INCONCLUSIVE")
    root = tmp_path / "state-root"
    state_path = root / "accepted-current.json"
    lock_path = root / "controller.lock"
    loaders = _loaders(data)
    bootstrap(
        plan_path=data["plan_path"],
        run_spec_path=data["run_spec_path"],
        state_root=root,
        state_path=state_path,
        lock_path=lock_path,
        training_policy_path=DEFAULT_POLICY_PATH,
        plan_loader=loaders["plan_loader"],
        run_spec_loader=loaders["run_spec_loader"],
    )
    result = reconcile(
        plan_path=data["plan_path"],
        report_path=data["report_path"],
        attestation_path=data["attestation_path"],
        training_progress_path=data["progress_path"],
        run_spec_path=data["run_spec_path"],
        state_root=root,
        state_path=state_path,
        lock_path=lock_path,
        **loaders,
    )
    assert result["promotion_applied"] is False
    assert result["accepted_state"]["model"] == data["plan"]["reference_model"]


def test_controller_rejects_report_without_valid_execution_attestation(tmp_path):
    data = _fixture(tmp_path, decision="PASS")
    root = tmp_path / "state-root"
    state_path = root / "accepted-current.json"
    lock_path = root / "controller.lock"
    loaders = _loaders(data)
    bootstrap(
        plan_path=data["plan_path"],
        run_spec_path=data["run_spec_path"],
        state_root=root,
        state_path=state_path,
        lock_path=lock_path,
        training_policy_path=DEFAULT_POLICY_PATH,
        plan_loader=loaders["plan_loader"],
        run_spec_loader=loaders["run_spec_loader"],
    )
    loaders["attestation_verifier"] = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("fabricated rows have no runner receipts")
    )
    with pytest.raises(RuntimeError, match="no runner receipts"):
        reconcile(
            plan_path=data["plan_path"],
            report_path=data["report_path"],
            attestation_path=data["attestation_path"],
            training_progress_path=data["progress_path"],
            run_spec_path=data["run_spec_path"],
            state_root=root,
            state_path=state_path,
            lock_path=lock_path,
            **loaders,
        )


def test_progress_replacement_during_authentication_is_rejected(tmp_path):
    data = _fixture(tmp_path, decision="PASS")
    root = tmp_path / "state-root"
    state_path = root / "accepted-current.json"
    lock_path = root / "controller.lock"
    loaders = _loaders(data)
    bootstrap(
        plan_path=data["plan_path"],
        run_spec_path=data["run_spec_path"],
        state_root=root,
        state_path=state_path,
        lock_path=lock_path,
        training_policy_path=DEFAULT_POLICY_PATH,
        plan_loader=loaders["plan_loader"],
        run_spec_loader=loaders["run_spec_loader"],
    )

    def replacing_loader(path):
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(b"replacement-progress")
        os.replace(replacement, path)
        return data["progress"]

    loaders["progress_loader"] = replacing_loader
    with pytest.raises(
        RuntimeError, match="training progress changed while being authenticated"
    ):
        reconcile(
            plan_path=data["plan_path"],
            report_path=data["report_path"],
            attestation_path=data["attestation_path"],
            training_progress_path=data["progress_path"],
            run_spec_path=data["run_spec_path"],
            state_root=root,
            state_path=state_path,
            lock_path=lock_path,
            **loaders,
        )


@pytest.mark.parametrize("symlink_name", ["models", "checkpoints"])
def test_controller_snapshot_directories_cannot_escape_state_root(
    tmp_path, symlink_name
):
    model = tmp_path / "model.bin.gz"
    checkpoint = tmp_path / "checkpoint.ckpt"
    model.write_bytes(b"model")
    checkpoint.write_bytes(b"checkpoint")
    root = tmp_path / "state-root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / symlink_name).symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="unsafe|symlink"):
        initialize_accepted_state(
            model_id="initial",
            model_path=model,
            checkpoint_path=checkpoint,
            training_policy_path=DEFAULT_POLICY_PATH,
            state_root=root,
            state_path=root / "accepted-current.json",
            lock_path=root / "controller.lock",
        )
    assert not list(outside.rglob("*"))
