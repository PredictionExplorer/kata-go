import json
import subprocess
from pathlib import Path

from risk_score.promotion_preflight import (
    bootstrap_backpressure,
    candidate_inventory,
    deployment_snapshot,
    filesystem_test,
)


def test_filesystem_test_proves_atomic_inode_preserving_rename(tmp_path):
    result = filesystem_test(tmp_path.resolve())
    assert result["atomic_rename_preserved_inode"] is True
    assert result["directory_fsync_succeeded"] is True
    assert not list(tmp_path.glob(".promotion-fs-test-*"))


def test_deployment_snapshot_binds_source_and_live_names(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "file").write_text("value", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "file"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "-m",
            "test",
        ],
        check=True,
    )
    run = tmp_path / "run"
    (run / "modelstobetested" / "candidate").mkdir(parents=True)
    (run / "models" / "champion").mkdir(parents=True)
    output = tmp_path / "snapshot.json"
    snapshot = deployment_snapshot(
        run_root=run.resolve(),
        repo=repo.resolve(),
        output=output,
        require_procfs=False,
    )
    assert snapshot["source_status"] == ""
    assert snapshot["candidate_names"] == ["candidate"]
    assert snapshot["legacy_model_names"] == ["champion"]
    assert snapshot["controller_accepted_names"] == []
    assert snapshot["process_inventory_source"] in {
        "linux-procfs",
        "unavailable",
    }
    assert json.loads(output.read_text(encoding="utf-8")) == snapshot


def test_candidate_inventory_hashes_model_checkpoint_and_tree(tmp_path):
    inbox = tmp_path / "inbox"
    candidate = inbox / "net-s500000-d1000000"
    candidate.mkdir(parents=True)
    (candidate / "model.bin.gz").write_bytes(b"model")
    (candidate / "model.ckpt").write_bytes(b"checkpoint")
    output = tmp_path / "inventory.json"
    result = candidate_inventory(inbox.resolve(), output)
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["sample_count"] == 500000
    assert len(result["candidates"][0]["directory_manifest_sha256"]) == 64


def test_bootstrap_backpressure_is_canonical_fail_closed_and_idempotent(tmp_path):
    output = (tmp_path / "promotion" / "operations" / "backpressure.json").resolve()
    policy_hash = "a" * 64
    first = bootstrap_backpressure(output, policy_hash)
    second = bootstrap_backpressure(output, policy_hash)

    assert first == second
    assert first["allowExport"] is False
    assert first["allowEvaluation"] is False
    assert first["exportPaused"] is True
    assert first["policy_hash"] == policy_hash
    assert output.read_bytes() == (
        json.dumps(
            first,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()

    allowing = {
        **first,
        "allowExport": True,
        "exportPaused": False,
        "reasons": [],
    }
    output.write_text(
        json.dumps(
            allowing,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    reset = bootstrap_backpressure(output, policy_hash)
    assert reset["allowExport"] is False
    assert reset["exportPaused"] is True
