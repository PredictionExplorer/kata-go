import fcntl
import json
import sys
from pathlib import Path

import pytest

import katago.utils.shuffle_input_gate as shuffle_input_gate
from katago.utils.shuffle_input_gate import (
    STATE_CONTRACT,
    run_if_changed,
)


def append_command(marker: Path, value: str = "x"):
    return [
        sys.executable,
        "-c",
        (
            "from pathlib import Path;"
            f"p=Path({str(marker)!r});"
            f"p.write_text(p.read_text() + {value!r} if p.exists() else {value!r})"
        ),
    ]


def run_gate(tmp_path, command, *, force_after_seconds=0.0):
    input_root = tmp_path / "selfplay"
    input_root.mkdir(exist_ok=True)
    return run_if_changed(
        input_root=input_root,
        state_file=tmp_path / "shuffle-input-state.json",
        lock_file=tmp_path / "shuffle-input.lock",
        output_root=tmp_path / "shuffleddata",
        force_after_seconds=force_after_seconds,
        command=command,
    )


def test_shuffle_gate_runs_once_then_skips_unchanged_input(tmp_path):
    input_root = tmp_path / "selfplay"
    input_root.mkdir()
    (input_root / "a.npz").write_bytes(b"first")
    marker = tmp_path / "marker"
    command = append_command(marker)

    first = run_gate(tmp_path, command)
    second = run_gate(tmp_path, command)

    assert first["status"] == "SHUFFLED"
    assert second["status"] == "SKIPPED_UNCHANGED"
    assert marker.read_text() == "x"
    state_path = tmp_path / "shuffle-input-state.json"
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    assert state["contract"] == STATE_CONTRACT
    assert state_bytes.endswith(b"\n")
    assert state["file_count"] == 1
    assert len(state["state_sha256"]) == 64


def test_shuffle_gate_detects_append_replace_delete_and_command_change(tmp_path):
    input_root = tmp_path / "selfplay"
    input_root.mkdir()
    first_file = input_root / "a.npz"
    first_file.write_bytes(b"first")
    marker = tmp_path / "marker"
    command = append_command(marker)
    assert run_gate(tmp_path, command)["status"] == "SHUFFLED"

    (input_root / "temp_file.npz").write_bytes(b"ignored")
    assert run_gate(tmp_path, command)["status"] == "SKIPPED_UNCHANGED"

    second_file = input_root / "b.npz"
    second_file.write_bytes(b"second")
    assert run_gate(tmp_path, command)["status"] == "SHUFFLED"

    first_file.write_bytes(b"replacement-with-another-size")
    assert run_gate(tmp_path, command)["status"] == "SHUFFLED"

    second_file.unlink()
    assert run_gate(tmp_path, command)["status"] == "SHUFFLED"

    changed_command = append_command(marker, "y")
    assert run_gate(tmp_path, changed_command)["status"] == "SHUFFLED"
    assert marker.read_text() == "xxxxy"


def test_shuffle_gate_detects_changed_summary_dependency(tmp_path):
    input_root = tmp_path / "selfplay"
    input_root.mkdir()
    (input_root / "a.npz").write_bytes(b"input")
    summary = tmp_path / "selfplay.summary.json"
    summary.write_text('{"first":1}\n', encoding="utf-8")
    marker = tmp_path / "marker"
    command = append_command(marker) + ["-summary-file", str(summary)]
    assert run_gate(tmp_path, command)["status"] == "SHUFFLED"
    assert run_gate(tmp_path, command)["status"] == "SKIPPED_UNCHANGED"

    summary.write_text('{"second":2}\n', encoding="utf-8")
    assert run_gate(tmp_path, command)["status"] == "SHUFFLED"
    assert marker.read_text() == "xx"


def test_shuffle_gate_does_not_advance_state_after_command_failure(tmp_path):
    input_root = tmp_path / "selfplay"
    input_root.mkdir()
    (input_root / "a.npz").write_bytes(b"input")
    failure = [sys.executable, "-c", "raise SystemExit(7)"]
    result = run_gate(tmp_path, failure)
    assert result == {
        "status": "FAILED",
        "returncode": 7,
        "combined_sha256": result["combined_sha256"],
    }
    assert not (tmp_path / "shuffle-input-state.json").exists()

    marker = tmp_path / "marker"
    assert run_gate(tmp_path, append_command(marker))["status"] == "SHUFFLED"
    assert marker.read_text() == "x"


def test_shuffle_gate_lock_prevents_overlapping_shufflers(tmp_path):
    input_root = tmp_path / "selfplay"
    input_root.mkdir()
    (input_root / "a.npz").write_bytes(b"input")
    lock_path = tmp_path / "shuffle-input.lock"
    lock_path.touch(mode=0o600)
    marker = tmp_path / "marker"

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_gate(tmp_path, append_command(marker))

    assert result == {"status": "SKIPPED_CONCURRENT"}
    assert not marker.exists()


def test_shuffle_gate_corrupt_state_fails_safe_by_reshuffling(tmp_path):
    input_root = tmp_path / "selfplay"
    input_root.mkdir()
    (input_root / "a.npz").write_bytes(b"input")
    marker = tmp_path / "marker"
    command = append_command(marker)
    assert run_gate(tmp_path, command)["status"] == "SHUFFLED"
    (tmp_path / "shuffle-input-state.json").write_text("{broken", encoding="utf-8")

    assert run_gate(tmp_path, command)["status"] == "SHUFFLED"
    assert marker.read_text() == "xx"


def test_shuffle_gate_force_interval_can_refresh_unchanged_input(tmp_path):
    input_root = tmp_path / "selfplay"
    input_root.mkdir()
    (input_root / "a.npz").write_bytes(b"input")
    marker = tmp_path / "marker"
    command = append_command(marker)
    assert run_gate(tmp_path, command)["status"] == "SHUFFLED"

    state_path = tmp_path / "shuffle-input-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["recorded_at_unix"] = 0
    state.pop("state_sha256")
    from katago.utils.shuffle_input_gate import _canonical_json, _sha256_json

    state["state_sha256"] = _sha256_json(state)
    state_path.write_text(_canonical_json(state) + "\n", encoding="utf-8")

    assert run_gate(
        tmp_path, command, force_after_seconds=1.0
    )["status"] == "SHUFFLED"
    assert marker.read_text() == "xx"


def test_shuffle_gate_rejects_symlinked_npz(tmp_path):
    input_root = tmp_path / "selfplay"
    input_root.mkdir()
    source = tmp_path / "outside.npz"
    source.write_bytes(b"input")
    (input_root / "linked.npz").symlink_to(source)

    with pytest.raises(ValueError, match="not a regular file"):
        run_gate(tmp_path, append_command(tmp_path / "marker"))


def test_shuffle_gate_fails_closed_on_inventory_walk_error(tmp_path, monkeypatch):
    input_root = tmp_path / "selfplay"
    input_root.mkdir()

    def broken_walk(root, *, followlinks, onerror):
        onerror(OSError("injected NFS traversal failure"))
        return ()

    monkeypatch.setattr(shuffle_input_gate.os, "walk", broken_walk)
    with pytest.raises(RuntimeError, match="did not stabilize"):
        shuffle_input_gate.input_inventory(input_root.resolve())
