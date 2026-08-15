import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256
from risk_score.selfplay_margin_loop import (
    MarginLoopError,
    MarginLoopSpec,
    SelfplayMarginLoop,
    build_gatekeeper_command,
)


REPO = Path(__file__).resolve().parents[2]


def binding(path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def publish_spec(tmp_path, *, win_rate=0.35, maximum_candidates=3):
    run = tmp_path / "run"
    deploy = tmp_path / "deploy"
    for path in (
        run,
        deploy,
        run / "modelstobetested",
        run / "models",
        run / "rejectedmodels",
        run / "supersededmodels",
        run / "selfplay",
        run / "gatekeepersgf",
        run / "state",
    ):
        path.mkdir(parents=True, exist_ok=True)

    executable = deploy / "tool"
    gatekeeper = deploy / "katago"
    config = deploy / "gatekeeper.cfg"
    executable.write_bytes(b"tool")
    gatekeeper.write_bytes(b"katago")
    config.write_text("numGamesPerGating = 100\n", encoding="utf-8")

    source_policy = (
        Path(__file__).parents[1]
        / "risk_score"
        / "selfplay_margin_policy_v1.json"
    )
    policy_value = json.loads(source_policy.read_text())
    policy_value["promotion"]["minimum_candidate_win_rate"] = win_rate
    policy_value["promotion"]["maximum_active_candidates"] = maximum_candidates
    policy = deploy / "policy.json"
    policy.write_text(json.dumps(policy_value, indent=2) + "\n", encoding="utf-8")

    value = {
        "schema_version": 1,
        "contract": "risk-score-selfplay-margin-loop-spec-v1",
        "policy": binding(policy),
        "run_root": str(run.resolve()),
        "paths": {
            "candidate_inbox": str((run / "modelstobetested").resolve()),
            "accepted_models": str((run / "models").resolve()),
            "rejected_models": str((run / "rejectedmodels").resolve()),
            "superseded_models": str((run / "supersededmodels").resolve()),
            "selfplay_root": str((run / "selfplay").resolve()),
            "gate_sgf_root": str((run / "gatekeepersgf").resolve()),
            "status": str((run / "state/status.json").resolve()),
            "lock": str((run / "state/loop.lock").resolve()),
        },
        "trainer": {
            "argv": [str(executable), "trainer"],
            "cwd": str(deploy.resolve()),
            "env": {"CUDA_VISIBLE_DEVICES": "7"},
        },
        "exporter": {
            "argv": [str(executable), "exporter"],
            "cwd": str(deploy.resolve()),
            "env": {},
        },
        "gatekeeper": {
            "binary": binding(gatekeeper),
            "config": binding(config),
            "cwd": str(deploy.resolve()),
            "env": {"CUDA_VISIBLE_DEVICES": "7"},
        },
        "cycle_sleep_seconds": 30,
    }
    value["spec_sha256"] = canonical_sha256(value)
    spec = tmp_path / "margin-loop.json"
    spec.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return spec, run


def test_loads_selfplay_only_policy_and_builds_permissive_gate(tmp_path):
    spec_path, run = publish_spec(tmp_path)
    spec = MarginLoopSpec.load(spec_path)

    command = build_gatekeeper_command(spec)

    assert spec.policy.minimum_candidate_win_rate == 0.35
    assert spec.policy.games_per_candidate == 100
    assert command[1] == "gatekeeper"
    assert command[command.index("-required-candidate-win-prop") + 1] == "0.35"
    assert command[command.index("-accepted-models-dir") + 1] == str(
        run / "models"
    )
    assert command[-1] == "-quit-if-no-nets-to-test"


def test_repository_margin_policy_and_gatekeeper_config_agree():
    policy = json.loads(
        (
            REPO / "python/risk_score/selfplay_margin_policy_v1.json"
        ).read_text()
    )
    assignments = {}
    for raw in (
        REPO
        / "cpp/configs/risk_score/margin_safety_gatekeeper_19x19.cfg"
    ).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            assignments[key] = value

    assert policy["promotion"]["external_position_data_allowed"] is False
    assert policy["promotion"]["minimum_candidate_win_rate"] == 0.35
    assert int(assignments["numGamesPerGating"]) == policy["promotion"][
        "games_per_candidate"
    ]
    assert int(assignments["maxVisits"]) == policy["promotion"]["visits_per_move"]
    assert assignments["useScoreMaximizingUtility"] == "false"
    assert assignments["bSizes"] == "19"
    assert assignments["komiMean"] == "7.5"
    assert assignments["handicapProb"] == "0"
    assert assignments["cudaDeviceToUseModel0Thread0"] == "0"
    assert assignments["cudaDeviceToUseModel1Thread0"] == "0"


def test_cycle_exports_gates_trains_and_coalesces_without_deleting(tmp_path):
    spec_path, run = publish_spec(tmp_path, maximum_candidates=3)
    for index in range(5):
        candidate = (
            run
            / "modelstobetested"
            / f"risk-b40-p15-w4-s{1000 + index}-d{2000 + index}"
        )
        candidate.mkdir()
        (candidate / "model.bin.gz").write_bytes(f"model-{index}".encode())
    calls = []

    def runner(argv, **kwargs):
        calls.append(tuple(argv))
        return SimpleNamespace(returncode=0)

    result = SelfplayMarginLoop(spec_path, runner=runner).once()

    assert [command[1] for command in calls] == [
        "exporter",
        "gatekeeper",
        "trainer",
    ]
    assert len(list((run / "modelstobetested").iterdir())) == 3
    assert sorted(path.name for path in (run / "supersededmodels").iterdir()) == [
        "risk-b40-p15-w4-s1000-d2000",
        "risk-b40-p15-w4-s1001-d2001",
    ]
    assert result["state"] == "cycle_complete"
    assert (run / "state/status.json").is_file()


def test_policy_rejects_too_weak_gate_and_unsafe_candidate_entry(tmp_path):
    weak_spec, _ = publish_spec(tmp_path / "weak", win_rate=0.2)
    with pytest.raises(MarginLoopError, match="win rate"):
        MarginLoopSpec.load(weak_spec)

    spec_path, run = publish_spec(tmp_path / "unsafe")
    (run / "modelstobetested" / "not-a-candidate").mkdir()
    with pytest.raises(MarginLoopError, match="malformed"):
        SelfplayMarginLoop(
            spec_path,
            runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
        ).once()


def test_command_failure_is_fail_closed(tmp_path):
    spec_path, _ = publish_spec(tmp_path)

    def runner(argv, **_kwargs):
        return SimpleNamespace(returncode=17 if argv[1] == "gatekeeper" else 0)

    with pytest.raises(MarginLoopError, match="gatekeeper failed"):
        SelfplayMarginLoop(spec_path, runner=runner).once()
