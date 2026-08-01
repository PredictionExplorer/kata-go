import json
from pathlib import Path
from types import SimpleNamespace

from risk_score.model_probe import probe_model
from risk_score.paired_stats import load_policy
from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256
from risk_score.promotion_evidence import validate_stage0_probe
from risk_score.promotion_controller import inspect_candidate
from risk_score.stage0_probe import run_stage0_probe


DEFAULT_POLICY = Path(__file__).parents[1] / "risk_score" / "promotion_policy_v2.json"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def position(index, *, next_player="B", hint="null"):
    return {
        "xSize": 19,
        "ySize": 19,
        "board": "/".join(["." * 19] * 19),
        "nextPla": next_player,
        "moveLocs": [],
        "movePlas": [],
        "initialTurnNumber": index,
        "hintLoc": hint,
    }


def fake_analysis(argv, **kwargs):
    queries = [
        json.loads(line)
        for line in kwargs["stdin"].read().decode().splitlines()
    ]
    for query in queries:
        visits = query["maxVisits"]
        current_player = query.get("initialPlayer", "B")
        for _player, _move in query["moves"]:
            current_player = "W" if current_player == "B" else "B"
        response = {
            "id": query["id"],
            "rootInfo": {
                "winrate": 0.5,
                "scoreLead": 0.0,
                "utility": 0.0,
                "resultUtility": 0.0,
                "scoreUtility": 0.0,
                "otherUtility": 0.0,
                "visits": visits,
                "currentPlayer": current_player,
            },
            "moveInfos": [
                {
                    "move": "D4",
                    "order": 0,
                    "visits": visits,
                    "prior": 0.5,
                    "scoreLead": 0.0,
                    "scoreSelfplay": 0.0,
                    "scoreStdev": 1.0,
                    "utility": 0.0,
                }
            ],
            "policy": [1.0] + [0.0] * 361,
        }
        kwargs["stdout"].write((canonical_json(response) + "\n").encode())
    return SimpleNamespace(returncode=0, stderr=b"")


def deterministic_config(path):
    path.write_text(
        "\n".join(
            (
                "forDeterministicTesting = true",
                "numAnalysisThreads = 1",
                "nnRandomize = false",
                "rootNoiseEnabled = false",
                "rootNumSymmetriesToSample = 1",
                "useUncertainty = false",
                "cpuctUtilityStdevScale = 0",
                "reportAnalysisWinratesAs = SIDETOMOVE",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_model_probe_requires_finite_analysis(tmp_path):
    binary = tmp_path / "katago"
    binary.write_bytes(b"binary")
    config = tmp_path / "analysis.cfg"
    deterministic_config(config)
    model = tmp_path / "model.bin.gz"
    model.write_bytes(b"model")
    result = probe_model(
        katago=binary,
        config=config,
        model=model,
        expected_model_sha256=file_sha256(model),
        require_idle_gpu=False,
        subprocess_runner=fake_analysis,
    )
    assert result["finite"] is True
    assert result["model_sha256"] == file_sha256(model)


def test_stage0_executor_publishes_validator_derived_measurements(tmp_path):
    policy = json.loads(DEFAULT_POLICY.read_text(encoding="utf-8"))
    policy["policy_version"] = "risk-score-stage0-test"
    stage = policy["evaluation_stages"]["stage_0_integrity_and_fixed_probes"]
    stage["fixed_analysis_positions"] = 2
    stage["exploitability_sentinel_positions"] = 1
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, policy)
    policy_hash = canonical_sha256(policy)

    suites = tmp_path / "suites"
    positions_dir = suites / "positions"
    positions_dir.mkdir(parents=True)
    audit = positions_dir / "audit.jsonl"
    exploit = positions_dir / "exploitability.jsonl"
    tactical = positions_dir / "tactical.jsonl"
    audit.write_text(
        canonical_json(position(0, next_player="B"))
        + "\n"
        + canonical_json(position(1, next_player="W"))
        + "\n",
        encoding="utf-8",
    )
    exploit.write_text(
        canonical_json(position(10, hint="D4")) + "\n", encoding="utf-8"
    )
    tactical.write_text(
        canonical_json(position(11, hint="D4")) + "\n", encoding="utf-8"
    )
    manifest = {
        "banks": [
            {
                "name": "audit",
                "qualifiedName": "audit",
                "positions": {
                    "path": "positions/audit.jsonl",
                    "sha256": file_sha256(audit),
                },
            },
            {
                "name": "exploitability",
                "qualifiedName": "exploitability",
                "positions": {
                    "path": "positions/exploitability.jsonl",
                    "sha256": file_sha256(exploit),
                },
            },
            {
                "name": "tactical",
                "qualifiedName": "tactical",
                "positions": {
                    "path": "positions/tactical.jsonl",
                    "sha256": file_sha256(tactical),
                },
            },
        ]
    }
    suite_manifest = suites / "manifest.json"
    write_json(suite_manifest, manifest)

    candidate_dir = tmp_path / "candidate-s1-d1"
    candidate_dir.mkdir()
    candidate = candidate_dir / "model.bin.gz"
    checkpoint = candidate_dir / "model.ckpt"
    champion = tmp_path / "champion.bin.gz"
    original = tmp_path / "original.bin.gz"
    powered = tmp_path / "powered.cfg"
    standard = tmp_path / "standard.cfg"
    binary = tmp_path / "katago"
    analysis = tmp_path / "analysis.cfg"
    for path, data in (
        (candidate, b"candidate"),
        (checkpoint, b"checkpoint"),
        (champion, b"champion"),
        (original, b"original"),
        (powered, b"powered"),
        (standard, b"standard"),
        (binary, b"binary"),
    ):
        path.write_bytes(data)
    deterministic_config(analysis)
    request = {
        "schema_version": 1,
        "contract": "risk-score-stage-0-request-v1",
        "candidate_hash": file_sha256(candidate),
        "checkpoint_hash": file_sha256(checkpoint),
        "candidate_manifest_hash": inspect_candidate(
            candidate_dir.resolve()
        ).directory_manifest_hash,
        "tested_champion_hash": file_sha256(champion),
        "original_hash": file_sha256(original),
        "policy_path": str(policy_path),
        "policy_hash": policy_hash,
        "policy_version": policy["policy_version"],
        "suite_manifest_path": str(suite_manifest),
        "suite_manifest_hash": file_sha256(suite_manifest),
        "config_hash": canonical_sha256(
            sorted({file_sha256(powered), file_sha256(standard)})
        ),
        "powered_config_path": str(powered),
        "powered_config_hash": file_sha256(powered),
        "standard_config_path": str(standard),
        "standard_config_hash": file_sha256(standard),
        "evaluation_key": "probe-test",
        "stage": "stage-0",
        "look": "automatic",
        "probe_contract": stage,
        "schedule_artifacts": {},
    }
    request_path = tmp_path / "request.json"
    write_json(request_path, request)
    output = tmp_path / "probe.json"
    result = run_stage0_probe(
        request_path=request_path,
        request_sha256=file_sha256(request_path),
        katago=binary,
        analysis_config=analysis,
        candidate_dir=candidate_dir,
        candidate_model=candidate,
        candidate_model_sha256=file_sha256(candidate),
        champion_model=champion,
        champion_model_sha256=file_sha256(champion),
        original_model=original,
        original_model_sha256=file_sha256(original),
        powered_config=powered,
        powered_config_sha256=file_sha256(powered),
        standard_config=standard,
        standard_config_sha256=file_sha256(standard),
        policy_path=policy_path,
        policy_sha256=policy_hash,
        suite_manifest_path=suite_manifest,
        suite_manifest_sha256=file_sha256(suite_manifest),
        output=output,
        subprocess_runner=fake_analysis,
    )
    assert "decision" not in result
    validated = validate_stage0_probe(
        output,
        expected_sha256=file_sha256(output),
        policy=load_policy(policy_path),
        candidate_hash=file_sha256(candidate),
        champion_hash=file_sha256(champion),
        original_hash=file_sha256(original),
        request_path=request_path,
        request_sha256=file_sha256(request_path),
    )
    assert validated["stage_0_passed"] is True
