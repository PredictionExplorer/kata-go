import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from risk_score.build_evaluation_suites import (
    MACHINE_GENERATOR_CONTRACT,
    MACHINE_MANIFEST_CONTRACT,
    MACHINE_REVIEW_MANIFEST_CONTRACT,
)
from risk_score.position_samples import semantic_position_sha256
from risk_score.suite_rotation import (
    ACTIVE_SUITE_CONTRACT,
    ActivationBlockedError,
    ROTATION_REQUEST_CONTRACT,
    StaleChampionError,
    SuiteRotationRegistry,
    SuiteValidationError,
    canonical_json,
    canonical_sha256,
    file_sha256,
    load_registry_spec,
    main,
    publish_continuity_manifest,
    publish_registry_spec,
    rotation_eligibility,
    validate_suite_manifest,
    watch,
)


UTC = timezone.utc
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_canonical(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)
    )


def tiny_v3_policy(path):
    source = Path(__file__).parents[1] / "risk_score" / "promotion_policy_v3.json"
    policy = json.loads(source.read_text(encoding="utf-8"))
    stages = policy["evaluation_stages"]
    stages["stage_1_cheap_paired_screen"]["ordinary_color_pairs"] = 1
    stages["stage_2_finalist_selection"].update(
        {
            "ordinary_color_pairs": 1,
            "lead_40_color_pairs": 1,
            "lead_80_color_pairs": 1,
        }
    )
    for look in stages["stage_3_promotion_confirmation"]["looks"]:
        look.update(
            {
                "powered_ordinary_color_pairs_per_matchup": 1,
                "standard_ordinary_color_pairs": 1,
                "lead_40_color_pairs": 1,
                "lead_80_color_pairs": 1,
                "minimum_independent_position_clusters": {
                    "powered_candidate_vs_champion": 1,
                    "powered_candidate_vs_original": 1,
                    "standard_candidate_vs_original": 1,
                    "lead_40": 1,
                    "lead_80": 1,
                },
            }
        )
    stages["deep_audit"].update(
        {
            "ordinary_color_pairs": 1,
            "lead_40_color_pairs": 1,
            "lead_80_color_pairs": 1,
        }
    )
    write_canonical(path, policy)
    return policy


class Clock:
    def __init__(self, current=T0):
        self.current = current

    def __call__(self):
        return self.current

    def set(self, value):
        self.current = value


@pytest.fixture
def registry_fixture(tmp_path):
    policy_path = tmp_path / "policy-v3.json"
    policy = tiny_v3_policy(policy_path)
    original = tmp_path / "models" / "original.bin.gz"
    champion = tmp_path / "models" / "champion-0.bin.gz"
    original.parent.mkdir()
    original.write_bytes(b"immutable-original")
    champion.write_bytes(b"champion-zero")
    spec_path = tmp_path / "registry-spec.json"
    spec = publish_registry_spec(
        spec_path,
        registry_root=tmp_path / "suite-registry",
        policy_path=policy_path,
        original_model_path=original,
        initial_champion_path=champion,
        initial_generation_id="generation-0",
        created_at_utc="2026-01-01T00:00:00.000000Z",
    )
    return {
        "tmp": tmp_path,
        "policy": policy,
        "policy_path": policy_path,
        "original": original,
        "champion": champion,
        "spec_path": spec_path,
        "spec": spec,
    }


def position(turn):
    return {
        "xSize": 5,
        "ySize": 5,
        "board": "X..../..O../...../...X./.....",
        "nextPla": "B",
        "moveLocs": [],
        "movePlas": [],
        "initialTurnNumber": turn,
        "hintLoc": "null",
        "metadata": "suite-rotation-fixture",
    }


def make_suite(
    root,
    spec,
    champion_sha256,
    *,
    nonce,
    review_mode="machine-consensus",
    overlap=False,
):
    root.mkdir()
    expected = {
        (
            holdout if label == "ordinary" else f"{label}-{holdout}"
        ): (label, holdout)
        for label in ("ordinary", "lead-40", "lead-80")
        for holdout in ("discovery", "confirmation", "audit")
    }
    banks = []
    bank_rows = {}
    for index, (qualified, (label, holdout)) in enumerate(expected.items()):
        turn = nonce * 100 + index
        if overlap and holdout == "confirmation" and label == "ordinary":
            turn = nonce * 100
        row = position(turn)
        semantic = semantic_position_sha256(row)
        content = canonical_sha256(row)
        positions_path = root / "position-banks" / f"{qualified}.jsonl"
        write_jsonl(positions_path, [row])
        schedule_rows = [
            {
                "color": "B",
                "positionSemanticSha256": semantic,
                "suiteQualifiedName": qualified,
            },
            {
                "color": "W",
                "positionSemanticSha256": semantic,
                "suiteQualifiedName": qualified,
            },
        ]
        schedule_path = root / "schedules" / f"{qualified}.jsonl"
        write_jsonl(schedule_path, schedule_rows)
        bank = {
            "name": qualified,
            "qualifiedName": qualified,
            "sourceLabel": label,
            "holdout": holdout,
            "kind": "ordinary" if label == "ordinary" else "specialized",
            "contentSha256s": [content],
            "semanticSha256s": [semantic],
            "independentClusterIds": [semantic],
            "independentClusterIdsSha256": canonical_sha256([semantic]),
            "positionIds": [f"position-{nonce}-{index}"],
            "positionIdsSha256": canonical_sha256([f"position-{nonce}-{index}"]),
            "positions": {
                "path": positions_path.relative_to(root).as_posix(),
                "sha256": file_sha256(positions_path),
                "rowCount": 1,
            },
            "schedule": {
                "path": schedule_path.relative_to(root).as_posix(),
                "sha256": file_sha256(schedule_path),
                "rowCount": 2,
                "pairCount": 1,
                "scheduleId": f"schedule-{nonce}-{index}",
                "baseSeed": f"seed-{nonce}-{index}",
            },
        }
        banks.append(bank)
        bank_rows[qualified] = bank

    discovery = bank_rows["discovery"]
    cell_body = {
        "cell_name": "powered_candidate_vs_champion",
        "stage": "stage-1",
        "look": "automatic",
        "comparison": "candidate-vs-champion-powered",
        "suite": "discovery",
        "search_mode": "powered",
        "visits": 1,
        "color_pairs": 1,
        "minimum_independent_position_clusters": 1,
        "independent_cluster_ids": discovery["semanticSha256s"],
        "independent_cluster_ids_hash": canonical_sha256(
            discovery["semanticSha256s"]
        ),
        "position_ids": discovery["positionIds"],
        "position_ids_hash": canonical_sha256(discovery["positionIds"]),
        "bank_name": "discovery",
        "bank_path": discovery["positions"]["path"],
        "bank_hash": discovery["positions"]["sha256"],
        "schedule_path": discovery["schedule"]["path"],
        "schedule_hash": discovery["schedule"]["sha256"],
        "schedule_id": discovery["schedule"]["scheduleId"],
        "schedule_row_count": 2,
        "maximal_look_schedule": False,
        "policy_hash": spec.policy_identity,
        "policy_version": "risk-seeking-checkpoint-promotion-v3",
        "source_revision": "a" * 40,
    }
    cell = {
        "cell_id": "suite-cell-" + canonical_sha256(cell_body),
        **cell_body,
    }
    source_hash = digest(f"reviewed-source-{nonce}")
    manifest = {
        "schemaVersion": 3,
        "manifestContract": MACHINE_MANIFEST_CONTRACT,
        "generatorContract": MACHINE_GENERATOR_CONTRACT,
        "scheduleGeneratorContract": "risk-score-paired-schedule-v1",
        "canonicalJsonContract": "utf-8-sort-keys-compact-json-lines-v1",
        "ordinaryAssignmentContract": (
            "policy-exact-seeded-semantic-hash-disjoint-holdouts-v2"
        ),
        "semanticPositionContract": (
            "canonical-xSize-ySize-board-nextPla-moveLocs-movePlas-"
            "initialTurnNumber-v1"
        ),
        "seed": f"suite-{nonce}",
        "policy_hash": spec.policy_identity,
        "policy_version": "risk-seeking-checkpoint-promotion-v3",
        "source_revision": "a" * 40,
        "exactPolicyQuotas": True,
        "policyHoldoutQuotas": {
            label: dict(spec.holdout_quotas[label])
            for label in ("ordinary", "lead-40", "lead-80")
        },
        "ordinaryWeights": {
            "discovery": 1.0,
            "confirmation": 1.0,
            "audit": 1.0,
        },
        "pairsPerPosition": 1,
        "botAIndex": 0,
        "botBIndex": 1,
        "acceptedLabels": ["lead-40", "lead-80", "ordinary"],
        "sources": [
            {
                "name": f"source-{nonce}.jsonl",
                "sha256": source_hash,
                "rowCount": 9,
                "blankLineCount": 0,
            }
        ],
        "inputRowCount": 9,
        "includedRowCount": 9,
        "assignedRowCount": 9,
        "unassigned": [],
        "exclusions": [],
        "banks": banks,
        "cells": [cell],
        "discovery_schedule_hash": discovery["schedule"]["sha256"],
        "machineReviewOnly": True,
        "curationSources": [
            {
                "source_name": f"source-{nonce}.jsonl",
                "contract": MACHINE_REVIEW_MANIFEST_CONTRACT,
                "review_mode": review_mode,
                "consensus_rules_version": 1,
                "policy_hash": spec.policy_identity,
                "allowed_labels": ["lead-40", "lead-80", "ordinary"],
                "output_sha256": source_hash,
                "manifest_sha256": digest(f"curation-file-{nonce}"),
                "manifest_identity": digest(f"curation-id-{nonce}"),
                "rejected_count": 0,
                "rejected_sha256": digest(f"rejected-{nonce}"),
                "models": {
                    "original": {
                        "role": "immutable_original",
                        "sha256": spec.original.sha256,
                    },
                    "champion": {
                        "role": "frozen_champion",
                        "sha256": champion_sha256,
                    },
                },
            }
        ],
    }
    manifest["manifestPayloadSha256"] = canonical_sha256(manifest)
    manifest_path = root / "manifest.json"
    write_canonical(manifest_path, manifest)
    return manifest_path


def bootstrap(fixture, clock):
    manifest = make_suite(
        fixture["tmp"] / "suite-0",
        fixture["spec"],
        file_sha256(fixture["champion"]),
        nonce=0,
    )
    registry = SuiteRotationRegistry(fixture["spec_path"], clock=clock)
    registry.bootstrap(manifest)
    return registry, manifest


def accept_champions(registry, fixture, clock, count):
    models = []
    previous = file_sha256(fixture["champion"])
    for index in range(1, count + 1):
        path = fixture["tmp"] / "models" / f"champion-{index}.bin.gz"
        path.write_bytes(f"champion-{index}".encode("utf-8"))
        clock.set(T0 + timedelta(days=index))
        registry.record_accepted_champion(
            path,
            generation_id=f"generation-{index}",
            expected_previous_champion_sha256=previous,
        )
        previous = file_sha256(path)
        models.append(path)
    return models


def prepare_rotation(registry_fixture, *, failure_hook=None):
    clock = Clock()
    registry, initial_manifest = bootstrap(registry_fixture, clock)
    champions = accept_champions(registry, registry_fixture, clock, 5)
    request_event = registry.request_rotation()
    request_id = request_event.payload["request_id"]
    current_hash = file_sha256(champions[-1])
    proposed_manifest = make_suite(
        registry_fixture["tmp"] / "suite-1",
        registry_fixture["spec"],
        current_hash,
        nonce=1,
    )
    registration = registry.register_suite(request_id, proposed_manifest)
    suite_id = registration.payload["suite_id"]
    current_evidence = registry_fixture["tmp"] / "current-shadow.json"
    previous_evidence = registry_fixture["tmp"] / "previous-shadow.json"
    write_canonical(current_evidence, {"decision": "PASS", "role": "current"})
    write_canonical(previous_evidence, {"decision": "PASS", "role": "previous"})
    continuity_path = registry_fixture["tmp"] / "continuity.json"
    publish_continuity_manifest(
        continuity_path,
        request_id=request_id,
        candidate_suite_id=suite_id,
        base_suite_id=file_sha256(initial_manifest),
        policy_hash=registry_fixture["spec"].policy_identity,
        current_champion_sha256=current_hash,
        previous_champion_sha256=file_sha256(champions[-2]),
        current_evidence_path=current_evidence,
        previous_evidence_path=previous_evidence,
        completed_at_utc="2026-01-07T00:00:00.000000Z",
    )
    registry.record_continuity(request_id, suite_id, continuity_path)
    if failure_hook is not None:
        registry.failure_hook = failure_hook
    return {
        "registry": registry,
        "clock": clock,
        "initial_manifest": initial_manifest,
        "champions": champions,
        "request_id": request_id,
        "suite_id": suite_id,
        "current_hash": current_hash,
    }


def test_rotation_trigger_is_fifth_accepted_champion_or_ninety_days(
    registry_fixture,
):
    clock = Clock()
    registry, _ = bootstrap(registry_fixture, clock)
    accept_champions(registry, registry_fixture, clock, 4)

    before = rotation_eligibility(registry.reconstruct(), clock())
    assert before.eligible is False
    assert before.accepted_champions_remaining == 1

    fifth = registry_fixture["tmp"] / "models" / "champion-5.bin.gz"
    fifth.write_bytes(b"champion-five")
    previous = registry.reconstruct().current_champion.sha256
    registry.record_accepted_champion(
        fifth,
        generation_id="generation-5",
        expected_previous_champion_sha256=previous,
    )
    due = registry.status(now=clock())
    assert due["cadence"]["eligible"] is True
    assert due["cadence"]["reasons"] == ["accepted-champion-interval"]
    assert due["cadence"]["candidate_results_used"] is False

    second_root = registry_fixture["tmp"] / "age-only"
    second_root.mkdir()
    original = second_root / "original"
    champion = second_root / "champion"
    original.write_bytes(b"age-original")
    champion.write_bytes(b"age-champion")
    age_spec_path = second_root / "spec.json"
    age_spec = publish_registry_spec(
        age_spec_path,
        registry_root=second_root / "registry",
        policy_path=registry_fixture["policy_path"],
        original_model_path=original,
        initial_champion_path=champion,
        initial_generation_id="age-generation",
        created_at_utc="2026-01-01T00:00:00.000000Z",
    )
    age_manifest = make_suite(
        second_root / "suite",
        age_spec,
        file_sha256(champion),
        nonce=8,
    )
    age_clock = Clock()
    age_registry = SuiteRotationRegistry(age_spec_path, clock=age_clock)
    age_registry.bootstrap(age_manifest)
    age_clock.set(T0 + timedelta(days=90))
    age_due = age_registry.status()
    assert age_due["cadence"]["eligible"] is True
    assert age_due["cadence"]["reasons"] == ["maximum-age"]


def test_suite_validation_rejects_non_consensus_provenance_and_semantic_overlap(
    registry_fixture,
):
    invalid_provenance = make_suite(
        registry_fixture["tmp"] / "invalid-provenance",
        registry_fixture["spec"],
        file_sha256(registry_fixture["champion"]),
        nonce=2,
        review_mode="human-review",
    )
    with pytest.raises(SuiteValidationError, match="machine-consensus"):
        validate_suite_manifest(invalid_provenance, registry_fixture["spec"])

    overlap = make_suite(
        registry_fixture["tmp"] / "overlap",
        registry_fixture["spec"],
        file_sha256(registry_fixture["champion"]),
        nonce=3,
        overlap=True,
    )
    with pytest.raises(SuiteValidationError, match="semantic overlap"):
        validate_suite_manifest(overlap, registry_fixture["spec"])


def test_in_flight_evaluation_stays_pinned_and_blocks_rotation(registry_fixture):
    prepared = prepare_rotation(registry_fixture)
    registry = prepared["registry"]
    old_suite = registry.reconstruct().active_suite_id
    pin = registry.pin_evaluation("evaluation-in-flight")
    assert pin["suite_id"] == old_suite
    registry.record_generation_boundary(
        "boundary-5",
        generation_id="generation-5",
        champion_sha256=prepared["current_hash"],
    )

    with pytest.raises(ActivationBlockedError, match="in-flight"):
        registry.activate_suite(
            prepared["request_id"],
            prepared["suite_id"],
            expected_active_suite_id=old_suite,
            expected_champion_sha256=prepared["current_hash"],
            boundary_id="boundary-5",
        )
    assert registry.reconstruct().pins["evaluation-in-flight"]["suite_id"] == old_suite
    assert registry.reconstruct().active_suite_id == old_suite

    registry.unpin_evaluation("evaluation-in-flight")
    registry.activate_suite(
        prepared["request_id"],
        prepared["suite_id"],
        expected_active_suite_id=old_suite,
        expected_champion_sha256=prepared["current_hash"],
        boundary_id="boundary-5",
    )
    assert registry.reconstruct().active_suite_id == prepared["suite_id"]


def test_activation_compare_and_swap_rejects_stale_champion(registry_fixture):
    prepared = prepare_rotation(registry_fixture)
    registry = prepared["registry"]
    old_suite = registry.reconstruct().active_suite_id
    registry.record_generation_boundary(
        "boundary-before-next-champion",
        generation_id="generation-5",
        champion_sha256=prepared["current_hash"],
    )
    next_champion = registry_fixture["tmp"] / "models" / "champion-6.bin.gz"
    next_champion.write_bytes(b"champion-six")
    registry.record_accepted_champion(
        next_champion,
        generation_id="generation-6",
        expected_previous_champion_sha256=prepared["current_hash"],
    )

    with pytest.raises(StaleChampionError, match="changed"):
        registry.activate_suite(
            prepared["request_id"],
            prepared["suite_id"],
            expected_active_suite_id=old_suite,
            expected_champion_sha256=prepared["current_hash"],
            boundary_id="boundary-before-next-champion",
        )
    assert registry.reconstruct().active_suite_id == old_suite


def test_activation_crash_replays_idempotently_and_repairs_projection(
    registry_fixture,
):
    fired = []

    def crash(step):
        if step == "activation-event-published" and not fired:
            fired.append(step)
            raise RuntimeError("simulated activation crash")

    prepared = prepare_rotation(registry_fixture, failure_hook=crash)
    registry = prepared["registry"]
    old_suite = registry.reconstruct().active_suite_id
    registry.record_generation_boundary(
        "boundary-crash",
        generation_id="generation-5",
        champion_sha256=prepared["current_hash"],
    )
    with pytest.raises(RuntimeError, match="activation crash"):
        registry.activate_suite(
            prepared["request_id"],
            prepared["suite_id"],
            expected_active_suite_id=old_suite,
            expected_champion_sha256=prepared["current_hash"],
            boundary_id="boundary-crash",
        )

    replay = SuiteRotationRegistry(registry_fixture["spec_path"])
    event = replay.activate_suite(
        prepared["request_id"],
        prepared["suite_id"],
        expected_active_suite_id=old_suite,
        expected_champion_sha256=prepared["current_hash"],
        boundary_id="boundary-crash",
    )
    state = replay.reconstruct()
    active = json.loads(replay.active_path.read_text(encoding="utf-8"))
    assert event.event_type == "suite.activated"
    assert state.active_suite_id == prepared["suite_id"]
    assert active["contract"] == ACTIVE_SUITE_CONTRACT
    assert active["suite_id"] == prepared["suite_id"]
    assert (
        len([item for item in state.events if item.event_type == "suite.activated"])
        == 1
    )


def test_old_suites_remain_immutable_and_retained_after_activation(
    registry_fixture,
):
    prepared = prepare_rotation(registry_fixture)
    registry = prepared["registry"]
    old_suite = registry.reconstruct().active_suite_id
    old_manifest = registry.suites_dir / old_suite / "manifest.json"
    old_bytes = old_manifest.read_bytes()
    registry.record_generation_boundary(
        "boundary-retention",
        generation_id="generation-5",
        champion_sha256=prepared["current_hash"],
    )
    registry.activate_suite(
        prepared["request_id"],
        prepared["suite_id"],
        expected_active_suite_id=old_suite,
        expected_champion_sha256=prepared["current_hash"],
        boundary_id="boundary-retention",
    )

    status = registry.status()
    retained = {item["suite_id"]: item for item in status["retained_suites"]}
    assert set(retained) == {old_suite, prepared["suite_id"]}
    assert retained[old_suite]["active"] is False
    assert retained[old_suite]["immutable"] is True
    assert retained[old_suite]["retained"] is True
    assert old_manifest.read_bytes() == old_bytes
    assert (registry.versions_dir / f"{old_suite}.json").is_file()
    assert load_registry_spec(registry_fixture["spec_path"]).identity == registry.spec.identity


def test_once_never_activates_without_privileged_deployment_handshake(
    registry_fixture,
):
    prepared = prepare_rotation(registry_fixture)
    registry = prepared["registry"]
    old_suite = registry.reconstruct().active_suite_id
    registry.record_generation_boundary(
        "boundary-ready-but-not-deployed",
        generation_id="generation-5",
        champion_sha256=prepared["current_hash"],
    )

    status_value = registry.once()

    assert registry.reconstruct().active_suite_id == old_suite
    assert status_value["active_suite"]["suite_id"] == old_suite
    assert prepared["suite_id"] in {
        item["suite_id"] for item in status_value["retained_suites"]
    }


def test_once_publishes_bound_requests_and_status_cli_is_machine_readable(
    registry_fixture, capsys
):
    clock = Clock()
    registry, _ = bootstrap(registry_fixture, clock)
    champions = accept_champions(registry, registry_fixture, clock, 5)

    status = registry.once()
    request_id = status["current_request_id"]
    request_path = registry.requests_dir / request_id / "manifest.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))

    assert status["state"] == "curation-requested"
    assert request["contract"] == ROTATION_REQUEST_CONTRACT
    assert request["models"]["original"]["sha256"] == registry.spec.original.sha256
    assert request["models"]["champion"]["sha256"] == file_sha256(champions[-1])
    assert {
        "curation_supplement",
        "curation_pipeline",
    } == set(request["requests"])

    assert main(["status", "--spec", str(registry_fixture["spec_path"])]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["contract"].endswith("rotation-status-v1")
    assert output["active_suite"]["suite_id"] == registry.reconstruct().active_suite_id


def test_watch_reconciles_once_before_sleep(registry_fixture):
    clock = Clock()
    registry, _ = bootstrap(registry_fixture, clock)
    calls = []

    def stop_after_first(interval):
        calls.append(interval)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        watch(
            registry_fixture["spec_path"],
            interval_seconds=12.5,
            sleeper=stop_after_first,
        )

    status_value = json.loads(
        registry.status_path.read_text(encoding="utf-8")
    )
    assert status_value["contract"].endswith("rotation-status-v1")
    assert calls == [12.5]
