import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from risk_score.promotion_state import (
    GENESIS_HASH,
    CandidateState,
    ChampionConflictError,
    ControllerLock,
    ControllerLockError,
    EventProvenance,
    EventRegistry,
    GenerationState,
    IllegalTransitionError,
    PassReportError,
    PromotionEvent,
    RegistryCorruptionError,
    StaleChampionError,
    Transition,
    atomic_write_json,
    bootstrap_champion,
    canonical_json_bytes,
    canonical_sha256,
    compare_and_swap_champion,
    load_champion,
    sha256_bytes,
    sha256_file,
)


def digest(label):
    return sha256_bytes(label.encode("utf-8"))


@pytest.fixture
def provenance():
    return EventProvenance(
        controller_hash=digest("controller"),
        source_hash=digest("source"),
        original_hash=digest("original"),
        config_hash=digest("config"),
        schedule_hash=digest("schedule"),
        policy_hash=digest("policy"),
    )


def bootstrap_registry(tmp_path, provenance):
    registry = EventRegistry(tmp_path / "promotion")
    event = registry.bootstrap_champion(
        champion_hash=digest("champion-0"),
        generation_id="generation-0",
        provenance=provenance,
        reason="initial production champion",
        actor="test-controller",
        timestamp_utc="2026-07-28T12:00:00.000000Z",
    )
    return registry, event


def confirm_candidate(registry, provenance, candidate_hash, path):
    champion = digest("champion-0")
    registry.transition_candidate(
        candidate_hash,
        path,
        CandidateState.DISCOVERED,
        provenance=provenance,
        champion_hash=champion,
        reason="complete export discovered",
        actor="test-controller",
    )
    registry.transition_candidate(
        candidate_hash,
        path,
        CandidateState.CLAIMED,
        provenance=provenance,
        champion_hash=champion,
        reason="candidate claimed",
        actor="test-controller",
    )
    registry.transition_candidate(
        candidate_hash,
        path,
        CandidateState.EVALUATING_INTEGRITY,
        provenance=provenance,
        champion_hash=champion,
        evaluation_key="integrity-1",
        reason="fixed probes started",
        actor="test-controller",
    )
    registry.transition_candidate(
        candidate_hash,
        path,
        CandidateState.EVALUATING_SCREEN,
        provenance=provenance,
        champion_hash=champion,
        evaluation_key="screen-1",
        reason="screen started",
        actor="test-controller",
    )
    registry.transition_candidate(
        candidate_hash,
        path,
        CandidateState.EVALUATING_FINALIST,
        provenance=provenance,
        champion_hash=champion,
        evaluation_key="finalist-1",
        reason="finalist evaluation started",
        actor="test-controller",
    )
    registry.transition_candidate(
        candidate_hash,
        path,
        CandidateState.EVALUATING_CONFIRMATION,
        provenance=provenance,
        champion_hash=champion,
        evaluation_key="confirmation-1",
        reason="independent confirmation started",
        actor="test-controller",
    )
    return registry.transition_candidate(
        candidate_hash,
        path,
        CandidateState.CONFIRMED,
        provenance=provenance,
        champion_hash=champion,
        evaluation_key="confirmation-1",
        reason="finalized PASS report",
        actor="test-controller",
    )


def write_pass_report(
    path,
    provenance,
    *,
    candidate_hash,
    tested_champion_hash,
    evaluation_key="confirmation-1",
    decision="PASS",
    finalized=True,
):
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "decision": decision,
            "finalized": finalized,
            "finalized_at_utc": "2026-07-28T13:00:00.000000Z",
            "candidate_hash": candidate_hash,
            "tested_champion_hash": tested_champion_hash,
            "original_hash": provenance.original_hash,
            "evaluation_key": evaluation_key,
            "config_hash": provenance.config_hash,
            "schedule_hash": provenance.schedule_hash,
            "policy_hash": provenance.policy_hash,
            "metrics": {"utility_lcb": 0.2},
        },
    )
    return sha256_file(path)


def test_canonical_json_and_hashes_are_stable():
    first = {"z": [3, 2, 1], "a": "é", "nested": {"b": True, "a": None}}
    second = {"nested": {"a": None, "b": True}, "a": "é", "z": [3, 2, 1]}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_bytes(first).startswith(b'{"a":"\xc3\xa9"')
    assert canonical_sha256(first) == canonical_sha256(second)
    assert len(canonical_sha256(first)) == 64
    with pytest.raises(ValueError, match="canonical-JSON"):
        canonical_json_bytes({"bad": float("nan")})


def test_event_append_replay_and_rollout_lifecycle(tmp_path, provenance):
    registry, bootstrap = bootstrap_registry(tmp_path, provenance)
    candidate = digest("candidate-1")
    candidate_path = "candidates/claimed/checkpoint-1"
    confirm_candidate(registry, provenance, candidate, candidate_path)

    registry.transition_generation(
        "generation-1",
        candidate,
        candidate_path,
        GenerationState.PROMOTION_INTENT,
        provenance=provenance,
        tested_champion_hash=digest("champion-0"),
        evaluation_key="confirmation-1",
        reason="promotion transaction prepared",
        actor="test-controller",
    )
    registry.transition_generation(
        "generation-1",
        candidate,
        candidate_path,
        GenerationState.CANARY,
        provenance=provenance,
        tested_champion_hash=digest("champion-0"),
        reason="one worker acknowledged candidate",
        actor="test-controller",
    )
    registry.transition_generation(
        "generation-1",
        candidate,
        candidate_path,
        GenerationState.ROLLOUT,
        provenance=provenance,
        tested_champion_hash=digest("champion-0"),
        reason="canary passed",
        actor="test-controller",
    )
    registry.transition_generation(
        "generation-1",
        candidate,
        candidate_path,
        GenerationState.ACTIVE,
        provenance=provenance,
        tested_champion_hash=digest("champion-0"),
        reason="all workers acknowledged candidate",
        actor="test-controller",
    )

    restarted = EventRegistry(tmp_path / "promotion")
    state = restarted.reconstruct()
    assert bootstrap.previous_hash == GENESIS_HASH
    assert state.current_champion_hash == candidate
    assert state.current_generation_id == "generation-1"
    assert state.candidates[candidate].state == CandidateState.CONFIRMED
    assert state.generations["generation-1"].state == GenerationState.ACTIVE
    assert [event.sequence for event in state.events] == list(
        range(1, len(state.events) + 1)
    )
    for previous, current in zip(state.events, state.events[1:]):
        assert current.previous_hash == previous.event_hash
    filenames = sorted(path.name for path in restarted.events_dir.glob("*.json"))
    assert filenames[0] == "00000000000000000001.json"
    assert filenames[-1] == f"{len(state.events):020d}.json"
    assert (
        restarted.bootstrap_champion(
            champion_hash=digest("champion-0"),
            generation_id="generation-0",
            provenance=provenance,
            reason="restart bootstrap retry",
            actor="replacement-controller",
        )
        == bootstrap
    )


def test_generation_rollback_restores_previous_champion(tmp_path, provenance):
    registry, _ = bootstrap_registry(tmp_path, provenance)
    candidate = digest("candidate-rollback")
    path = "candidates/claimed/candidate-rollback"
    confirm_candidate(registry, provenance, candidate, path)
    for target in (
        GenerationState.PROMOTION_INTENT,
        GenerationState.CANARY,
        GenerationState.ROLLOUT,
        GenerationState.ACTIVE,
    ):
        registry.transition_generation(
            "generation-rollback",
            candidate,
            path,
            target,
            provenance=provenance,
            tested_champion_hash=digest("champion-0"),
            evaluation_key=(
                "confirmation-1"
                if target == GenerationState.PROMOTION_INTENT
                else None
            ),
            reason=f"advance to {target.value}",
            actor="test-controller",
        )
    for target in (
        GenerationState.ROLLBACK_PENDING,
        GenerationState.ROLLED_BACK,
    ):
        registry.transition_generation(
            "generation-rollback",
            candidate,
            path,
            target,
            provenance=provenance,
            tested_champion_hash=digest("champion-0"),
            restore_champion_hash=digest("champion-0"),
            reason=f"advance to {target.value}",
            actor="test-controller",
        )

    state = registry.reconstruct()
    assert state.current_champion_hash == digest("champion-0")
    assert state.current_generation_id == "generation-0"
    assert (
        state.generations["generation-rollback"].state
        == GenerationState.ROLLED_BACK
    )


def test_candidate_transition_retry_is_idempotent_after_restart(
    tmp_path, provenance
):
    registry, _ = bootstrap_registry(tmp_path, provenance)
    candidate = digest("candidate-idempotent")
    kwargs = {
        "provenance": provenance,
        "champion_hash": digest("champion-0"),
        "reason": "candidate discovered",
        "actor": "test-controller",
    }
    first = registry.transition_candidate(
        candidate,
        "candidates/claimed/idempotent",
        CandidateState.DISCOVERED,
        **kwargs,
    )
    restarted = EventRegistry(tmp_path / "promotion")
    retry = restarted.transition_candidate(
        candidate,
        "candidates/claimed/idempotent",
        CandidateState.DISCOVERED,
        **kwargs,
    )

    assert retry == first
    assert restarted.reconstruct().last_sequence == 2


@pytest.mark.parametrize(
    "target_state",
    [
        CandidateState.DISCOVERED,
        CandidateState.CLAIMED,
        CandidateState.EVALUATING_INTEGRITY,
        CandidateState.EVALUATING_SCREEN,
        CandidateState.EVALUATING_FINALIST,
        CandidateState.EVALUATING_CONFIRMATION,
    ],
)
def test_restart_reconstructs_every_candidate_evaluation_state(
    tmp_path, provenance, target_state
):
    registry, _ = bootstrap_registry(tmp_path, provenance)
    candidate = digest("candidate-restart-" + target_state.value)
    path = "candidates/claimed/" + target_state.value
    champion = digest("champion-0")
    transitions = (
        (CandidateState.DISCOVERED, None),
        (CandidateState.CLAIMED, None),
        (CandidateState.EVALUATING_INTEGRITY, "integrity"),
        (CandidateState.EVALUATING_SCREEN, "screen"),
        (CandidateState.EVALUATING_FINALIST, "finalist"),
        (CandidateState.EVALUATING_CONFIRMATION, "confirmation-look-1"),
    )
    expected_key = None
    for state, evaluation_key in transitions:
        registry.transition_candidate(
            candidate,
            path,
            state,
            provenance=provenance,
            champion_hash=champion,
            evaluation_key=evaluation_key,
            reason=f"advance to {state.value}",
            actor="test-controller",
            payload=(
                {"look": "look-1"}
                if state == CandidateState.EVALUATING_CONFIRMATION
                else None
            ),
        )
        expected_key = evaluation_key
        if state == target_state:
            break

    restarted = EventRegistry(tmp_path / "promotion")
    record = restarted.reconstruct().candidates[candidate]
    assert record.state == target_state
    assert record.evaluation_key == expected_key
    assert record.candidate_path == path


def test_confirmation_can_advance_to_prespecified_second_look_once(
    tmp_path, provenance
):
    registry, _ = bootstrap_registry(tmp_path, provenance)
    candidate = digest("candidate-two-look")
    path = "candidates/claimed/candidate-two-look"
    champion = digest("champion-0")
    for target, evaluation_key in (
        (CandidateState.DISCOVERED, None),
        (CandidateState.CLAIMED, None),
        (CandidateState.EVALUATING_INTEGRITY, "integrity"),
        (CandidateState.EVALUATING_SCREEN, "screen"),
        (CandidateState.EVALUATING_FINALIST, "finalist"),
        (CandidateState.EVALUATING_CONFIRMATION, "confirmation-look-1"),
    ):
        registry.transition_candidate(
            candidate,
            path,
            target,
            provenance=provenance,
            champion_hash=champion,
            evaluation_key=evaluation_key,
            reason=f"advance to {target.value}",
            actor="test-controller",
            payload=(
                {"look": "look-1"}
                if target == CandidateState.EVALUATING_CONFIRMATION
                else None
            ),
        )

    second = registry.transition_candidate(
        candidate,
        path,
        CandidateState.EVALUATING_CONFIRMATION,
        provenance=provenance,
        champion_hash=champion,
        evaluation_key="confirmation-look-2",
        reason="prespecified cumulative look 2 started",
        actor="test-controller",
        payload={
            "look": "look-2",
            "previous_evaluation_key": "confirmation-look-1",
            "prespecified_cumulative_look": True,
        },
    )
    retry = EventRegistry(tmp_path / "promotion").transition_candidate(
        candidate,
        path,
        CandidateState.EVALUATING_CONFIRMATION,
        provenance=provenance,
        champion_hash=champion,
        evaluation_key="confirmation-look-2",
        reason="retry after restart",
        actor="replacement-controller",
        payload={
            "look": "look-2",
            "previous_evaluation_key": "confirmation-look-1",
            "prespecified_cumulative_look": True,
        },
    )
    state = registry.reconstruct()
    confirmation_events = [
        event
        for event in state.events
        if event.transition == Transition.EVALUATION_CONFIRMATION_STARTED
    ]
    assert retry == second
    assert len(confirmation_events) == 2
    assert state.candidates[candidate].evaluation_key == "confirmation-look-2"


def test_append_rejects_illegal_transition_without_writing(tmp_path, provenance):
    registry, _ = bootstrap_registry(tmp_path, provenance)
    with pytest.raises(IllegalTransitionError, match="not discovered"):
        registry.transition_candidate(
            digest("never-discovered"),
            "candidates/claimed/missing",
            CandidateState.CLAIMED,
            provenance=provenance,
            champion_hash=digest("champion-0"),
            reason="invalid claim",
            actor="test-controller",
        )
    assert registry.reconstruct().last_sequence == 1


def test_replay_rejects_validly_hashed_illegal_transition(tmp_path, provenance):
    registry, bootstrap = bootstrap_registry(tmp_path, provenance)
    illegal = PromotionEvent.build(
        sequence=2,
        previous_hash=bootstrap.event_hash,
        timestamp_utc="2026-07-28T12:01:00.000000Z",
        provenance=provenance,
        candidate_hash=digest("not-discovered"),
        candidate_path="candidates/claimed/not-discovered",
        champion_hash=digest("champion-0"),
        transition=Transition.CANDIDATE_CLAIMED,
        evaluation_key=None,
        reason="crafted invalid transition",
        actor="test-controller",
    )
    atomic_write_json(
        registry.events_dir / "00000000000000000002.json", illegal.to_dict()
    )

    with pytest.raises(IllegalTransitionError, match="not discovered"):
        registry.reconstruct()


def test_replay_detects_hash_corruption_chain_break_and_gaps(
    tmp_path, provenance
):
    registry, _ = bootstrap_registry(tmp_path, provenance)
    registry.transition_candidate(
        digest("candidate-corrupt"),
        "candidates/claimed/corrupt",
        CandidateState.DISCOVERED,
        provenance=provenance,
        champion_hash=digest("champion-0"),
        reason="candidate discovered",
        actor="test-controller",
    )
    second_path = registry.events_dir / "00000000000000000002.json"
    value = json.loads(second_path.read_text(encoding="utf-8"))
    value["reason"] = "tampered without rehash"
    atomic_write_json(second_path, value)
    with pytest.raises(RegistryCorruptionError, match="hash mismatch"):
        registry.reconstruct()

    value["event_hash"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "event_hash"}
    )
    value["previous_hash"] = digest("wrong-previous")
    value["event_hash"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "event_hash"}
    )
    atomic_write_json(second_path, value)
    with pytest.raises(RegistryCorruptionError, match="previous hash"):
        registry.reconstruct()

    gap_path = registry.events_dir / "00000000000000000003.json"
    os.replace(second_path, gap_path)
    with pytest.raises(RegistryCorruptionError, match="sequence gap"):
        registry.reconstruct()


def test_candidate_hash_path_contradictions_are_rejected(tmp_path, provenance):
    registry, _ = bootstrap_registry(tmp_path, provenance)
    path = "candidates/claimed/duplicate-name"
    registry.transition_candidate(
        digest("candidate-a"),
        path,
        CandidateState.DISCOVERED,
        provenance=provenance,
        champion_hash=digest("champion-0"),
        reason="candidate A discovered",
        actor="test-controller",
    )
    with pytest.raises(RegistryCorruptionError, match="recorded with both"):
        registry.transition_candidate(
            digest("candidate-b"),
            path,
            CandidateState.DISCOVERED,
            provenance=provenance,
            champion_hash=digest("champion-0"),
            reason="candidate B collision",
            actor="test-controller",
        )
    assert registry.reconstruct().last_sequence == 2


def test_candidate_path_moves_are_explicit_and_reconstructable(
    tmp_path, provenance
):
    registry, _ = bootstrap_registry(tmp_path, provenance)
    candidate = digest("moving-candidate")
    registry.transition_candidate(
        candidate,
        "inbox/moving-candidate",
        CandidateState.DISCOVERED,
        provenance=provenance,
        champion_hash=digest("champion-0"),
        reason="candidate discovered",
        actor="test-controller",
    )
    claimed = registry.transition_candidate(
        candidate,
        "candidates/claimed/moving-candidate",
        CandidateState.CLAIMED,
        provenance=provenance,
        champion_hash=digest("champion-0"),
        reason="candidate atomically claimed",
        actor="test-controller",
    )
    assert claimed.payload["previous_candidate_path"] == "inbox/moving-candidate"
    assert (
        registry.reconstruct().candidates[candidate].candidate_path
        == "candidates/claimed/moving-candidate"
    )
    assert (
        registry.transition_candidate(
            candidate,
            "candidates/claimed/moving-candidate",
            CandidateState.CLAIMED,
            provenance=provenance,
            champion_hash=digest("champion-0"),
            reason="restart claim retry",
            actor="replacement-controller",
        )
        == claimed
    )

    with pytest.raises(
        RegistryCorruptionError, match="without matching previous_candidate_path"
    ):
        registry.append_event(
            Transition.EVALUATION_INTEGRITY_STARTED,
            provenance=provenance,
            candidate_hash=candidate,
            candidate_path="candidates/quarantined/moving-candidate",
            champion_hash=digest("champion-0"),
            evaluation_key="integrity-invalid-move",
            reason="crafted path contradiction",
            actor="test-controller",
        )


def test_crash_temporary_files_and_index_remnants_are_ignored(
    tmp_path, provenance
):
    registry, _ = bootstrap_registry(tmp_path, provenance)
    (registry.events_dir / ".00000000000000000002.json.dead.tmp").write_bytes(
        b'{"truncated":'
    )
    (registry.events_dir / "00000000000000000002.json.tmp").write_bytes(
        b'{"truncated":'
    )
    (registry.events_dir / "registry-index.sqlite").write_bytes(b"not authoritative")

    state = EventRegistry(tmp_path / "promotion").reconstruct()
    assert state.last_sequence == 1
    assert state.current_champion_hash == digest("champion-0")


def test_controller_lock_is_nonblocking_and_releasable(tmp_path):
    path = tmp_path / "controller.lock"
    first = ControllerLock(path, owner="controller-one").acquire()
    second = ControllerLock(path, owner="controller-two")
    try:
        python_path = str(Path(__file__).resolve().parents[1])
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (python_path, environment.get("PYTHONPATH")))
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "from risk_score.promotion_state import "
                    "ControllerLock, ControllerLockError\n"
                    "try:\n"
                    f"    ControllerLock({str(path)!r}, owner='child').acquire()\n"
                    "except ControllerLockError:\n"
                    "    sys.exit(23)\n"
                    "sys.exit(0)\n"
                ),
            ],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 23, result.stderr
        with pytest.raises(ControllerLockError, match="another controller"):
            second.acquire()
    finally:
        first.release()

    with second:
        assert second.acquired
    assert not second.acquired
    assert path.exists()


def test_champion_bootstrap_and_pass_report_cas_are_idempotent(
    tmp_path, provenance
):
    champion_path = tmp_path / "champion.json"
    initial = bootstrap_champion(
        champion_path,
        champion_hash=digest("champion-0"),
        generation_id="generation-0",
        provenance=provenance,
        actor="test-controller",
        timestamp_utc="2026-07-28T12:00:00.000000Z",
    )
    retry = bootstrap_champion(
        champion_path,
        champion_hash=digest("champion-0"),
        generation_id="generation-0",
        provenance=provenance,
        actor="test-controller",
    )
    assert retry == initial

    candidate = digest("candidate-cas")
    report_path = tmp_path / "pass-report.json"
    report_hash = write_pass_report(
        report_path,
        provenance,
        candidate_hash=candidate,
        tested_champion_hash=digest("champion-0"),
    )
    promoted = compare_and_swap_champion(
        champion_path,
        expected_champion_hash=digest("champion-0"),
        candidate_hash=candidate,
        generation_id="generation-1",
        pass_report_path=report_path,
        pass_report_hash=report_hash,
        evaluation_key="confirmation-1",
        provenance=provenance,
        actor="test-controller",
        timestamp_utc="2026-07-28T14:00:00.000000Z",
    )
    before_retry = champion_path.read_bytes()
    promoted_retry = compare_and_swap_champion(
        champion_path,
        expected_champion_hash=digest("champion-0"),
        candidate_hash=candidate,
        generation_id="generation-1",
        pass_report_path=report_path,
        pass_report_hash=report_hash,
        evaluation_key="confirmation-1",
        provenance=provenance,
        actor="test-controller",
    )

    assert promoted_retry == promoted
    assert champion_path.read_bytes() == before_retry
    assert load_champion(champion_path).champion_hash == candidate
    assert promoted.previous_champion_hash == digest("champion-0")
    assert promoted.pass_report_hash == report_hash


def test_champion_cas_rejects_stale_expected_champion(tmp_path, provenance):
    champion_path = tmp_path / "champion.json"
    bootstrap_champion(
        champion_path,
        champion_hash=digest("champion-0"),
        generation_id="generation-0",
        provenance=provenance,
        actor="test-controller",
    )
    candidate_a = digest("candidate-a")
    report_a = tmp_path / "report-a.json"
    report_a_hash = write_pass_report(
        report_a,
        provenance,
        candidate_hash=candidate_a,
        tested_champion_hash=digest("champion-0"),
    )
    compare_and_swap_champion(
        champion_path,
        expected_champion_hash=digest("champion-0"),
        candidate_hash=candidate_a,
        generation_id="generation-a",
        pass_report_path=report_a,
        pass_report_hash=report_a_hash,
        evaluation_key="confirmation-1",
        provenance=provenance,
        actor="test-controller",
    )

    candidate_b = digest("candidate-b")
    report_b = tmp_path / "report-b.json"
    report_b_hash = write_pass_report(
        report_b,
        provenance,
        candidate_hash=candidate_b,
        tested_champion_hash=digest("champion-0"),
    )
    before = champion_path.read_bytes()
    with pytest.raises(StaleChampionError, match="stale tested champion"):
        compare_and_swap_champion(
            champion_path,
            expected_champion_hash=digest("champion-0"),
            candidate_hash=candidate_b,
            generation_id="generation-b",
            pass_report_path=report_b,
            pass_report_hash=report_b_hash,
            evaluation_key="confirmation-1",
            provenance=provenance,
            actor="test-controller",
        )
    assert champion_path.read_bytes() == before


def test_champion_cas_rejects_nonfinal_or_mismatched_report(
    tmp_path, provenance
):
    champion_path = tmp_path / "champion.json"
    bootstrap_champion(
        champion_path,
        champion_hash=digest("champion-0"),
        generation_id="generation-0",
        provenance=provenance,
        actor="test-controller",
    )
    candidate = digest("candidate-invalid-report")
    report_path = tmp_path / "invalid-report.json"
    report_hash = write_pass_report(
        report_path,
        provenance,
        candidate_hash=candidate,
        tested_champion_hash=digest("champion-0"),
        finalized=False,
    )
    with pytest.raises(PassReportError, match="not finalized"):
        compare_and_swap_champion(
            champion_path,
            expected_champion_hash=digest("champion-0"),
            candidate_hash=candidate,
            generation_id="generation-invalid",
            pass_report_path=report_path,
            pass_report_hash=report_hash,
            evaluation_key="confirmation-1",
            provenance=provenance,
            actor="test-controller",
        )

    report_hash = write_pass_report(
        report_path,
        provenance,
        candidate_hash=digest("different-candidate"),
        tested_champion_hash=digest("champion-0"),
    )
    with pytest.raises(PassReportError, match="candidate_hash mismatch"):
        compare_and_swap_champion(
            champion_path,
            expected_champion_hash=digest("champion-0"),
            candidate_hash=candidate,
            generation_id="generation-invalid",
            pass_report_path=report_path,
            pass_report_hash=report_hash,
            evaluation_key="confirmation-1",
            provenance=provenance,
            actor="test-controller",
        )


def test_bootstrap_rejects_different_existing_champion(tmp_path, provenance):
    champion_path = tmp_path / "champion.json"
    bootstrap_champion(
        champion_path,
        champion_hash=digest("champion-0"),
        generation_id="generation-0",
        provenance=provenance,
        actor="test-controller",
    )
    with pytest.raises(ChampionConflictError, match="conflicts"):
        bootstrap_champion(
            champion_path,
            champion_hash=digest("champion-other"),
            generation_id="generation-other",
            provenance=provenance,
            actor="test-controller",
        )


def test_reference_pins_and_retention_queries_are_safe(tmp_path, provenance):
    registry, _ = bootstrap_registry(tmp_path, provenance)
    candidate = digest("superseded-candidate")
    path = "candidates/claimed/superseded"
    registry.transition_candidate(
        candidate,
        path,
        CandidateState.DISCOVERED,
        provenance=provenance,
        champion_hash=digest("champion-0"),
        reason="candidate discovered",
        actor="test-controller",
    )
    registry.transition_candidate(
        candidate,
        path,
        CandidateState.SUPERSEDED,
        provenance=provenance,
        champion_hash=digest("champion-0"),
        reason="newer checkpoint coalesced",
        actor="test-controller",
    )
    state = registry.reconstruct()
    assert state.can_delete(candidate)
    assert not state.can_delete(provenance.original_hash)
    assert not state.can_delete(digest("champion-0"))

    first_pin = registry.pin_reference(
        "incident-42",
        candidate,
        kind="incident-investigation",
        owner="operator",
        provenance=provenance,
        champion_hash=digest("champion-0"),
        reason="retain while incident remains open",
        actor="test-controller",
    )
    retry_pin = registry.pin_reference(
        "incident-42",
        candidate,
        kind="incident-investigation",
        owner="operator",
        provenance=provenance,
        champion_hash=digest("champion-0"),
        reason="retain while incident remains open",
        actor="test-controller",
    )
    assert retry_pin == first_pin
    state = registry.reconstruct()
    assert not state.can_delete(candidate)
    assert any(
        reason.startswith("pin:incident-42")
        for reason in state.retention_status(candidate).reasons
    )

    registry.unpin_reference(
        "incident-42",
        provenance=provenance,
        champion_hash=digest("champion-0"),
        reason="incident closed",
        actor="test-controller",
    )
    historical_retry = registry.pin_reference(
        "incident-42",
        candidate,
        kind="incident-investigation",
        owner="operator",
        provenance=provenance,
        champion_hash=digest("champion-0"),
        reason="retain while incident remains open",
        actor="test-controller",
    )
    assert historical_retry == first_pin
    assert registry.reconstruct().can_delete(candidate)
    assert (
        registry.unpin_reference(
            "incident-42",
            provenance=provenance,
            champion_hash=digest("champion-0"),
            reason="idempotent close retry",
            actor="test-controller",
        )
        is None
    )
    assert registry.reconstruct().can_delete(candidate)
