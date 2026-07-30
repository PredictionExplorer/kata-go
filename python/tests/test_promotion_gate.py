import copy
import hashlib
import json

import pytest

from risk_score.promotion_gate import (
    CONTINUE_TO_LOOK_2,
    EXPECTED_POLICY_HASH,
    FAIL,
    INCONCLUSIVE,
    PASS,
    PINNED_POLICY_REGISTRY,
    PROMOTE,
    STOP_HARM,
    STOP_MAXIMUM_INCONCLUSIVE,
    V1_POLICY_HASH,
    V1_POLICY_VERSION,
    V2_POLICY_HASH,
    V2_POLICY_VERSION,
    evaluate_promotion_gate,
    policy_hash,
    validate_policy,
)
from risk_score.paired_stats import (
    V1_POLICY_PATH,
    exact_zero_event_upper_bound,
    load_policy,
)
from risk_score.promotion_state import (
    atomic_write_json,
    load_finalized_pass_report,
    sha256_file,
)


def sha(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def canonical_hash(value):
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_file_hash(value):
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256((text + "\n").encode("utf-8")).hexdigest()


def metric(
    name,
    *,
    lower,
    upper,
    alpha,
    color_pairs,
    position_ids,
    schedule_id,
    suite,
):
    bootstrap_seed = int.from_bytes(
        hashlib.sha256(f"20260728:{name}".encode("utf-8")).digest()[:8],
        "big",
    )
    nominal_confidence = 0.99 if alpha in (0.002, 0.008) else 0.95
    return {
        "metric": name,
        "available": True,
        "complete": True,
        "estimate": (lower + upper) / 2.0,
        "lower_bound": lower,
        "upper_bound": upper,
        "one_sided_alpha": alpha,
        "allocated_one_sided_confidence": 1.0 - alpha,
        "nominal_one_sided_confidence": nominal_confidence,
        "color_pairs": color_pairs,
        "position_clusters": len(position_ids),
        "degrees_of_freedom": len(position_ids) - 1,
        "small_cluster_correction": {
            "method": "CR1_BESSEL_WITH_STUDENT_T",
            "variance_multiplier": len(position_ids) / (len(position_ids) - 1.0),
        },
        "position_values": [
            {
                "position_id": position_id,
                "pair_count": color_pairs // len(position_ids),
                "schedule_ids": [schedule_id],
                "stratum": {"schedule_id": schedule_id, "suite": suite},
                "value": (lower + upper) / 2.0,
            }
            for position_id in position_ids
        ],
        "bootstrap": {
            "method": (
                "stratified_wild_position_cluster_rademacher_"
                "centered_within_declared_strata"
            ),
            "replications": 9999,
            "seed": bootstrap_seed,
            "stratum_dimensions": ["schedule_id", "suite"],
            "stratum_cluster_counts": [
                {
                    "schedule_id": schedule_id,
                    "suite": suite,
                    "position_clusters": len(position_ids),
                }
            ],
            "lower_bound": lower,
            "upper_bound": upper,
        },
    }


def passing_evidence(
    look_number=1,
    policy=None,
    *,
    authoritative_manifest=False,
):
    policy = load_policy() if policy is None else policy
    candidate = sha("candidate")
    champion = sha("champion")
    original = sha("original")
    stage_look = next(
        look
        for look in policy["evaluation_stages"]["stage_3_promotion_confirmation"][
            "looks"
        ]
        if look["look_number"] == look_number
    )
    alpha_look = next(
        look
        for look in policy["confidence"]["sequential_testing"]["looks"]
        if look["look_number"] == look_number
    )
    routine_alpha = alpha_look["routine_one_sided_alpha"]
    catastrophe_alpha = alpha_look["catastrophe_one_sided_alpha"]
    runner_contract = (
        "risk-score-pair-safe-evaluation-runner-v3"
        if policy.get("schema_version") == 2
        else "risk-score-pair-safe-evaluation-runner-v2"
    )
    powered_pairs = stage_look["powered_ordinary_color_pairs_per_matchup"]
    standard_pairs = stage_look["standard_ordinary_color_pairs"]
    lead_40_pairs = stage_look["lead_40_color_pairs"]
    lead_80_pairs = stage_look["lead_80_color_pairs"]
    cluster_minima = stage_look.get(
        "minimum_independent_position_clusters",
        {
            "powered_candidate_vs_champion": 8,
            "powered_candidate_vs_original": 8,
            "standard_candidate_vs_original": 8,
            "lead_40": 8,
            "lead_80": 8,
        },
    )
    powered_config_hash = sha("powered-config")
    standard_config_hash = sha("standard-config")
    binary_hash = sha("katago-binary")
    definitions = {
        "powered_candidate_vs_champion": {
            "comparison": "candidate-vs-champion-powered",
            "suite": "confirmation",
            "search_mode": "powered",
            "reference_hash": champion,
            "pairs": powered_pairs,
            "minimum_clusters": cluster_minima[
                "powered_candidate_vs_champion"
            ],
            "visits": 2000,
        },
        "powered_candidate_vs_original": {
            "comparison": "candidate-vs-original-powered",
            "suite": "confirmation",
            "search_mode": "powered",
            "reference_hash": original,
            "pairs": powered_pairs,
            "minimum_clusters": cluster_minima[
                "powered_candidate_vs_original"
            ],
            "visits": 2000,
        },
        "standard_candidate_vs_original": {
            "comparison": "candidate-vs-original-standard",
            "suite": "confirmation",
            "search_mode": "standard",
            "reference_hash": original,
            "pairs": standard_pairs,
            "minimum_clusters": cluster_minima[
                "standard_candidate_vs_original"
            ],
            "visits": 800,
        },
        "lead_40": {
            "comparison": "candidate-vs-champion-powered-lead-40",
            "suite": "lead-40",
            "search_mode": "powered",
            "reference_hash": champion,
            "pairs": lead_40_pairs,
            "minimum_clusters": cluster_minima["lead_40"],
            "visits": 2000,
        },
        "lead_80": {
            "comparison": "candidate-vs-champion-powered-lead-80",
            "suite": "lead-80",
            "search_mode": "powered",
            "reference_hash": champion,
            "pairs": lead_80_pairs,
            "minimum_clusters": cluster_minima["lead_80"],
            "visits": 2000,
        },
    }
    cells = {}
    suite_entries = {
        cell_name: {
            "suite": definition["suite"],
            "suite_hash": sha("suite-bank-" + cell_name),
            "schedule_id": f"schedule-{cell_name}",
            "schedule_hash": sha(f"schedule-{cell_name}"),
            "color_pairs": definition["pairs"],
            "position_ids": sorted(
                f"{cell_name}-position-{index}"
                for index in range(definition["minimum_clusters"])
            ),
        }
        for cell_name, definition in definitions.items()
    }
    discovery_schedule_hash = sha("discovery-schedule")
    authoritative_entries = {
        cell_name: {
            "cell_name": cell_name,
            "stage": "stage-3",
            "look": f"look-{look_number}",
            "look_number": look_number,
            "comparison": definitions[cell_name]["comparison"],
            "suite": definitions[cell_name]["suite"],
            "search_mode": definitions[cell_name]["search_mode"],
            "visits": definitions[cell_name]["visits"],
            "color_pairs": definitions[cell_name]["pairs"],
            "minimum_independent_position_clusters": definitions[cell_name][
                "minimum_clusters"
            ],
            "independent_cluster_ids": [
                sha("cluster:" + position_id)
                for position_id in suite_entries[cell_name]["position_ids"]
            ],
            "position_ids": suite_entries[cell_name]["position_ids"],
            "bank_hash": suite_entries[cell_name]["suite_hash"],
            "schedule_hash": suite_entries[cell_name]["schedule_hash"],
            "schedule_id": suite_entries[cell_name]["schedule_id"],
            "policy_hash": policy_hash(policy),
            "policy_version": policy["policy_version"],
            "source_revision": policy["frozen_plan"]["source_revision"],
        }
        for cell_name in definitions
    }
    for entry in authoritative_entries.values():
        entry["independent_cluster_ids_hash"] = canonical_hash(
            entry["independent_cluster_ids"]
        )
        entry["position_ids_hash"] = canonical_hash(entry["position_ids"])
    suite_payload = {
        **(
            {
                "schemaVersion": 2,
                "manifestContract": (
                    "risk-score-authoritative-evaluation-manifest-v2"
                ),
            }
            if authoritative_manifest
            else {"schema_version": 1}
        ),
        "policy_hash": policy_hash(policy),
        "source_revision": policy["frozen_plan"]["source_revision"],
        "discovery_schedule_hash": discovery_schedule_hash,
        "cells": (
            list(authoritative_entries.values())
            if authoritative_manifest
            else suite_entries
        ),
    }
    suite_manifest = dict(suite_payload)
    suite_manifest["manifestPayloadSha256"] = canonical_hash(suite_payload)
    suite_manifest_hash = canonical_file_hash(suite_manifest)
    risk_differences = {}

    def make_metric(name, cell_name, lower, upper, alpha):
        definition = definitions[cell_name]
        return metric(
            name,
            lower=lower,
            upper=upper,
            alpha=alpha,
            color_pairs=definition["pairs"],
            position_ids=suite_entries[cell_name]["position_ids"],
            schedule_id=f"schedule-{cell_name}",
            suite=definition["suite"],
        )

    for cell_name, definition in definitions.items():
        config_hash = (
            powered_config_hash
            if definition["search_mode"] == "powered"
            else standard_config_hash
        )
        schedule_id = f"schedule-{cell_name}"
        schedule_hash = suite_entries[cell_name]["schedule_hash"]
        suite_hash = suite_entries[cell_name]["suite_hash"]
        position_ids = suite_entries[cell_name]["position_ids"]
        if definition["search_mode"] == "powered":
            search_settings = {
                "use_score_maximizing_utility": True,
                "win_weight": 4.0,
                "score_power": 1.5,
                "score_scale": 20.0,
            }
        else:
            search_settings = {"use_score_maximizing_utility": False}
        runner_spec = {
            "candidate_model_sha": candidate,
            "reference_model_sha": definition["reference_hash"],
            "original_model_sha": original,
            "config_sha": config_hash,
            "schedule_sha": schedule_hash,
            "policy_sha": policy_hash(policy),
            "comparison": definition["comparison"],
            "suite": definition["suite"],
            "stage": "stage-3",
            "look": f"look-{look_number}",
            "topology": "gpu7-eight-processes",
            "suite_manifest_sha": suite_manifest_hash,
            "suite_bank_sha": suite_hash,
            "schedule_id": schedule_id,
        }
        if policy.get("schema_version") == 2:
            runner_spec["max_visits"] = definition["visits"]
        execution_manifest = {
            "katagoBinarySha256": binary_hash,
            "moveTraces": True,
            "extraArgv": [],
            "effectiveShardCount": 1,
            "effectiveMaxParallelism": 1,
            "maxAttempts": 2,
            "cwd": "/frozen/evaluation",
            "timeout": None,
            "replaceEnv": False,
            "effectiveEnvironmentSha256": sha("runner-environment"),
        }
        runner_evaluation_key = "eval-" + canonical_hash(
            {
                "runnerContract": runner_contract,
                "evaluationSpec": runner_spec,
                "execution": execution_manifest,
            }
        )
        runner_payload = {
            "schemaVersion": 1,
            "runnerContract": runner_contract,
            "evaluationKey": runner_evaluation_key,
            "evaluationSpec": runner_spec,
            "execution": execution_manifest,
            "cell": {
                "comparison": definition["comparison"],
                "suite": definition["suite"],
                "stage": "stage-3",
                "look": f"look-{look_number}",
                "gameCount": 2 * definition["pairs"],
                "colorPairCount": definition["pairs"],
            },
            "schedule": {
                "sha256": schedule_hash,
                "scheduleId": schedule_id,
                "rowCount": 2 * definition["pairs"],
                "pairCount": definition["pairs"],
                "suiteManifestSha256": suite_manifest_hash,
                "suiteBankSha256": suite_hash,
                **(
                    {
                        "manifestCell": authoritative_entries[cell_name],
                        "manifestCellSha256": canonical_hash(
                            authoritative_entries[cell_name]
                        ),
                    }
                    if authoritative_manifest
                    else {}
                ),
            },
            "results": {
                "path": "results.jsonl",
                "sha256": sha("results-" + cell_name),
                "rowCount": 2 * definition["pairs"],
            },
            "moves": {
                "path": "moves.jsonl",
                "sha256": sha("moves-" + cell_name),
                "rowCount": 1000,
            },
            "shards": [],
        }
        if policy.get("schema_version") == 2:
            runner_payload["cell"]["maxVisits"] = definition["visits"]
        runner_manifest = dict(runner_payload)
        runner_manifest["manifestPayloadSha256"] = canonical_hash(runner_payload)
        runner_manifest_hash = canonical_file_hash(runner_manifest)
        execution_hash = canonical_hash(execution_manifest)
        metrics = {}
        risks = {}
        if cell_name == "powered_candidate_vs_champion":
            metrics = {
                "realized_utility": make_metric(
                    "realized_utility", cell_name, 0.10, 0.30, routine_alpha
                ),
                "win_rate": make_metric(
                    "win_rate", cell_name, 0.50, 0.60, routine_alpha
                ),
            }
            risk_names = ("final_20", "final_50", "high_confidence_loss")
        elif cell_name == "powered_candidate_vs_original":
            metrics = {
                "realized_utility": make_metric(
                    "realized_utility", cell_name, 0.08, 0.25, routine_alpha
                ),
                "win_rate": make_metric(
                    "win_rate", cell_name, 0.49, 0.58, routine_alpha
                ),
            }
            risk_names = ()
        elif cell_name == "standard_candidate_vs_original":
            metrics = {
                "win_rate": make_metric(
                    "win_rate", cell_name, 0.46, 0.56, routine_alpha
                )
            }
            risk_names = ()
        elif cell_name == "lead_40":
            risk_names = ("lead_40_loss", "targeted_lead_40_suite_loss")
        else:
            risk_names = ("lead_80_loss", "targeted_lead_80_suite_loss")
        for risk_name in risk_names:
            risk = make_metric(
                risk_name, cell_name, -0.002, 0.001, catastrophe_alpha
            )
            risk.update(
                {
                    "candidate_events": 1,
                    "reference_events": 1,
                    "direction": "candidate_minus_reference",
                    "matched_within_game": True,
                }
            )
            risks[risk_name] = risk
            risk_differences[risk_name] = copy.deepcopy(risk)
        data_binding = {
            "candidate_hash": candidate,
            "reference_hash": definition["reference_hash"],
            "comparison": definition["comparison"],
            "suite": definition["suite"],
            "suite_hash": suite_hash,
            "schedule_id": schedule_id,
            "schedule_hash": schedule_hash,
            "config_hash": config_hash,
            "runner_manifest_hash": runner_manifest_hash,
            "execution_hash": execution_hash,
            "katago_binary_hash": binary_hash,
        }
        artifact = {
            "schema_version": 1,
            "finalized": True,
            "policy_hash": policy_hash(policy),
            "look": {"number": look_number},
            "counts": {"color_pairs": definition["pairs"]},
            "data_binding": data_binding,
            "metrics": metrics,
            "risk_differences": risks,
        }
        artifact_hash = canonical_hash(artifact)
        metric_names = sorted(list(metrics) + list(risks))
        statistics_manifest = {
            "schema_version": 1,
            "finalized": True,
            "cell_name": cell_name,
            **data_binding,
            "statistics_artifact_hash": artifact_hash,
            "color_pairs": definition["pairs"],
            "position_ids": position_ids,
            "metric_names": metric_names,
        }
        cells[cell_name] = {
            "comparison": definition["comparison"],
            "suite": definition["suite"],
            "stage": "stage-3",
            "look": f"look-{look_number}",
            "topology": "gpu7-eight-processes",
            "search_mode": definition["search_mode"],
            "candidate_hash": candidate,
            "reference_hash": definition["reference_hash"],
            "visits": definition["visits"],
            "color_pairs": definition["pairs"],
            "config_hash": config_hash,
            "schedule_id": schedule_id,
            "schedule_hash": schedule_hash,
            "suite_hash": suite_hash,
            "katago_binary_hash": binary_hash,
            "runner_manifest": runner_manifest,
            "runner_manifest_hash": runner_manifest_hash,
            "execution_manifest": execution_manifest,
            "execution_hash": execution_hash,
            "statistics_artifact": artifact,
            "statistics_artifact_hash": artifact_hash,
            "statistics_manifest": statistics_manifest,
            "statistics_manifest_hash": canonical_hash(statistics_manifest),
            "candidate_search": copy.deepcopy(search_settings),
            "reference_search": copy.deepcopy(search_settings),
            "validation": {"promotion_valid": True},
        }

    lead_positions = sorted(
        set(suite_entries["lead_40"]["position_ids"]).union(
            suite_entries["lead_80"]["position_ids"]
        )
    )
    combined_pairs = lead_40_pairs + lead_80_pairs
    combined_metric = metric(
        "combined_lead_realized_utility",
        lower=-0.01,
        upper=0.10,
        alpha=routine_alpha,
        color_pairs=combined_pairs,
        position_ids=lead_positions,
        schedule_id="combined-lead-schedules",
        suite="combined-lead",
    )
    for row in combined_metric["position_values"]:
        source = "lead-40" if row["position_id"].startswith("lead_40") else "lead-80"
        source_cell = "lead_40" if source == "lead-40" else "lead_80"
        row["schedule_ids"] = [f"schedule-{source_cell}"]
        row["stratum"] = {
            "schedule_id": f"schedule-{source_cell}",
            "suite": source,
        }
    combined_metric["bootstrap"]["stratum_cluster_counts"] = [
        {
            "schedule_id": "schedule-lead_40",
            "suite": "lead-40",
            "position_clusters": len(
                suite_entries["lead_40"]["position_ids"]
            ),
        },
        {
            "schedule_id": "schedule-lead_80",
            "suite": "lead-80",
            "position_clusters": len(
                suite_entries["lead_80"]["position_ids"]
            ),
        },
    ]
    lead_source_hashes = {
        name: cells[name]["statistics_artifact_hash"]
        for name in ("lead_40", "lead_80")
    }
    combined_artifact = {
        "schema_version": 1,
        "finalized": True,
        "policy_hash": policy_hash(policy),
        "look": {"number": look_number},
        "source_statistics_artifact_hashes": lead_source_hashes,
        "counts": {"color_pairs": combined_pairs},
        "position_ids": lead_positions,
        "metrics": {"combined_lead_realized_utility": combined_metric},
    }
    combined_artifact_hash = canonical_hash(combined_artifact)
    combined_manifest = {
        "schema_version": 1,
        "finalized": True,
        "source_cells": ["lead_40", "lead_80"],
        "source_statistics_artifact_hashes": lead_source_hashes,
        "color_pairs": combined_pairs,
        "position_ids": lead_positions,
        "metric_names": ["combined_lead_realized_utility"],
        "statistics_artifact_hash": combined_artifact_hash,
    }
    runner_specs = [
        cells[name]["runner_manifest"]["evaluationSpec"] for name in definitions
    ]
    config_bundle_hash = canonical_hash(
        sorted({cell["config_hash"] for cell in cells.values()})
    )
    schedule_bundle_hash = canonical_hash(
        sorted({cell["schedule_hash"] for cell in cells.values()})
    )
    return {
        "schema_version": 1,
        "policy_version": policy["policy_version"],
        "policy_hash": policy_hash(policy),
        "candidate_hash": candidate,
        "champion_hash": champion,
        "original_hash": original,
        "confirmation_finalized": True,
        "evaluation_key": "matrix-" + canonical_hash(runner_specs),
        "config_hash": config_bundle_hash,
        "schedule_hash": schedule_bundle_hash,
        "look_number": look_number,
        "evaluation_stage": 3,
        "thresholds_overridden": False,
        "alpha_allocation_overridden": False,
        "attempt": {
            "generation_id": "generation-7",
            "attempt_number": 1,
            "promotions_for_generation": 0,
        },
        "confirmation_matrix": cells,
        "combined_lead_artifact": combined_artifact,
        "combined_lead_artifact_hash": combined_artifact_hash,
        "combined_lead_manifest": combined_manifest,
        "combined_lead_manifest_hash": canonical_hash(combined_manifest),
        "risk_differences": risk_differences,
        "discovery": {
            "stage_1_passed": True,
            "stage_2_passed": True,
            "dominated_by_later_safe_finalist": False,
            "confirmation_schedule_independent": True,
            "confirmation_candidate_count": 1,
        },
        "validity": {
            "promotion_valid": True,
            "missing_games": 0,
            "duplicate_game_ids": 0,
            "incomplete_pairs": 0,
            "duplicate_pair_members": 0,
            "resignations": 0,
            "turn_limits": 0,
            "unresolved_rows": 0,
            "structural_errors": 0,
            "perspective_violations": 0,
            "clamp_violations": 0,
            "endpoint_violations": 0,
            "nonfinite_violations": 0,
            "decomposition_violations": 0,
            "resolved_missing_numeric_scores": 2,
            "true_no_results": 0,
            "total_games": 2 * sum(item["pairs"] for item in definitions.values()),
            "true_no_result_rate": 0.0,
            "full_move_diagnostics": True,
        },
        "exploitability": {
            "stage_0_passed": True,
            "fixed_analysis_positions": 256,
            "fixed_analysis_visits": 200,
            "exploitability_sentinel_positions": 16,
            "exploitability_sentinel_visits": 2000,
            "hard_tactical_failures": 0,
            "hard_exploitability_failures": 0,
            "unresolved_failures": 0,
            "model_runtime_errors": 0,
            "selected_move_endpoint_mass_dominated": False,
            "visit_stability_acceptable": True,
        },
        "provenance": {
            "complete": True,
            "immutable_inputs": True,
            "immutable_original": True,
            "candidate_hash": candidate,
            "champion_hash": champion,
            "original_hash": original,
            "policy_hash": policy_hash(policy),
            "source_revision_hash": policy["frozen_plan"]["source_revision"],
            "binary_hash": binary_hash,
            "config_hashes": {
                "powered_match": powered_config_hash,
                "standard_match": standard_config_hash,
            },
            "schedule_hashes": {
                name: cell["schedule_hash"] for name, cell in cells.items()
            },
            "suite_hashes": {
                **{
                    name: cell["suite_hash"] for name, cell in cells.items()
                },
                "tactical": sha("tactical-suite"),
                "exploitability": sha("exploitability-suite"),
            },
            "discovery_schedule_hash": discovery_schedule_hash,
            "suite_manifest": suite_manifest,
            "suite_manifest_hash": suite_manifest_hash,
        },
    }


def check_by_code(report, code):
    return next(check for check in report["checks"] if check["code"] == code)


def rehash_statistics_cell(evidence, cell_name):
    cell = evidence["confirmation_matrix"][cell_name]
    artifact_hash = canonical_hash(cell["statistics_artifact"])
    cell["statistics_artifact_hash"] = artifact_hash
    cell["statistics_manifest"]["statistics_artifact_hash"] = artifact_hash
    cell["statistics_manifest_hash"] = canonical_hash(cell["statistics_manifest"])


def rehash_combined_lead(evidence):
    artifact_hash = canonical_hash(evidence["combined_lead_artifact"])
    evidence["combined_lead_artifact_hash"] = artifact_hash
    evidence["combined_lead_manifest"]["statistics_artifact_hash"] = artifact_hash
    evidence["combined_lead_manifest_hash"] = canonical_hash(
        evidence["combined_lead_manifest"]
    )


def rehash_suite_manifest(evidence):
    manifest = evidence["provenance"]["suite_manifest"]
    payload = dict(manifest)
    payload.pop("manifestPayloadSha256", None)
    manifest["manifestPayloadSha256"] = canonical_hash(payload)
    evidence["provenance"]["suite_manifest_hash"] = canonical_file_hash(manifest)


def test_v1_is_byte_stable_and_remains_a_pinned_historical_policy():
    policy = load_policy(V1_POLICY_PATH)
    validate_policy(policy)
    assert hashlib.sha256(V1_POLICY_PATH.read_bytes()).hexdigest() == (
        "be026cb8142e9e427757f8f8fa4fb3332937fd4a88868b064c07d66ff30154b1"
    )
    assert policy["policy_version"] == V1_POLICY_VERSION
    assert policy_hash(policy) == V1_POLICY_HASH
    assert PINNED_POLICY_REGISTRY[V1_POLICY_VERSION]["policy_hash"] == V1_POLICY_HASH


def test_non_object_evidence_fails_closed_without_raising():
    report = evaluate_promotion_gate([])
    assert report["decision"] == FAIL
    assert report["next_action"] == STOP_HARM
    assert check_by_code(report, "EVIDENCE_OBJECT")["status"] == FAIL


def test_v2_is_the_default_pinned_frozen_policy_with_cumulative_counts():
    policy = load_policy()
    validate_policy(policy)
    assert policy["schema_version"] == 2
    assert policy["policy_version"] == V2_POLICY_VERSION
    assert policy["status"] == "frozen"
    assert policy["supersedes"] == {
        "policy_version": V1_POLICY_VERSION,
        "policy_hash": V1_POLICY_HASH,
    }
    assert policy_hash(policy) == policy_hash(copy.deepcopy(policy))
    assert policy_hash(policy) == V2_POLICY_HASH == EXPECTED_POLICY_HASH
    stage_3 = policy["evaluation_stages"]["stage_3_promotion_confirmation"]
    assert stage_3["look_data_relationship"] == "cumulative_prefix"
    assert stage_3["color_pairs_per_independent_position_cluster"] == 1
    assert [
        (
            look["powered_ordinary_color_pairs_per_matchup"],
            look["standard_ordinary_color_pairs"],
            look["lead_40_color_pairs"],
            look["lead_80_color_pairs"],
        )
        for look in stage_3["looks"]
    ] == [(512, 128, 512, 1024), (1024, 128, 1024, 2048)]
    for look in stage_3["looks"]:
        assert look["minimum_independent_position_clusters"] == {
            "powered_candidate_vs_champion": look[
                "powered_ordinary_color_pairs_per_matchup"
            ],
            "powered_candidate_vs_original": look[
                "powered_ordinary_color_pairs_per_matchup"
            ],
            "standard_candidate_vs_original": look[
                "standard_ordinary_color_pairs"
            ],
            "lead_40": look["lead_40_color_pairs"],
            "lead_80": look["lead_80_color_pairs"],
        }


def test_v1_historical_evidence_still_validates_under_the_v1_gate_contract():
    policy = load_policy(V1_POLICY_PATH)
    evidence = passing_evidence(policy=policy)
    report = evaluate_promotion_gate(evidence, policy=policy)
    assert report["decision"] == PASS
    assert report["policy_version"] == V1_POLICY_VERSION
    assert report["policy_hash"] == V1_POLICY_HASH


@pytest.mark.parametrize(
    "risk_name, source_cell",
    [
        ("final_20", "powered_candidate_vs_champion"),
        ("final_50", "powered_candidate_vs_champion"),
        ("high_confidence_loss", "powered_candidate_vs_champion"),
        ("lead_40_loss", "lead_40"),
        ("lead_80_loss", "lead_80"),
        ("targeted_lead_40_suite_loss", "lead_40"),
        ("targeted_lead_80_suite_loss", "lead_80"),
    ],
)
def test_v2_final_look_zero_event_bound_is_feasible_for_every_risk(
    risk_name, source_cell
):
    policy = load_policy()
    final_look = policy["evaluation_stages"]["stage_3_promotion_confirmation"][
        "looks"
    ][-1]
    alpha = policy["confidence"]["sequential_testing"]["looks"][-1][
        "catastrophe_one_sided_alpha"
    ]
    clusters = final_look["minimum_independent_position_clusters"][source_cell]
    threshold = policy["promotion_thresholds"][
        "candidate_minus_reference_risk_upper_bounds"
    ][risk_name]
    assert exact_zero_event_upper_bound(alpha, clusters) <= threshold


def test_all_complete_confirmation_evidence_passes_deterministically():
    evidence = passing_evidence()
    first = evaluate_promotion_gate(evidence)
    second = evaluate_promotion_gate(copy.deepcopy(evidence))

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first
    assert first["decision"] == PASS
    assert first["next_action"] == PROMOTE
    assert first["continuation_eligible"] is False
    assert first["finalized"] is True
    assert first["tested_champion_hash"] == evidence["champion_hash"]
    assert first["evaluation_key"] == evidence["evaluation_key"]
    assert first["config_hash"] == evidence["config_hash"]
    assert first["schedule_hash"] == evidence["schedule_hash"]
    assert first["reason_codes"] == []
    assert first["ranking_summary"] == {
        "schema_version": 1,
        "source_bound": True,
        "source_cell": "powered_candidate_vs_champion",
        "candidate_hash": evidence["candidate_hash"],
        "look_number": 1,
        "statistics_artifact_hash": evidence["confirmation_matrix"][
            "powered_candidate_vs_champion"
        ]["statistics_artifact_hash"],
        "statistics_manifest_hash": evidence["confirmation_matrix"][
            "powered_candidate_vs_champion"
        ]["statistics_manifest_hash"],
        "realized_powered_utility_lower_bound": 0.1,
        "final50_risk_upper_bound": 0.001,
        "final_50_risk_upper_bound": 0.001,
    }
    assert first["realized_powered_utility_lower_bound"] == 0.1
    assert first["final50_risk_upper_bound"] == 0.001
    assert [check["code"] for check in first["checks"]] == sorted(
        check["code"] for check in first["checks"]
    )
    assert all(check["status"] == PASS for check in first["checks"])
    assert first["alpha_allocation"] == {
        "method": "prespecified_bonferroni_alpha_spending",
        "routine_one_sided_alpha": 0.01,
        "catastrophe_one_sided_alpha": 0.002,
        "routine_allocated_one_sided_confidence": 0.99,
        "catastrophe_allocated_one_sided_confidence": 0.998,
    }


def test_authoritative_v2_suite_manifest_is_bound_through_runner_and_gate():
    evidence = passing_evidence(authoritative_manifest=True)
    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == PASS
    assert report["next_action"] == PROMOTE
    for cell_name in evidence["confirmation_matrix"]:
        prefix = "MATRIX_" + cell_name.upper()
        assert check_by_code(report, prefix + "_RUNNER_MANIFEST_CELL")[
            "status"
        ] == PASS
        assert check_by_code(
            report,
            "SUITE_MANIFEST_"
            + cell_name.upper()
            + "_INDEPENDENT_CLUSTER_IDS",
        )["status"] == PASS


def test_pass_report_is_accepted_by_champion_transaction_loader(tmp_path):
    evidence = passing_evidence()
    report = evaluate_promotion_gate(evidence)
    path = tmp_path / "pass-report.json"
    atomic_write_json(path, report)

    loaded = load_finalized_pass_report(
        path,
        expected_report_hash=sha256_file(path),
    )
    assert loaded.candidate_hash == evidence["candidate_hash"]
    assert loaded.tested_champion_hash == evidence["champion_hash"]
    assert loaded.evaluation_key == evidence["evaluation_key"]
    assert loaded.config_hash == evidence["config_hash"]
    assert loaded.schedule_hash == evidence["schedule_hash"]


def test_second_prespecified_look_uses_extended_counts_and_second_alpha_allocation():
    report = evaluate_promotion_gate(passing_evidence(look_number=2))
    assert report["decision"] == PASS
    assert report["next_action"] == PROMOTE
    assert report["look_number"] == 2
    assert report["alpha_allocation"]["routine_one_sided_alpha"] == 0.04
    assert report["alpha_allocation"]["catastrophe_one_sided_alpha"] == 0.008


@pytest.mark.parametrize(
    "path, value, expected_check",
    [
        (
            (
                "confirmation_matrix",
                "powered_candidate_vs_champion",
                "statistics_artifact",
                "metrics",
                "realized_utility",
                "lower_bound",
            ),
            0.0,
            "POWERED_UTILITY_VS_CHAMPION_LOWER_BOUND",
        ),
        (
            (
                "confirmation_matrix",
                "powered_candidate_vs_champion",
                "statistics_artifact",
                "metrics",
                "win_rate",
                "lower_bound",
            ),
            0.47,
            "POWERED_WIN_RATE_VS_CHAMPION_LOWER_BOUND",
        ),
        (
            (
                "combined_lead_artifact",
                "metrics",
                "combined_lead_realized_utility",
                "lower_bound",
            ),
            -0.05,
            "COMBINED_LEAD_UTILITY_LOWER_BOUND",
        ),
    ],
)
def test_unmet_first_look_margin_continues_when_upper_bound_allows_benefit(
    path, value, expected_check
):
    evidence = passing_evidence()
    target = evidence
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    if path[0] == "confirmation_matrix":
        rehash_statistics_cell(evidence, path[1])
    else:
        rehash_combined_lead(evidence)
    report = evaluate_promotion_gate(evidence)

    assert report["decision"] == INCONCLUSIVE
    assert report["next_action"] == CONTINUE_TO_LOOK_2
    assert report["continuation_eligible"] is True
    assert check_by_code(report, expected_check)["status"] == INCONCLUSIVE


def test_first_look_upper_bound_proving_no_benefit_stops_as_harm():
    evidence = passing_evidence()
    metric_value = evidence["confirmation_matrix"][
        "powered_candidate_vs_champion"
    ]["statistics_artifact"]["metrics"]["realized_utility"]
    metric_value.update(
        {
            "lower_bound": -0.20,
            "estimate": -0.10,
            "upper_bound": 0.0,
        }
    )
    rehash_statistics_cell(evidence, "powered_candidate_vs_champion")

    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert report["next_action"] == STOP_HARM
    assert (
        check_by_code(report, "POWERED_UTILITY_VS_CHAMPION_LOWER_BOUND")[
            "status"
        ]
        == FAIL
    )


def test_final_look_unmet_margin_stops_maximum_inconclusive():
    evidence = passing_evidence(look_number=2)
    metric_value = evidence["confirmation_matrix"][
        "powered_candidate_vs_champion"
    ]["statistics_artifact"]["metrics"]["realized_utility"]
    metric_value["lower_bound"] = 0.0
    rehash_statistics_cell(evidence, "powered_candidate_vs_champion")

    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == INCONCLUSIVE
    assert report["next_action"] == STOP_MAXIMUM_INCONCLUSIVE
    assert report["continuation_eligible"] is False


def test_risk_upper_bound_equal_to_frozen_limit_is_allowed():
    evidence = passing_evidence()
    limit = load_policy()["promotion_thresholds"][
        "candidate_minus_reference_risk_upper_bounds"
    ]["final_20"]
    risk = evidence["confirmation_matrix"]["powered_candidate_vs_champion"][
        "statistics_artifact"
    ]["risk_differences"]["final_20"]
    risk["upper_bound"] = limit
    risk["bootstrap"]["upper_bound"] = limit
    evidence["risk_differences"]["final_20"] = copy.deepcopy(risk)
    rehash_statistics_cell(evidence, "powered_candidate_vs_champion")

    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == PASS
    assert check_by_code(report, "RISK_FINAL_20_UPPER_BOUND")["status"] == PASS


def test_proven_risk_violation_fails_but_unresolved_margin_is_inconclusive():
    evidence = passing_evidence()
    risk = evidence["confirmation_matrix"]["powered_candidate_vs_champion"][
        "statistics_artifact"
    ]["risk_differences"]["final_20"]
    risk["estimate"] = 0.015
    risk["lower_bound"] = 0.01
    risk["upper_bound"] = 0.03
    evidence["risk_differences"]["final_20"] = copy.deepcopy(risk)
    rehash_statistics_cell(evidence, "powered_candidate_vs_champion")
    assert evaluate_promotion_gate(evidence)["decision"] == INCONCLUSIVE

    risk["estimate"] = 0.025
    risk["lower_bound"] = 0.021
    evidence["risk_differences"]["final_20"] = copy.deepcopy(risk)
    rehash_statistics_cell(evidence, "powered_candidate_vs_champion")
    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert check_by_code(report, "RISK_FINAL_20_UPPER_BOUND")["status"] == FAIL


def test_bootstrap_is_reported_as_sensitivity_not_a_second_gate():
    evidence = passing_evidence()
    artifact = evidence["confirmation_matrix"]["powered_candidate_vs_champion"][
        "statistics_artifact"
    ]
    artifact["metrics"]["realized_utility"]["bootstrap"]["lower_bound"] = -1.0
    artifact["risk_differences"]["final_20"]["bootstrap"]["upper_bound"] = 0.5
    evidence["risk_differences"]["final_20"] = copy.deepcopy(
        artifact["risk_differences"]["final_20"]
    )
    rehash_statistics_cell(evidence, "powered_candidate_vs_champion")

    assert evaluate_promotion_gate(evidence)["decision"] == PASS


def test_zero_event_risk_recomputes_exact_bound_from_independent_clusters():
    evidence = passing_evidence()
    risk = evidence["confirmation_matrix"]["powered_candidate_vs_champion"][
        "statistics_artifact"
    ]["risk_differences"]["final_50"]
    risk["candidate_events"] = 0
    risk["reference_events"] = 0
    risk["upper_bound"] = 0.0
    risk["bootstrap"]["upper_bound"] = 0.0
    risk["zero_event_uncertainty_upper_bound"] = 0.0
    risk["zero_event_independent_position_clusters"] = risk["position_clusters"]
    risk["zero_event_uncertainty_method"] = (
        "one_sided_exact_no_event_bound_using_independent_position_clusters"
    )
    evidence["risk_differences"]["final_50"] = copy.deepcopy(risk)
    rehash_statistics_cell(evidence, "powered_candidate_vs_champion")

    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert (
        check_by_code(report, "RISK_FINAL_50_ZERO_EVENT_UNCERTAINTY")["status"]
        == FAIL
    )
    assert (
        check_by_code(report, "RISK_FINAL_50_ZERO_EVENT_BOOTSTRAP_CORRECTION")[
            "status"
        ]
        == FAIL
    )

    exact = exact_zero_event_upper_bound(0.002, risk["position_clusters"])
    risk["zero_event_uncertainty_upper_bound"] = exact
    risk["upper_bound"] = exact
    risk["bootstrap"]["upper_bound"] = exact
    evidence["risk_differences"]["final_50"] = copy.deepcopy(risk)
    rehash_statistics_cell(evidence, "powered_candidate_vs_champion")
    corrected = evaluate_promotion_gate(evidence)
    assert (
        check_by_code(corrected, "RISK_FINAL_50_ZERO_EVENT_UNCERTAINTY")[
            "status"
        ]
        == PASS
    )
    assert (
        check_by_code(corrected, "RISK_FINAL_50_ZERO_EVENT_ANALYTIC_CORRECTION")[
            "status"
        ]
        == PASS
    )
    assert corrected["decision"] == INCONCLUSIVE
    assert corrected["next_action"] == CONTINUE_TO_LOOK_2
    assert (
        check_by_code(corrected, "RISK_FINAL_50_UPPER_BOUND")["status"]
        == INCONCLUSIVE
    )


def test_missing_required_v2_matrix_cell_fails_closed():
    evidence = passing_evidence()
    del evidence["confirmation_matrix"]["powered_candidate_vs_original"]

    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert report["next_action"] == STOP_HARM
    assert (
        check_by_code(report, "MATRIX_POWERED_CANDIDATE_VS_ORIGINAL_PRESENT")[
            "status"
        ]
        == FAIL
    )
    assert report["reason_codes"] == sorted(report["reason_codes"])


def test_metric_requires_exact_cell_sample_size_and_metric_name():
    evidence = passing_evidence()
    evidence["confirmation_matrix"]["powered_candidate_vs_champion"][
        "color_pairs"
    ] -= 1
    report = evaluate_promotion_gate(evidence)
    assert (
        check_by_code(
            report, "MATRIX_POWERED_CANDIDATE_VS_CHAMPION_COLOR_PAIRS"
        )["status"]
        == FAIL
    )

    evidence = passing_evidence()
    cell = evidence["confirmation_matrix"]["powered_candidate_vs_champion"]
    cell["statistics_artifact"]["metrics"]["realized_utility"]["color_pairs"] -= 1
    rehash_statistics_cell(evidence, "powered_candidate_vs_champion")
    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert (
        check_by_code(report, "POWERED_UTILITY_VS_CHAMPION_INFERENCE")["status"]
        == FAIL
    )

    evidence = passing_evidence()
    cell = evidence["confirmation_matrix"]["powered_candidate_vs_champion"]
    cell["statistics_artifact"]["metrics"]["realized_utility"]["metric"] = "win_rate"
    rehash_statistics_cell(evidence, "powered_candidate_vs_champion")
    assert (
        check_by_code(
            evaluate_promotion_gate(evidence),
            "POWERED_UTILITY_VS_CHAMPION_INFERENCE",
        )["status"]
        == FAIL
    )


def test_swapped_metric_or_finalized_artifact_cannot_pass():
    evidence = passing_evidence()
    champion = evidence["confirmation_matrix"]["powered_candidate_vs_champion"]
    original = evidence["confirmation_matrix"]["powered_candidate_vs_original"]
    champion["statistics_artifact"]["metrics"]["realized_utility"] = copy.deepcopy(
        original["statistics_artifact"]["metrics"]["realized_utility"]
    )
    rehash_statistics_cell(evidence, "powered_candidate_vs_champion")
    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert (
        check_by_code(report, "POWERED_UTILITY_VS_CHAMPION_INFERENCE")["status"]
        == FAIL
    )

    evidence = passing_evidence()
    champion = evidence["confirmation_matrix"]["powered_candidate_vs_champion"]
    original = evidence["confirmation_matrix"]["powered_candidate_vs_original"]
    artifact_fields = (
        "statistics_artifact",
        "statistics_artifact_hash",
        "statistics_manifest",
        "statistics_manifest_hash",
    )
    for field in artifact_fields:
        champion[field], original[field] = original[field], champion[field]
    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert (
        check_by_code(
            report,
            "MATRIX_POWERED_CANDIDATE_VS_CHAMPION_STATISTICS_DATA_BINDING",
        )["status"]
        == FAIL
    )


def test_two_position_clusters_never_pass_even_with_full_pair_count():
    evidence = passing_evidence()
    cell_name = "powered_candidate_vs_champion"
    cell = evidence["confirmation_matrix"][cell_name]
    artifact = cell["statistics_artifact"]
    position_ids = cell["statistics_manifest"]["position_ids"][:2]
    for container in (artifact["metrics"], artifact["risk_differences"]):
        for value in container.values():
            value["position_clusters"] = 2
            value["degrees_of_freedom"] = 1
            value["small_cluster_correction"]["variance_multiplier"] = 2.0
            value["position_values"] = value["position_values"][:2]
            value["bootstrap"]["stratum_cluster_counts"][0][
                "position_clusters"
            ] = 2
    cell["statistics_manifest"]["position_ids"] = position_ids
    evidence["provenance"]["suite_manifest"]["cells"][cell_name][
        "position_ids"
    ] = position_ids
    rehash_suite_manifest(evidence)
    for name, value in artifact["risk_differences"].items():
        evidence["risk_differences"][name] = copy.deepcopy(value)
    rehash_statistics_cell(evidence, cell_name)
    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert (
        check_by_code(
            report,
            "SUITE_MANIFEST_POWERED_CANDIDATE_VS_CHAMPION_POSITION_IDS",
        )["status"]
        == FAIL
    )


@pytest.mark.parametrize(
    "cell_name, inference_check",
    [
        (
            "powered_candidate_vs_champion",
            "POWERED_UTILITY_VS_CHAMPION_INFERENCE",
        ),
        (
            "powered_candidate_vs_original",
            "POWERED_UTILITY_VS_ORIGINAL_INFERENCE",
        ),
        (
            "standard_candidate_vs_original",
            "STANDARD_WIN_RATE_VS_ORIGINAL_INFERENCE",
        ),
        ("lead_40", "RISK_LEAD_40_LOSS_INFERENCE"),
        ("lead_80", "RISK_LEAD_80_LOSS_INFERENCE"),
    ],
)
def test_v2_rejects_g_minus_one_clusters_even_when_pair_count_is_exact(
    cell_name, inference_check
):
    evidence = passing_evidence()
    cell = evidence["confirmation_matrix"][cell_name]
    artifact = cell["statistics_artifact"]
    for container in (artifact["metrics"], artifact["risk_differences"]):
        for metric_value in container.values():
            metric_value["position_values"] = metric_value["position_values"][:-1]
            metric_value["position_values"][0]["pair_count"] = 2
            metric_value["position_clusters"] -= 1
            metric_value["degrees_of_freedom"] -= 1
            clusters = metric_value["position_clusters"]
            metric_value["small_cluster_correction"]["variance_multiplier"] = (
                clusters / (clusters - 1.0)
            )
            metric_value["bootstrap"]["stratum_cluster_counts"][0][
                "position_clusters"
            ] = clusters
    cell["statistics_manifest"]["position_ids"] = cell["statistics_manifest"][
        "position_ids"
    ][:-1]
    evidence["provenance"]["suite_manifest"]["cells"][cell_name][
        "position_ids"
    ] = cell["statistics_manifest"]["position_ids"]
    for name, metric_value in artifact["risk_differences"].items():
        evidence["risk_differences"][name] = copy.deepcopy(metric_value)
    rehash_statistics_cell(evidence, cell_name)

    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert check_by_code(report, inference_check)["status"] == FAIL
    assert (
        check_by_code(
            report,
            "SUITE_MANIFEST_" + cell_name.upper() + "_POSITION_IDS",
        )["status"]
        == FAIL
    )


def test_combined_lead_must_use_exact_union_without_double_counting_positions():
    evidence = passing_evidence()
    artifact = evidence["combined_lead_artifact"]
    artifact["position_ids"][-1] = artifact["position_ids"][0]
    artifact["metrics"]["combined_lead_realized_utility"]["position_values"][-1][
        "position_id"
    ] = artifact["position_ids"][0]
    evidence["combined_lead_artifact_hash"] = canonical_hash(artifact)
    evidence["combined_lead_manifest"]["statistics_artifact_hash"] = evidence[
        "combined_lead_artifact_hash"
    ]
    evidence["combined_lead_manifest_hash"] = canonical_hash(
        evidence["combined_lead_manifest"]
    )
    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert (
        check_by_code(report, "COMBINED_LEAD_POSITION_IDS")["status"] == FAIL
    )


def test_frozen_suite_manifest_and_aggregate_bundle_hashes_are_enforced():
    evidence = passing_evidence()
    evidence["provenance"]["discovery_schedule_hash"] = evidence[
        "confirmation_matrix"
    ]["lead_40"]["schedule_hash"]
    evidence["provenance"]["suite_manifest"]["discovery_schedule_hash"] = evidence[
        "provenance"
    ]["discovery_schedule_hash"]
    rehash_suite_manifest(evidence)
    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert (
        check_by_code(report, "DISCOVERY_CONFIRMATION_HASH_INDEPENDENCE")[
            "status"
        ]
        == FAIL
    )

    evidence = passing_evidence()
    evidence["config_hash"] = sha("arbitrary-but-valid-config-bundle")
    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert check_by_code(report, "CONFIG_BUNDLE_RECOMPUTED")["status"] == FAIL


def test_source_revision_and_per_cell_execution_manifests_are_exact():
    evidence = passing_evidence()
    evidence["provenance"]["source_revision_hash"] = "b" * 40
    del evidence["confirmation_matrix"]["lead_80"]["runner_manifest"]
    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert (
        check_by_code(report, "PROVENANCE_SOURCE_REVISION_HASH")["status"] == FAIL
    )
    assert (
        check_by_code(report, "MATRIX_LEAD_80_RUNNER_MANIFEST_HASH")["status"]
        == FAIL
    )


def test_proven_hash_mismatch_and_policy_hash_mismatch_are_failures():
    evidence = passing_evidence()
    evidence["provenance"]["champion_hash"] = sha("different-champion")
    evidence["policy_hash"] = sha("different-policy")

    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert check_by_code(report, "POLICY_HASH")["status"] == FAIL
    assert check_by_code(report, "PROVENANCE_CHAMPION_HASH")["status"] == FAIL


def test_powered_matrix_requires_identical_frozen_objective_for_both_bots():
    evidence = passing_evidence()
    evidence["confirmation_matrix"]["powered_candidate_vs_champion"][
        "candidate_search"
    ]["win_weight"] = 2.0

    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert (
        check_by_code(
            report,
            "MATRIX_POWERED_CANDIDATE_VS_CHAMPION_CANDIDATE_WIN_WEIGHT",
        )["status"]
        == FAIL
    )


def test_no_result_rate_must_be_strictly_below_point_one_percent():
    evidence = passing_evidence()
    evidence["validity"]["true_no_results"] = 1
    evidence["validity"]["total_games"] = 1000
    evidence["validity"]["true_no_result_rate"] = 0.001

    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert check_by_code(report, "TRUE_NO_RESULT_RATE") == {
        "code": "TRUE_NO_RESULT_RATE",
        "status": FAIL,
        "actual": 0.001,
        "expected": 0.001,
        "operator": "<",
    }


def test_proven_safety_violation_takes_precedence_over_other_missing_data():
    evidence = passing_evidence()
    del evidence["confirmation_matrix"]["standard_candidate_vs_original"]
    evidence["validity"]["resignations"] = 1

    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert check_by_code(report, "VALIDITY_RESIGNATIONS")["status"] == FAIL
    assert (
        check_by_code(report, "MATRIX_STANDARD_CANDIDATE_VS_ORIGINAL_PRESENT")[
            "status"
        ]
        == FAIL
    )


def test_attempt_budget_and_fallback_holdout_are_enforced():
    evidence = passing_evidence()
    evidence["attempt"]["attempt_number"] = 2
    evidence["attempt"]["new_holdout_block"] = True
    evidence["attempt"]["new_alpha_allocation"] = True
    assert evaluate_promotion_gate(evidence)["decision"] == PASS

    evidence["attempt"]["new_holdout_block"] = False
    report = evaluate_promotion_gate(evidence)
    assert report["decision"] == FAIL
    assert check_by_code(report, "FALLBACK_NEW_HOLDOUT_BLOCK")["status"] == FAIL


def test_changed_v1_threshold_is_rejected_even_with_recomputed_hash():
    policy = load_policy(V1_POLICY_PATH)
    policy["promotion_thresholds"][
        "powered_win_rate_vs_champion_lower_bound_strictly_above"
    ] = 0.46
    evidence = passing_evidence()
    evidence["policy_hash"] = policy_hash(policy)
    evidence["provenance"]["policy_hash"] = policy_hash(policy)

    report = evaluate_promotion_gate(evidence, policy=policy)
    assert report["decision"] == FAIL
    assert report["reason_codes"] == ["POLICY_INVALID"]
