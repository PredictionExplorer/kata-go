#!/usr/bin/env python3
"""Versioned PASS/FAIL/INCONCLUSIVE gate for checkpoint promotion.

The gate consumes finalized confirmation evidence. Under v2, INCONCLUSIVE is
reserved for complete evidence whose prespecified statistical margins remain
unresolved; malformed, missing, provenance, and safety evidence fails closed.
PASS is possible only when every emitted check passes. Checks and reason codes
are sorted to make reports byte-stable after canonical JSON serialization.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from .paired_stats import (
        DEFAULT_POLICY_PATH,
        canonical_sha256,
        exact_zero_event_upper_bound,
        load_policy,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from paired_stats import (
        DEFAULT_POLICY_PATH,
        canonical_sha256,
        exact_zero_event_upper_bound,
        load_policy,
    )


GATE_REPORT_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1
V1_POLICY_VERSION = "risk-seeking-checkpoint-promotion-v1"
V1_POLICY_HASH = "d3578dfdf99e4aace0461310b7225c1d42051fc4e87770e22d89b8545645d324"
V2_POLICY_VERSION = "risk-seeking-checkpoint-promotion-v2"
V2_POLICY_HASH = "8562bcd7b835ae0cfcfe517a290748258da229b3fcf588dc99b3703c2b8f6023"
V3_POLICY_VERSION = "risk-seeking-checkpoint-promotion-v3"
V3_POLICY_HASH = "0151ddcdee764b1e599eb5313f9dfae944e671ff8098dd471425f8d646ba3318"
PINNED_POLICY_REGISTRY = {
    V1_POLICY_VERSION: {
        "schema_version": 1,
        "policy_hash": V1_POLICY_HASH,
    },
    V2_POLICY_VERSION: {
        "schema_version": 2,
        "policy_hash": V2_POLICY_HASH,
    },
    V3_POLICY_VERSION: {
        "schema_version": 3,
        "policy_hash": V3_POLICY_HASH,
    },
}
POLICY_REGISTRY = PINNED_POLICY_REGISTRY
EXPECTED_POLICY_VERSION = V3_POLICY_VERSION
EXPECTED_POLICY_HASH = V3_POLICY_HASH
PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"
PROMOTE = "PROMOTE"
CONTINUE_TO_LOOK_2 = "CONTINUE_TO_LOOK_2"
STOP_HARM = "STOP_HARM"
STOP_MAXIMUM_INCONCLUSIVE = "STOP_MAXIMUM_INCONCLUSIVE"
_MISSING = object()
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
CELL_COMPARISONS = {
    "powered_candidate_vs_champion": "candidate-vs-champion-powered",
    "powered_candidate_vs_original": "candidate-vs-original-powered",
    "standard_candidate_vs_original": "candidate-vs-original-standard",
    "lead_40": "candidate-vs-champion-powered-lead-40",
    "lead_80": "candidate-vs-champion-powered-lead-80",
}
CELL_SUITES = {
    "powered_candidate_vs_champion": "confirmation",
    "powered_candidate_vs_original": "confirmation",
    "standard_candidate_vs_original": "confirmation",
    "lead_40": "lead-40",
    "lead_80": "lead-80",
}
RISK_CELL_BINDINGS = {
    "final_20": "powered_candidate_vs_champion",
    "final_50": "powered_candidate_vs_champion",
    "high_confidence_loss": "powered_candidate_vs_champion",
    "lead_40_loss": "lead_40",
    "targeted_lead_40_suite_loss": "lead_40",
    "lead_80_loss": "lead_80",
    "targeted_lead_80_suite_loss": "lead_80",
}
CELL_PAIR_COUNT_KEYS = {
    "powered_candidate_vs_champion": "powered_ordinary_color_pairs_per_matchup",
    "powered_candidate_vs_original": "powered_ordinary_color_pairs_per_matchup",
    "standard_candidate_vs_original": "standard_ordinary_color_pairs",
    "lead_40": "lead_40_color_pairs",
    "lead_80": "lead_80_color_pairs",
}


def policy_hash(policy: Mapping[str, Any]) -> str:
    return canonical_sha256(policy)


def _path(value: Any, *parts: str) -> Any:
    current = value
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _canonical_file_sha256(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ) + "\n"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _policy_errors(policy: Any) -> List[str]:
    if not isinstance(policy, dict):
        return ["POLICY_NOT_OBJECT"]
    errors: List[str] = []
    policy_version = policy.get("policy_version")
    registry_entry = PINNED_POLICY_REGISTRY.get(policy_version)
    if registry_entry is None:
        errors.append("POLICY_VERSION_UNSUPPORTED")
    else:
        if policy.get("schema_version") != registry_entry["schema_version"]:
            errors.append("POLICY_SCHEMA_VERSION_MISMATCH")
        if canonical_sha256(policy) != registry_entry["policy_hash"]:
            errors.append("POLICY_CONTENT_HASH_MISMATCH")

    exact_values: Tuple[Tuple[Tuple[str, ...], Any], ...] = (
        (("status",), "frozen"),
        (("objective", "win_weight"), 4.0),
        (("objective", "score_power"), 1.5),
        (("objective", "score_scale"), 20.0),
        (("objective", "missing_numeric_score_term"), 0.0),
        (("confidence", "routine", "nominal_one_sided_confidence"), 0.95),
        (("confidence", "catastrophe", "nominal_one_sided_confidence"), 0.99),
        (("confidence", "sequential_testing", "look_count"), 2),
        (
            ("confidence", "sequential_testing", "data_dependent_thresholds_allowed"),
            False,
        ),
        (
            (
                "evaluation_stages",
                "stage_0_integrity_and_fixed_probes",
                "fixed_analysis_positions",
            ),
            256,
        ),
        (
            (
                "evaluation_stages",
                "stage_0_integrity_and_fixed_probes",
                "fixed_analysis_visits",
            ),
            200,
        ),
        (
            (
                "evaluation_stages",
                "stage_1_cheap_paired_screen",
                "ordinary_color_pairs",
            ),
            32,
        ),
        (("evaluation_stages", "stage_1_cheap_paired_screen", "visits"), 400),
        (("evaluation_stages", "stage_2_finalist_selection", "ordinary_color_pairs"), 128),
        (("evaluation_stages", "stage_2_finalist_selection", "ordinary_visits"), 800),
        (("evaluation_stages", "stage_2_finalist_selection", "lead_40_color_pairs"), 32),
        (("evaluation_stages", "stage_2_finalist_selection", "lead_80_color_pairs"), 32),
        (("evaluation_stages", "stage_2_finalist_selection", "lead_visits"), 800),
        (("evaluation_stages", "stage_2_finalist_selection", "maximum_survivors"), 4),
        (
            ("evaluation_stages", "stage_3_promotion_confirmation", "powered_visits"),
            2000,
        ),
        (
            ("evaluation_stages", "stage_3_promotion_confirmation", "standard_visits"),
            800,
        ),
        (("evaluation_stages", "deep_audit", "promotion_interval"), 5),
        (("promotion_thresholds", "powered_utility_vs_champion_lower_bound_strictly_above"), 0.0),
        (("promotion_thresholds", "powered_utility_vs_original_lower_bound_strictly_above"), 0.0),
        (("promotion_thresholds", "combined_lead_utility_lower_bound_strictly_above"), -0.05),
        (("promotion_thresholds", "powered_win_rate_vs_champion_lower_bound_strictly_above"), 0.47),
        (("promotion_thresholds", "powered_win_rate_vs_original_lower_bound_strictly_above"), 0.47),
        (("promotion_thresholds", "standard_win_rate_vs_original_lower_bound_strictly_above"), 0.45),
        (("promotion_thresholds", "true_no_result_rate_strictly_below"), 0.001),
        (("rollout", "worker_count"), 7),
        (("rollout", "canary_workers"), 1),
        (("rollout", "intermediate_workers"), 3),
        (("rollout", "full_workers"), 7),
        (("rollout", "switch_networks_mid_game"), False),
        (("queue", "screen_interval_new_training_samples"), 500000),
        (("queue", "confirmation_interval_new_training_samples"), 1000000),
        (("queue", "maximum_active_evaluator_entries"), 3),
        (("retention", "evaluated_anchor_count"), 5),
        (("retention", "minimum_free_space_fraction"), 0.1),
        (
            ("bootstrap", "method"),
            "stratified_wild_position_cluster_rademacher_centered_within_declared_strata",
        ),
        (("bootstrap", "weights"), "rademacher"),
        (("bootstrap", "replications"), 9999),
        (("bootstrap", "strata"), ["schedule_id", "suite"]),
        (("bootstrap", "centering"), "within_each_declared_stratum"),
        (("bootstrap", "report_as_sensitivity"), True),
        (
            ("bootstrap", "zero_event_upper_bound"),
            "one_sided_exact_no_event_bound_using_independent_position_clusters",
        ),
        (("attempt_budget", "maximum_confirmation_attempts_per_generation"), 2),
        (("attempt_budget", "maximum_promotions_per_generation"), 1),
    )
    for parts, expected in exact_values:
        actual = _path(policy, *parts)
        if actual is _MISSING:
            errors.append("POLICY_MISSING_" + "_".join(parts).upper())
        elif actual != expected:
            errors.append("POLICY_CHANGED_" + "_".join(parts).upper())

    version_exact_values = (
        (
            (
                "evaluation_stages",
                "stage_0_integrity_and_fixed_probes",
                "exploitability_sentinel_visits",
            ),
            2000,
        ),
        (("evaluation_stages", "deep_audit", "ordinary_color_pairs"), 1024),
        (("evaluation_stages", "deep_audit", "exploitability_positions"), 128),
        (("rollout", "canary_games"), 2000),
        (("rollout", "canary_fresh_audit_color_pairs"), 1024),
    )
    if policy_version == V3_POLICY_VERSION:
        version_exact_values = (
            (("evaluation_stages", "deep_audit", "ordinary_color_pairs"), 2048),
            (("evaluation_stages", "deep_audit", "lead_40_color_pairs"), 1024),
            (("evaluation_stages", "deep_audit", "lead_80_color_pairs"), 2048),
            (("evaluation_stages", "deep_audit", "visits"), [2000, 8000]),
            (
                ("evaluation_stages", "deep_audit", "controls"),
                ["candidate", "champion", "original", "b28"],
            ),
            (("rollout", "canary_games"), 4000),
            (("rollout", "canary_fresh_audit_color_pairs"), 2048),
        )
        if (
            _path(
                policy,
                "evaluation_stages",
                "stage_0_integrity_and_fixed_probes",
                "exploitability_sentinel_visits",
            )
            is not _MISSING
        ):
            errors.append("POLICY_V3_FORBIDS_EXPLOITABILITY_SENTINELS")
        if (
            _path(
                policy,
                "evaluation_stages",
                "deep_audit",
                "exploitability_positions",
            )
            is not _MISSING
        ):
            errors.append("POLICY_V3_FORBIDS_EXPLOITABILITY_AUDIT_BANK")
    for parts, expected in version_exact_values:
        actual = _path(policy, *parts)
        if actual is _MISSING:
            errors.append("POLICY_MISSING_" + "_".join(parts).upper())
        elif actual != expected:
            errors.append("POLICY_CHANGED_" + "_".join(parts).upper())

    looks = _path(policy, "confidence", "sequential_testing", "looks")
    expected_alphas = ((1, 0.01, 0.002), (2, 0.04, 0.008))
    if not isinstance(looks, list) or len(looks) != 2:
        errors.append("POLICY_INVALID_SEQUENTIAL_LOOKS")
    else:
        actual_alphas = tuple(
            (
                look.get("look_number"),
                look.get("routine_one_sided_alpha"),
                look.get("catastrophe_one_sided_alpha"),
            )
            for look in looks
            if isinstance(look, dict)
        )
        if actual_alphas != expected_alphas:
            errors.append("POLICY_CHANGED_SEQUENTIAL_ALPHA_ALLOCATION")

    stage_3_looks = _path(
        policy,
        "evaluation_stages",
        "stage_3_promotion_confirmation",
        "looks",
    )
    expected_counts_by_version = {
        V1_POLICY_VERSION: (
            (1, 256, 128, 64, 64),
            (2, 512, 128, 128, 128),
        ),
        V2_POLICY_VERSION: (
            (1, 512, 128, 512, 1024),
            (2, 1024, 128, 1024, 2048),
        ),
        V3_POLICY_VERSION: (
            (1, 512, 128, 512, 1024),
            (2, 1024, 128, 1024, 2048),
        ),
    }
    expected_counts = expected_counts_by_version.get(policy_version)
    if not isinstance(stage_3_looks, list) or len(stage_3_looks) != 2:
        errors.append("POLICY_INVALID_STAGE_3_LOOKS")
    else:
        actual_counts = tuple(
            (
                look.get("look_number"),
                look.get("powered_ordinary_color_pairs_per_matchup"),
                look.get("standard_ordinary_color_pairs"),
                look.get("lead_40_color_pairs"),
                look.get("lead_80_color_pairs"),
            )
            for look in stage_3_looks
            if isinstance(look, dict)
        )
        if expected_counts is None or actual_counts != expected_counts:
            errors.append("POLICY_CHANGED_STAGE_3_COUNTS")

    expected_risks = {
        "final_20": 0.02,
        "final_50": 0.01,
        "lead_40_loss": 0.005,
        "lead_80_loss": 0.0025,
        "high_confidence_loss": 0.005,
        "targeted_lead_40_suite_loss": 0.03,
        "targeted_lead_80_suite_loss": 0.02,
    }
    actual_risks = _path(
        policy,
        "promotion_thresholds",
        "candidate_minus_reference_risk_upper_bounds",
    )
    if actual_risks != expected_risks:
        errors.append("POLICY_CHANGED_CATASTROPHE_THRESHOLDS")

    required_cells = {
        "powered_candidate_vs_champion",
        "powered_candidate_vs_original",
        "standard_candidate_vs_original",
        "lead_40",
        "lead_80",
    }
    matrix = _path(policy, "required_confirmation_matrix")
    if not isinstance(matrix, dict) or set(matrix) != required_cells:
        errors.append("POLICY_INVALID_REQUIRED_CONFIRMATION_MATRIX")

    if policy_version in {V2_POLICY_VERSION, V3_POLICY_VERSION}:
        expected_supersedes = (
            {
                "policy_version": V1_POLICY_VERSION,
                "policy_hash": V1_POLICY_HASH,
            }
            if policy_version == V2_POLICY_VERSION
            else {
                "policy_version": V2_POLICY_VERSION,
                "policy_hash": V2_POLICY_HASH,
            }
        )
        if policy.get("supersedes") != expected_supersedes:
            errors.append("POLICY_INVALID_SUPERSEDES_BINDING")

        if policy_version == V3_POLICY_VERSION:
            expected_machine_curation = {
                "final_contract": "risk-score-reviewed-position-bank-v2",
                "review_mode": "machine-consensus",
                "consensus_rules_version": 1,
                "stability_margin": 5.0,
                "allowed_labels": ["ordinary", "lead-40", "lead-80"],
                "model_roles": ["immutable_original", "frozen_champion"],
                "search_modes": ["standard", "powered"],
                "visits": [2000, 8000],
                "symmetry_semantics": "katago-shape-preserving-d4-v1",
                "automatic_promotion_requires_transitive_suite_provenance": True,
            }
            if policy.get("machine_curation_contract") != expected_machine_curation:
                errors.append("POLICY_INVALID_MACHINE_CURATION_CONTRACT")

        stage_3 = _path(
            policy,
            "evaluation_stages",
            "stage_3_promotion_confirmation",
        )
        if not isinstance(stage_3, dict):
            errors.append("POLICY_INVALID_STAGE_3")
        else:
            if stage_3.get("look_data_relationship") != "cumulative_prefix":
                errors.append("POLICY_INVALID_CUMULATIVE_PREFIX_SEMANTICS")
            if (
                stage_3.get("independent_position_cluster_semantics")
                != "one_color_pair_per_independent_position_cluster"
                or stage_3.get("color_pairs_per_independent_position_cluster") != 1
            ):
                errors.append("POLICY_INVALID_INDEPENDENT_CLUSTER_SEMANTICS")

        sequential = _path(policy, "confidence", "sequential_testing")
        if isinstance(sequential, dict) and isinstance(looks, list):
            for family, alpha_name in (
                ("routine", "routine_one_sided_alpha"),
                ("catastrophe", "catastrophe_one_sided_alpha"),
            ):
                family_alpha = _path(
                    policy,
                    "confidence",
                    family,
                    "family_one_sided_alpha",
                )
                allocated = [
                    look.get(alpha_name)
                    for look in looks
                    if isinstance(look, dict)
                ]
                if not (
                    _is_finite_number(family_alpha)
                    and len(allocated) == sequential.get("look_count")
                    and all(
                        _is_finite_number(value) and 0.0 < float(value) < 1.0
                        for value in allocated
                    )
                    and math.isclose(
                        sum(float(value) for value in allocated),
                        float(family_alpha),
                        rel_tol=0.0,
                        abs_tol=1.0e-15,
                    )
                ):
                    errors.append(
                        "POLICY_INVALID_" + family.upper() + "_ALPHA_SPENDING"
                    )

        if isinstance(stage_3_looks, list) and len(stage_3_looks) == 2:
            ordered_stage_looks = sorted(
                (
                    look
                    for look in stage_3_looks
                    if isinstance(look, dict)
                    and isinstance(look.get("look_number"), int)
                    and not isinstance(look.get("look_number"), bool)
                ),
                key=lambda look: look["look_number"],
            )
            if len(ordered_stage_looks) != 2 or [
                look["look_number"] for look in ordered_stage_looks
            ] != [1, 2]:
                errors.append("POLICY_INVALID_CUMULATIVE_LOOK_ORDER")
            else:
                previous_counts: Dict[str, int] = {}
                previous_minima: Dict[str, int] = {}
                for look in ordered_stage_looks:
                    minima = look.get("minimum_independent_position_clusters")
                    if not isinstance(minima, dict) or set(minima) != required_cells:
                        errors.append(
                            "POLICY_INVALID_LOOK_"
                            + str(look["look_number"])
                            + "_MINIMUM_INDEPENDENT_POSITION_CLUSTERS"
                        )
                        continue
                    for cell_name in sorted(required_cells):
                        pair_count = look.get(CELL_PAIR_COUNT_KEYS[cell_name])
                        minimum = minima.get(cell_name)
                        if not (
                            _is_nonnegative_integer(pair_count)
                            and pair_count > 0
                            and _is_nonnegative_integer(minimum)
                            and minimum > 0
                            and minimum == pair_count
                        ):
                            errors.append(
                                "POLICY_INVALID_LOOK_"
                                + str(look["look_number"])
                                + "_"
                                + cell_name.upper()
                                + "_PAIR_CLUSTER_COUNTS"
                            )
                            continue
                        if (
                            cell_name in previous_counts
                            and (
                                pair_count < previous_counts[cell_name]
                                or minimum < previous_minima[cell_name]
                            )
                        ):
                            errors.append(
                                "POLICY_NONMONOTONIC_CUMULATIVE_"
                                + cell_name.upper()
                            )
                        previous_counts[cell_name] = pair_count
                        previous_minima[cell_name] = minimum

                final_look = ordered_stage_looks[-1]
                final_minima = final_look.get(
                    "minimum_independent_position_clusters"
                )
                alpha_look = next(
                    (
                        look
                        for look in looks
                        if isinstance(look, dict) and look.get("look_number") == 2
                    ),
                    None,
                )
                catastrophe_alpha = (
                    alpha_look.get("catastrophe_one_sided_alpha")
                    if isinstance(alpha_look, dict)
                    else None
                )
                if (
                    isinstance(final_minima, dict)
                    and _is_finite_number(catastrophe_alpha)
                    and 0.0 < float(catastrophe_alpha) < 1.0
                    and isinstance(actual_risks, dict)
                ):
                    for risk_name, source_cell in sorted(
                        RISK_CELL_BINDINGS.items()
                    ):
                        clusters = final_minima.get(source_cell)
                        threshold = actual_risks.get(risk_name)
                        if not (
                            _is_nonnegative_integer(clusters)
                            and clusters > 0
                            and _is_finite_number(threshold)
                            and exact_zero_event_upper_bound(
                                float(catastrophe_alpha),
                                clusters,
                            )
                            <= float(threshold)
                        ):
                            errors.append(
                                "POLICY_INFEASIBLE_FINAL_ZERO_EVENT_"
                                + risk_name.upper()
                            )
                else:
                    errors.append("POLICY_INVALID_FINAL_ZERO_EVENT_INPUTS")
    return sorted(set(errors))


def validate_policy(policy: Mapping[str, Any]) -> None:
    errors = _policy_errors(policy)
    if errors:
        raise ValueError("invalid frozen promotion policy: " + ", ".join(errors))


def _look_configuration(policy: Mapping[str, Any], look_number: int) -> Optional[Dict[str, Any]]:
    stage_looks = _path(
        policy,
        "evaluation_stages",
        "stage_3_promotion_confirmation",
        "looks",
    )
    alpha_looks = _path(policy, "confidence", "sequential_testing", "looks")
    if not isinstance(stage_looks, list) or not isinstance(alpha_looks, list):
        return None
    stage = next(
        (look for look in stage_looks if look.get("look_number") == look_number),
        None,
    )
    alpha = next(
        (look for look in alpha_looks if look.get("look_number") == look_number),
        None,
    )
    if stage is None or alpha is None:
        return None
    result = dict(stage)
    result.update(alpha)
    return result


def _metric_from(container: Any, name: str) -> Any:
    if not isinstance(container, dict):
        return _MISSING
    if container.get("metric") == name:
        return container
    metrics = container.get("metrics")
    if isinstance(metrics, dict) and name in metrics:
        return metrics[name]
    if name in container:
        return container[name]
    return _MISSING


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not _MISSING:
            return value
    return _MISSING


def evaluate_promotion_gate(
    evidence: Mapping[str, Any],
    *,
    policy: Optional[Dict[str, Any]] = None,
    expected_policy_hash: Optional[str] = None,
    expected_candidate_hash: Optional[str] = None,
    expected_champion_hash: Optional[str] = None,
    expected_original_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate finalized Stage-3 evidence under the frozen policy."""

    active_policy = load_policy() if policy is None else policy
    computed_policy_hash = policy_hash(active_policy) if isinstance(active_policy, dict) else None
    evidence_is_object = isinstance(evidence, dict)
    evidence_type = type(evidence).__name__
    if not evidence_is_object:
        evidence = {}
    candidate_hash = evidence.get("candidate_hash") if isinstance(evidence, dict) else None
    champion_hash = evidence.get("champion_hash") if isinstance(evidence, dict) else None
    original_hash = evidence.get("original_hash") if isinstance(evidence, dict) else None
    evaluation_key = (
        evidence.get("evaluation_key") if isinstance(evidence, dict) else None
    )
    config_hash = evidence.get("config_hash") if isinstance(evidence, dict) else None
    schedule_hash = (
        evidence.get("schedule_hash") if isinstance(evidence, dict) else None
    )
    checks: Dict[str, Dict[str, Any]] = {}
    reason_codes: List[str] = []
    continuation_inconclusive_codes: Set[str] = set()

    def report_value(value: Any) -> Any:
        if value is _MISSING:
            return None
        if isinstance(value, dict):
            return {
                str(key): report_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [report_value(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value

    def add(
        code: str,
        status: str,
        *,
        actual: Any = _MISSING,
        expected: Any = _MISSING,
        operator: Optional[str] = None,
        fail_reason: Optional[str] = None,
        missing_reason: Optional[str] = None,
    ) -> None:
        check: Dict[str, Any] = {"code": code, "status": status}
        if actual is not _MISSING:
            check["actual"] = report_value(actual)
        if expected is not _MISSING:
            check["expected"] = report_value(expected)
        if operator is not None:
            check["operator"] = operator
        checks[code] = check
        if status == FAIL:
            reason_codes.append(fail_reason or f"{code}_FAILED")
        elif status == INCONCLUSIVE:
            reason_codes.append(missing_reason or f"{code}_MISSING")

    def require_equal(
        code: str,
        actual: Any,
        expected: Any,
        *,
        malformed_is_fail: bool = False,
    ) -> None:
        if actual is _MISSING or actual is None:
            add(code, INCONCLUSIVE, expected=expected, operator="==")
        elif malformed_is_fail and not _is_finite_number(actual):
            add(code, FAIL, actual=actual, expected=expected, operator="==")
        else:
            add(
                code,
                PASS if actual == expected else FAIL,
                actual=actual,
                expected=expected,
                operator="==",
            )

    def require_boolean(code: str, actual: Any, expected: bool) -> None:
        if actual is _MISSING or actual is None:
            add(code, INCONCLUSIVE, expected=expected, operator="==")
        elif not isinstance(actual, bool):
            add(code, FAIL, actual=actual, expected=expected, operator="==")
        else:
            add(
                code,
                PASS if actual is expected else FAIL,
                actual=actual,
                expected=expected,
                operator="==",
            )

    def require_sha(code: str, actual: Any, expected: Optional[str] = None) -> None:
        if actual is _MISSING or actual is None:
            add(code, INCONCLUSIVE, expected=expected if expected is not None else "sha256")
        elif not _is_sha256(actual):
            add(code, FAIL, actual=actual, expected=expected or "64 lowercase hex characters")
        elif expected is not None and actual != expected:
            add(code, FAIL, actual=actual, expected=expected, operator="==")
        else:
            add(code, PASS, actual=actual, expected=expected or "sha256")

    def require_object_hash(
        code: str,
        value: Any,
        actual_hash: Any,
        *,
        canonical_file: bool = False,
    ) -> Optional[str]:
        if not isinstance(value, dict):
            add(code, INCONCLUSIVE, expected="object and matching SHA-256")
            return None
        expected_hash = (
            _canonical_file_sha256(value) if canonical_file else canonical_sha256(value)
        )
        require_sha(code, actual_hash, expected_hash)
        return expected_hash

    def numeric_threshold(
        code: str,
        actual: Any,
        operator: str,
        threshold: float,
    ) -> None:
        if actual is _MISSING or actual is None:
            add(code, INCONCLUSIVE, expected=threshold, operator=operator)
            return
        if not _is_finite_number(actual):
            add(code, FAIL, actual=actual, expected=threshold, operator=operator)
            return
        value = float(actual)
        passed = {
            ">": value > threshold,
            "<": value < threshold,
            "<=": value <= threshold,
            ">=": value >= threshold,
            "==": value == threshold,
        }[operator]
        add(
            code,
            PASS if passed else FAIL,
            actual=value,
            expected=threshold,
            operator=operator,
        )

    add(
        "EVIDENCE_OBJECT",
        PASS if evidence_is_object else FAIL,
        actual="object" if evidence_is_object else evidence_type,
        expected="object",
    )
    policy_errors = _policy_errors(active_policy)
    add(
        "POLICY_SCHEMA_AND_FROZEN_VALUES",
        PASS if not policy_errors else FAIL,
        actual=policy_errors,
        expected=[],
        operator="==",
        fail_reason="POLICY_INVALID",
    )
    if policy_errors:
        return {
            "schema_version": GATE_REPORT_SCHEMA_VERSION,
            "schema_name": "risk-seeking-promotion-gate-report",
            "decision": FAIL,
            "next_action": STOP_HARM,
            "continuation_eligible": False,
            "reason_codes": sorted(set(reason_codes)),
            "checks": [checks[code] for code in sorted(checks)],
            "finalized": True,
            "candidate_hash": candidate_hash,
            "champion_hash": champion_hash,
            "tested_champion_hash": champion_hash,
            "original_hash": original_hash,
            "evaluation_key": evaluation_key,
            "config_hash": config_hash,
            "schedule_hash": schedule_hash,
            "policy_hash": computed_policy_hash,
            "policy_version": active_policy.get("policy_version")
            if isinstance(active_policy, dict)
            else None,
            "look_number": evidence.get("look_number")
            if isinstance(evidence, dict)
            else None,
            "ranking_summary": {
                "source_bound": False,
                "realized_powered_utility_lower_bound": None,
                "final_50_risk_upper_bound": None,
            },
        }

    require_equal(
        "EVIDENCE_SCHEMA_VERSION",
        evidence.get("schema_version", _MISSING),
        EVIDENCE_SCHEMA_VERSION,
    )
    require_equal(
        "POLICY_VERSION",
        evidence.get("policy_version", _MISSING),
        active_policy["policy_version"],
    )
    require_sha(
        "POLICY_HASH",
        evidence.get("policy_hash", _MISSING),
        computed_policy_hash,
    )
    if expected_policy_hash is not None:
        require_sha("EXPECTED_POLICY_HASH", computed_policy_hash, expected_policy_hash)

    require_sha("CANDIDATE_HASH", candidate_hash, expected_candidate_hash)
    require_sha("CHAMPION_HASH", champion_hash, expected_champion_hash)
    require_sha("ORIGINAL_HASH", original_hash, expected_original_hash)
    require_boolean(
        "CONFIRMATION_EVIDENCE_FINALIZED",
        evidence.get("confirmation_finalized", _MISSING),
        True,
    )
    if _nonempty(evaluation_key):
        add(
            "EVALUATION_KEY",
            PASS,
            actual=evaluation_key,
            expected="nonempty string",
        )
    elif evaluation_key is None:
        add("EVALUATION_KEY", INCONCLUSIVE, expected="nonempty string")
    else:
        add(
            "EVALUATION_KEY",
            FAIL,
            actual=evaluation_key,
            expected="nonempty string",
        )
    require_sha("CONFIG_BUNDLE_HASH", config_hash)
    require_sha("SCHEDULE_BUNDLE_HASH", schedule_hash)
    if _is_sha256(candidate_hash) and _is_sha256(champion_hash):
        add(
            "CANDIDATE_DIFFERS_FROM_CHAMPION",
            PASS if candidate_hash != champion_hash else FAIL,
            actual=candidate_hash != champion_hash,
            expected=True,
            operator="==",
        )
    else:
        add("CANDIDATE_DIFFERS_FROM_CHAMPION", INCONCLUSIVE, expected=True)
    if _is_sha256(candidate_hash) and _is_sha256(original_hash):
        add(
            "CANDIDATE_DIFFERS_FROM_ORIGINAL",
            PASS if candidate_hash != original_hash else FAIL,
            actual=candidate_hash != original_hash,
            expected=True,
            operator="==",
        )
    else:
        add("CANDIDATE_DIFFERS_FROM_ORIGINAL", INCONCLUSIVE, expected=True)

    look_number = evidence.get("look_number", _MISSING)
    look_config: Optional[Dict[str, Any]] = None
    if look_number is _MISSING or look_number is None:
        add("SEQUENTIAL_LOOK_NUMBER", INCONCLUSIVE, expected=[1, 2])
    elif not isinstance(look_number, int) or isinstance(look_number, bool):
        add("SEQUENTIAL_LOOK_NUMBER", FAIL, actual=look_number, expected=[1, 2])
    else:
        look_config = _look_configuration(active_policy, look_number)
        add(
            "SEQUENTIAL_LOOK_NUMBER",
            PASS if look_config is not None else FAIL,
            actual=look_number,
            expected=[1, 2],
        )
    require_boolean(
        "NO_DATA_DEPENDENT_THRESHOLD_OVERRIDE",
        evidence.get("thresholds_overridden", _MISSING),
        False,
    )
    require_boolean(
        "NO_ALPHA_ALLOCATION_OVERRIDE",
        evidence.get("alpha_allocation_overridden", _MISSING),
        False,
    )
    require_equal(
        "EVALUATION_STAGE",
        evidence.get("evaluation_stage", _MISSING),
        3,
    )

    attempt = evidence.get("attempt", _MISSING)
    attempt_number = _path(attempt, "attempt_number")
    generation_id = _path(attempt, "generation_id")
    promotions_for_generation = _path(attempt, "promotions_for_generation")
    maximum_attempts = active_policy["attempt_budget"][
        "maximum_confirmation_attempts_per_generation"
    ]
    if attempt_number is _MISSING:
        add("ATTEMPT_BUDGET", INCONCLUSIVE, expected=f"1..{maximum_attempts}")
    elif not isinstance(attempt_number, int) or isinstance(attempt_number, bool):
        add("ATTEMPT_BUDGET", FAIL, actual=attempt_number, expected=f"1..{maximum_attempts}")
    else:
        add(
            "ATTEMPT_BUDGET",
            PASS if 1 <= attempt_number <= maximum_attempts else FAIL,
            actual=attempt_number,
            expected=f"1..{maximum_attempts}",
        )
    if _nonempty(generation_id):
        add("GENERATION_ID", PASS, actual=generation_id, expected="nonempty string")
    elif generation_id is _MISSING or generation_id is None:
        add("GENERATION_ID", INCONCLUSIVE, expected="nonempty string")
    else:
        add("GENERATION_ID", FAIL, actual=generation_id, expected="nonempty string")
    numeric_threshold(
        "PROMOTIONS_PER_GENERATION_BUDGET",
        promotions_for_generation,
        "<",
        float(active_policy["attempt_budget"]["maximum_promotions_per_generation"]),
    )
    if isinstance(attempt_number, int) and attempt_number > 1:
        require_boolean(
            "FALLBACK_NEW_HOLDOUT_BLOCK",
            _path(attempt, "new_holdout_block"),
            True,
        )
        require_boolean(
            "FALLBACK_NEW_ALPHA_ALLOCATION",
            _path(attempt, "new_alpha_allocation"),
            True,
        )

    matrix = evidence.get("confirmation_matrix", _MISSING)
    v2_policy = active_policy.get("schema_version") in {2, 3}
    machine_review_v3 = active_policy.get("policy_version") == V3_POLICY_VERSION
    stage_3 = active_policy["evaluation_stages"]["stage_3_promotion_confirmation"]
    cell_specs = active_policy["required_confirmation_matrix"]
    provenance = evidence.get("provenance", _MISSING)
    suite_manifest = _path(provenance, "suite_manifest")
    suite_manifest_hash = _path(provenance, "suite_manifest_hash")
    require_object_hash(
        "PROVENANCE_SUITE_MANIFEST_HASH",
        suite_manifest,
        suite_manifest_hash,
        canonical_file=True,
    )
    if isinstance(suite_manifest, dict):
        suite_payload = dict(suite_manifest)
        suite_payload_hash = suite_payload.pop("manifestPayloadSha256", None)
        require_equal(
            "PROVENANCE_SUITE_MANIFEST_PAYLOAD_HASH",
            suite_payload_hash,
            canonical_sha256(suite_payload),
        )
    else:
        add("PROVENANCE_SUITE_MANIFEST_PAYLOAD_HASH", INCONCLUSIVE)
    require_equal(
        "PROVENANCE_SUITE_MANIFEST_POLICY",
        _path(suite_manifest, "policy_hash"),
        computed_policy_hash,
    )
    require_equal(
        "PROVENANCE_SUITE_MANIFEST_SOURCE_REVISION",
        _path(suite_manifest, "source_revision"),
        active_policy["frozen_plan"]["source_revision"],
    )
    raw_suite_cells = _path(suite_manifest, "cells")
    authoritative_suite_manifest = isinstance(raw_suite_cells, list)
    if machine_review_v3 and not authoritative_suite_manifest:
        add(
            "PROVENANCE_SUITE_MANIFEST_V3_REQUIRED",
            FAIL,
            actual=type(raw_suite_cells).__name__,
            expected="authoritative v3 cell list",
        )
    if authoritative_suite_manifest:
        require_equal(
            "PROVENANCE_SUITE_MANIFEST_SCHEMA",
            _path(suite_manifest, "schemaVersion"),
            3 if machine_review_v3 else 2,
        )
        require_equal(
            "PROVENANCE_SUITE_MANIFEST_CONTRACT",
            _path(suite_manifest, "manifestContract"),
            (
                "risk-score-authoritative-evaluation-manifest-v3"
                if machine_review_v3
                else "risk-score-authoritative-evaluation-manifest-v2"
            ),
        )
        if machine_review_v3:
            require_equal(
                "PROVENANCE_SUITE_MACHINE_REVIEW_ONLY",
                _path(suite_manifest, "machineReviewOnly"),
                True,
            )
            require_equal(
                "PROVENANCE_SUITE_ACCEPTED_LABELS",
                _path(suite_manifest, "acceptedLabels"),
                ["lead-40", "lead-80", "ordinary"],
            )
            curation_sources = _path(suite_manifest, "curationSources")
            if not isinstance(curation_sources, list) or not curation_sources:
                add("PROVENANCE_SUITE_CURATION_SOURCES", FAIL)
            elif any(
                not isinstance(source, dict)
                or source.get("contract") != "risk-score-reviewed-position-bank-v2"
                or source.get("review_mode") != "machine-consensus"
                or source.get("consensus_rules_version") != 1
                or source.get("policy_hash") != computed_policy_hash
                or source.get("allowed_labels") != ["lead-40", "lead-80", "ordinary"]
                or not _is_sha256(source.get("output_sha256"))
                or not _is_sha256(source.get("manifest_sha256"))
                or not _is_nonnegative_integer(source.get("rejected_count"))
                or not _is_sha256(source.get("rejected_sha256"))
                for source in curation_sources
            ):
                add("PROVENANCE_SUITE_CURATION_SOURCES", FAIL)
            else:
                add("PROVENANCE_SUITE_CURATION_SOURCES", PASS)

    def selected_suite_manifest_cell(cell_name: str) -> Any:
        if isinstance(raw_suite_cells, dict):
            return _path(raw_suite_cells, cell_name)
        if isinstance(raw_suite_cells, list):
            matches = [
                entry
                for entry in raw_suite_cells
                if isinstance(entry, dict)
                and entry.get("cell_name") == cell_name
                and entry.get("stage") == "stage-3"
                and (
                    entry.get("look_number") == look_number
                    or entry.get("look")
                    == (
                        f"look-{look_number}"
                        if isinstance(look_number, int)
                        else None
                    )
                )
            ]
            return matches[0] if len(matches) == 1 else _MISSING
        return _MISSING

    cells: Dict[str, Any] = {}
    cell_artifacts: Dict[str, Any] = {}
    cell_statistics_manifests: Dict[str, Any] = {}
    expected_pair_counts: Dict[str, Optional[int]] = {}
    expected_cluster_counts: Dict[str, Optional[int]] = {}
    for cell_name in sorted(cell_specs):
        cell = _path(matrix, cell_name)
        cells[cell_name] = cell
        code_prefix = "MATRIX_" + cell_name.upper()
        if cell is _MISSING:
            add(code_prefix + "_PRESENT", INCONCLUSIVE, expected=True)
            continue
        if not isinstance(cell, dict):
            add(code_prefix + "_PRESENT", FAIL, actual=type(cell).__name__, expected="object")
            continue
        add(code_prefix + "_PRESENT", PASS, actual=True, expected=True)
        spec = cell_specs[cell_name]
        require_equal(
            code_prefix + "_COMPARISON",
            cell.get("comparison", _MISSING),
            CELL_COMPARISONS[cell_name],
        )
        require_equal(
            code_prefix + "_SUITE",
            cell.get("suite", _MISSING),
            CELL_SUITES[cell_name],
        )
        require_equal(
            code_prefix + "_STAGE",
            cell.get("stage", _MISSING),
            "stage-3",
        )
        require_equal(
            code_prefix + "_LOOK",
            cell.get("look", _MISSING),
            f"look-{look_number}" if isinstance(look_number, int) else None,
        )
        topology = cell.get("topology", _MISSING)
        if _nonempty(topology):
            add(code_prefix + "_TOPOLOGY", PASS, actual=topology, expected="nonempty")
        elif topology is _MISSING or topology is None:
            add(code_prefix + "_TOPOLOGY", INCONCLUSIVE, expected="nonempty")
        else:
            add(code_prefix + "_TOPOLOGY", FAIL, actual=topology, expected="nonempty")
        require_equal(
            code_prefix + "_SEARCH_MODE",
            cell.get("search_mode", _MISSING),
            spec["search_mode"],
        )
        require_sha(code_prefix + "_CONFIG_HASH", cell.get("config_hash", _MISSING))
        for role in ("candidate", "reference"):
            search_settings = _path(cell, role + "_search")
            if spec["search_mode"] == "powered":
                require_boolean(
                    code_prefix + "_" + role.upper() + "_POWERED_UTILITY",
                    _path(search_settings, "use_score_maximizing_utility"),
                    True,
                )
                for objective_name in ("win_weight", "score_power", "score_scale"):
                    require_equal(
                        code_prefix
                        + "_"
                        + role.upper()
                        + "_"
                        + objective_name.upper(),
                        _path(search_settings, objective_name),
                        active_policy["objective"][objective_name],
                    )
            else:
                require_boolean(
                    code_prefix + "_" + role.upper() + "_POWERED_UTILITY",
                    _path(search_settings, "use_score_maximizing_utility"),
                    False,
                )
        require_sha(code_prefix + "_CANDIDATE_HASH", cell.get("candidate_hash", _MISSING), candidate_hash)
        expected_reference = champion_hash if spec["reference"] == "champion" else original_hash
        require_sha(
            code_prefix + "_REFERENCE_HASH",
            cell.get("reference_hash", _MISSING),
            expected_reference,
        )
        expected_visits = (
            stage_3["powered_visits"]
            if spec["search_mode"] == "powered"
            else stage_3["standard_visits"]
        )
        require_equal(
            code_prefix + "_VISITS",
            cell.get("visits", _MISSING),
            expected_visits,
        )
        if look_config is None:
            expected_pair_counts[cell_name] = None
            expected_cluster_counts[cell_name] = None
            add(code_prefix + "_COLOR_PAIRS", INCONCLUSIVE)
        else:
            pair_key = CELL_PAIR_COUNT_KEYS[cell_name]
            expected_pair_counts[cell_name] = int(look_config[pair_key])
            minimum_clusters = look_config.get(
                "minimum_independent_position_clusters"
            )
            expected_cluster_counts[cell_name] = (
                int(minimum_clusters[cell_name])
                if v2_policy
                and isinstance(minimum_clusters, dict)
                and _is_nonnegative_integer(minimum_clusters.get(cell_name))
                else None
            )
            require_equal(
                code_prefix + "_COLOR_PAIRS",
                cell.get("color_pairs", _MISSING),
                look_config[pair_key],
            )
        schedule_id = cell.get("schedule_id", _MISSING)
        if _nonempty(schedule_id):
            add(code_prefix + "_SCHEDULE_ID", PASS, actual=schedule_id, expected="nonempty string")
        elif schedule_id is _MISSING or schedule_id is None:
            add(code_prefix + "_SCHEDULE_ID", INCONCLUSIVE, expected="nonempty string")
        else:
            add(code_prefix + "_SCHEDULE_ID", FAIL, actual=schedule_id, expected="nonempty string")
        require_sha(code_prefix + "_SCHEDULE_HASH", cell.get("schedule_hash", _MISSING))
        require_boolean(
            code_prefix + "_VALID",
            _path(cell, "validation", "promotion_valid"),
            True,
        )
        require_sha(
            code_prefix + "_KATAGO_BINARY_HASH",
            cell.get("katago_binary_hash", _MISSING),
        )
        require_sha(code_prefix + "_SUITE_HASH", cell.get("suite_hash", _MISSING))
        require_sha(
            code_prefix + "_EXECUTION_HASH",
            cell.get("execution_hash", _MISSING),
        )

        runner_manifest = cell.get("runner_manifest", _MISSING)
        runner_manifest_hash = cell.get("runner_manifest_hash", _MISSING)
        require_object_hash(
            code_prefix + "_RUNNER_MANIFEST_HASH",
            runner_manifest,
            runner_manifest_hash,
            canonical_file=True,
        )
        if isinstance(runner_manifest, dict):
            require_equal(
                code_prefix + "_RUNNER_SCHEMA_VERSION",
                runner_manifest.get("schemaVersion", _MISSING),
                1,
            )
            require_equal(
                code_prefix + "_RUNNER_CONTRACT",
                runner_manifest.get("runnerContract", _MISSING),
                (
                    "risk-score-pair-safe-evaluation-runner-v3"
                    if v2_policy
                    else "risk-score-pair-safe-evaluation-runner-v2"
                ),
            )
            runner_payload = dict(runner_manifest)
            runner_payload_hash = runner_payload.pop("manifestPayloadSha256", None)
            require_equal(
                code_prefix + "_RUNNER_PAYLOAD_HASH",
                runner_payload_hash,
                canonical_sha256(runner_payload),
            )
            runner_spec = _path(runner_manifest, "evaluationSpec")
            expected_runner_spec = {
                "candidate_model_sha": candidate_hash,
                "reference_model_sha": expected_reference,
                "original_model_sha": original_hash,
                "config_sha": cell.get("config_hash"),
                "schedule_sha": cell.get("schedule_hash"),
                "policy_sha": computed_policy_hash,
                "comparison": CELL_COMPARISONS[cell_name],
                "suite": CELL_SUITES[cell_name],
                "stage": cell.get("stage"),
                "look": cell.get("look"),
                "topology": cell.get("topology"),
                "suite_manifest_sha": suite_manifest_hash,
                "suite_bank_sha": cell.get("suite_hash"),
                "schedule_id": cell.get("schedule_id"),
            }
            if v2_policy:
                expected_runner_spec["max_visits"] = cell.get("visits")
            require_equal(
                code_prefix + "_RUNNER_EVALUATION_SPEC",
                runner_spec,
                expected_runner_spec,
            )
            expected_runner_cell = {
                "comparison": CELL_COMPARISONS[cell_name],
                "suite": CELL_SUITES[cell_name],
                "stage": cell.get("stage"),
                "look": cell.get("look"),
                "gameCount": (
                    2 * expected_pair_counts[cell_name]
                    if expected_pair_counts.get(cell_name) is not None
                    else None
                ),
                "colorPairCount": expected_pair_counts.get(cell_name),
            }
            if v2_policy:
                expected_runner_cell["maxVisits"] = cell.get("visits")
            require_equal(
                code_prefix + "_RUNNER_CELL",
                runner_manifest.get("cell", _MISSING),
                expected_runner_cell,
            )
            require_equal(
                code_prefix + "_RUNNER_SCHEDULE_HASH",
                _path(runner_manifest, "schedule", "sha256"),
                cell.get("schedule_hash"),
            )
            require_equal(
                code_prefix + "_RUNNER_SCHEDULE_ID",
                _path(runner_manifest, "schedule", "scheduleId"),
                cell.get("schedule_id"),
            )
            require_equal(
                code_prefix + "_RUNNER_SUITE_MANIFEST",
                _path(runner_manifest, "schedule", "suiteManifestSha256"),
                suite_manifest_hash,
            )
            require_equal(
                code_prefix + "_RUNNER_SUITE_BANK",
                _path(runner_manifest, "schedule", "suiteBankSha256"),
                cell.get("suite_hash"),
            )
            require_equal(
                code_prefix + "_RUNNER_PAIR_COUNT",
                _path(runner_manifest, "schedule", "pairCount"),
                expected_pair_counts.get(cell_name),
            )
            if authoritative_suite_manifest:
                manifest_cell = selected_suite_manifest_cell(cell_name)
                require_equal(
                    code_prefix + "_RUNNER_MANIFEST_CELL",
                    _path(runner_manifest, "schedule", "manifestCell"),
                    manifest_cell,
                )
                require_sha(
                    code_prefix + "_RUNNER_MANIFEST_CELL_HASH",
                    _path(
                        runner_manifest,
                        "schedule",
                        "manifestCellSha256",
                    ),
                    (
                        canonical_sha256(manifest_cell)
                        if isinstance(manifest_cell, dict)
                        else None
                    ),
                )
            expected_rows = (
                2 * expected_pair_counts[cell_name]
                if expected_pair_counts.get(cell_name) is not None
                else None
            )
            require_equal(
                code_prefix + "_RUNNER_SCHEDULE_ROWS",
                _path(runner_manifest, "schedule", "rowCount"),
                expected_rows,
            )
            require_equal(
                code_prefix + "_RUNNER_RESULT_ROWS",
                _path(runner_manifest, "results", "rowCount"),
                expected_rows,
            )
            require_sha(
                code_prefix + "_RUNNER_RESULTS_HASH",
                _path(runner_manifest, "results", "sha256"),
            )
            require_sha(
                code_prefix + "_RUNNER_MOVES_HASH",
                _path(runner_manifest, "moves", "sha256"),
            )

        execution_manifest = cell.get("execution_manifest", _MISSING)
        require_object_hash(
            code_prefix + "_EXECUTION_MANIFEST_HASH",
            execution_manifest,
            cell.get("execution_hash", _MISSING),
        )
        if isinstance(execution_manifest, dict):
            require_equal(
                code_prefix + "_EXECUTION_MANIFEST_BINDING",
                execution_manifest,
                _path(runner_manifest, "execution"),
            )
            require_equal(
                code_prefix + "_EXECUTION_BINARY_BINDING",
                _path(execution_manifest, "katagoBinarySha256"),
                cell.get("katago_binary_hash"),
            )
            require_boolean(
                code_prefix + "_EXECUTION_FULL_MOVE_TRACES",
                _path(execution_manifest, "moveTraces"),
                True,
            )
            if isinstance(runner_manifest, dict):
                require_equal(
                    code_prefix + "_RUNNER_EXECUTION_KEY",
                    runner_manifest.get("evaluationKey", _MISSING),
                    "eval-"
                    + canonical_sha256(
                        {
                            "runnerContract": runner_manifest.get(
                                "runnerContract"
                            ),
                            "evaluationSpec": runner_manifest.get(
                                "evaluationSpec"
                            ),
                            "execution": execution_manifest,
                        }
                    ),
                )

        statistics_artifact = cell.get("statistics_artifact", _MISSING)
        statistics_artifact_hash = cell.get("statistics_artifact_hash", _MISSING)
        require_object_hash(
            code_prefix + "_STATISTICS_ARTIFACT_HASH",
            statistics_artifact,
            statistics_artifact_hash,
        )
        statistics_manifest = cell.get("statistics_manifest", _MISSING)
        cell_statistics_manifests[cell_name] = statistics_manifest
        require_object_hash(
            code_prefix + "_STATISTICS_MANIFEST_HASH",
            statistics_manifest,
            cell.get("statistics_manifest_hash", _MISSING),
        )
        if isinstance(statistics_manifest, dict):
            expected_statistics_binding = {
                "schema_version": 1,
                "finalized": True,
                "cell_name": cell_name,
                "candidate_hash": candidate_hash,
                "reference_hash": expected_reference,
                "comparison": CELL_COMPARISONS[cell_name],
                "suite": CELL_SUITES[cell_name],
                "suite_hash": cell.get("suite_hash"),
                "schedule_id": cell.get("schedule_id"),
                "schedule_hash": cell.get("schedule_hash"),
                "config_hash": cell.get("config_hash"),
                "runner_manifest_hash": runner_manifest_hash,
                "execution_hash": cell.get("execution_hash"),
                "katago_binary_hash": cell.get("katago_binary_hash"),
                "statistics_artifact_hash": statistics_artifact_hash,
                "color_pairs": expected_pair_counts.get(cell_name),
                "position_ids": _path(statistics_manifest, "position_ids"),
                "metric_names": _path(statistics_manifest, "metric_names"),
            }
            require_equal(
                code_prefix + "_STATISTICS_MANIFEST_BINDING",
                statistics_manifest,
                expected_statistics_binding,
            )
        if isinstance(statistics_artifact, dict):
            require_boolean(
                code_prefix + "_STATISTICS_FINALIZED",
                statistics_artifact.get("finalized", _MISSING),
                True,
            )
            require_equal(
                code_prefix + "_STATISTICS_POLICY",
                statistics_artifact.get("policy_hash", _MISSING),
                computed_policy_hash,
            )
            require_equal(
                code_prefix + "_STATISTICS_LOOK",
                _path(statistics_artifact, "look", "number"),
                look_number,
            )
            require_equal(
                code_prefix + "_STATISTICS_PAIR_COUNT",
                _path(statistics_artifact, "counts", "color_pairs"),
                expected_pair_counts.get(cell_name),
            )
            require_equal(
                code_prefix + "_STATISTICS_DATA_BINDING",
                statistics_artifact.get("data_binding", _MISSING),
                {
                    "candidate_hash": candidate_hash,
                    "reference_hash": expected_reference,
                    "comparison": CELL_COMPARISONS[cell_name],
                    "suite": CELL_SUITES[cell_name],
                    "suite_hash": cell.get("suite_hash"),
                    "schedule_id": cell.get("schedule_id"),
                    "schedule_hash": cell.get("schedule_hash"),
                    "config_hash": cell.get("config_hash"),
                    "runner_manifest_hash": runner_manifest_hash,
                    "execution_hash": cell.get("execution_hash"),
                    "katago_binary_hash": cell.get("katago_binary_hash"),
                },
            )
        cell_artifacts[cell_name] = statistics_artifact

    for cell_name in sorted(cell_specs):
        code_prefix = "SUITE_MANIFEST_" + cell_name.upper()
        suite_entry = selected_suite_manifest_cell(cell_name)
        cell = cells.get(cell_name, _MISSING)
        require_equal(
            code_prefix + "_SUITE",
            _path(suite_entry, "suite"),
            CELL_SUITES[cell_name],
        )
        if not isinstance(cell, dict):
            for suffix in (
                "SUITE_HASH",
                "SCHEDULE_ID",
                "SCHEDULE_HASH",
                "COLOR_PAIRS",
                "POSITION_IDS",
                "STATISTICS_POSITIONS",
            ):
                add(code_prefix + "_" + suffix, INCONCLUSIVE)
            continue
        require_equal(
            code_prefix + "_SUITE_HASH",
            _first_present(
                _path(suite_entry, "suite_hash"),
                _path(suite_entry, "bank_hash"),
            ),
            _path(cell, "suite_hash"),
        )
        require_equal(
            code_prefix + "_SCHEDULE_ID",
            _path(suite_entry, "schedule_id"),
            _path(cell, "schedule_id"),
        )
        require_equal(
            code_prefix + "_SCHEDULE_HASH",
            _path(suite_entry, "schedule_hash"),
            _path(cell, "schedule_hash"),
        )
        require_equal(
            code_prefix + "_COLOR_PAIRS",
            _path(suite_entry, "color_pairs"),
            expected_pair_counts.get(cell_name),
        )
        if authoritative_suite_manifest:
            require_equal(
                code_prefix + "_MINIMUM_CLUSTERS",
                _path(
                    suite_entry,
                    "minimum_independent_position_clusters",
                ),
                expected_cluster_counts.get(cell_name),
            )
            independent_cluster_ids = _path(
                suite_entry,
                "independent_cluster_ids",
            )
            valid_independent_clusters = (
                isinstance(independent_cluster_ids, list)
                and all(_is_sha256(value) for value in independent_cluster_ids)
                and len(independent_cluster_ids)
                == expected_cluster_counts.get(cell_name)
                and len(independent_cluster_ids)
                == len(set(independent_cluster_ids))
            )
            add(
                code_prefix + "_INDEPENDENT_CLUSTER_IDS",
                PASS if valid_independent_clusters else FAIL,
                actual=(
                    len(independent_cluster_ids)
                    if isinstance(independent_cluster_ids, list)
                    else independent_cluster_ids
                ),
                expected=expected_cluster_counts.get(cell_name),
                operator="==",
            )
            require_sha(
                code_prefix + "_INDEPENDENT_CLUSTER_IDS_HASH",
                _path(
                    suite_entry,
                    "independent_cluster_ids_hash",
                ),
                (
                    canonical_sha256(independent_cluster_ids)
                    if isinstance(independent_cluster_ids, list)
                    else None
                ),
            )
        position_ids = _path(suite_entry, "position_ids")
        minimum_clusters = expected_cluster_counts.get(cell_name)
        valid_position_count = (
            len(position_ids) == minimum_clusters
            if v2_policy
            and isinstance(position_ids, list)
            and minimum_clusters is not None
            else isinstance(position_ids, list) and len(position_ids) >= 3
        )
        if not (
            isinstance(position_ids, list)
            and valid_position_count
            and all(_nonempty(value) for value in position_ids)
            and len(position_ids) == len(set(position_ids))
        ):
            expected_positions: Any = (
                minimum_clusters
                if v2_policy and minimum_clusters is not None
                else "at least three"
            )
            add(
                code_prefix + "_POSITION_IDS",
                FAIL if position_ids is not _MISSING else INCONCLUSIVE,
                actual=position_ids,
                expected=(
                    f"exactly {expected_positions} unique nonempty frozen position IDs"
                    if isinstance(expected_positions, int)
                    else "at least three unique nonempty frozen position IDs"
                ),
            )
        else:
            add(
                code_prefix + "_POSITION_IDS",
                PASS,
                actual=len(position_ids),
                expected=(
                    minimum_clusters
                    if v2_policy
                    else "at least three unique frozen position IDs"
                ),
            )
        require_equal(
            code_prefix + "_STATISTICS_POSITIONS",
            _path(cell_statistics_manifests.get(cell_name), "position_ids"),
            (
                sorted(position_ids)
                if isinstance(position_ids, list)
                and all(_nonempty(value) for value in position_ids)
                else position_ids
            ),
        )

    complete_cells = [
        cells[name] for name in sorted(cell_specs) if isinstance(cells.get(name), dict)
    ]
    if (
        len(complete_cells) == len(cell_specs)
        and all(_is_sha256(cell.get("config_hash")) for cell in complete_cells)
        and all(_is_sha256(cell.get("schedule_hash")) for cell in complete_cells)
    ):
        recomputed_config_hash = canonical_sha256(
            sorted({cell.get("config_hash") for cell in complete_cells})
        )
        recomputed_schedule_hash = canonical_sha256(
            sorted({cell.get("schedule_hash") for cell in complete_cells})
        )
        require_sha("CONFIG_BUNDLE_RECOMPUTED", config_hash, recomputed_config_hash)
        require_sha("SCHEDULE_BUNDLE_RECOMPUTED", schedule_hash, recomputed_schedule_hash)
        runner_specs = [
            _path(cells[name], "runner_manifest", "evaluationSpec")
            for name in cell_specs
        ]
        if all(isinstance(spec, dict) for spec in runner_specs):
            require_equal(
                "EVALUATION_KEY_RECOMPUTED",
                evaluation_key,
                "matrix-" + canonical_sha256(runner_specs),
            )
        else:
            add("EVALUATION_KEY_RECOMPUTED", INCONCLUSIVE)
    else:
        add("CONFIG_BUNDLE_RECOMPUTED", INCONCLUSIVE)
        add("SCHEDULE_BUNDLE_RECOMPUTED", INCONCLUSIVE)
        add("EVALUATION_KEY_RECOMPUTED", INCONCLUSIVE)

    thresholds = active_policy["promotion_thresholds"]
    powered_champion = cell_artifacts.get(
        "powered_candidate_vs_champion", _MISSING
    )
    powered_original = cell_artifacts.get("powered_candidate_vs_original", _MISSING)
    standard_original = cell_artifacts.get(
        "standard_candidate_vs_original", _MISSING
    )
    utility_champion = _metric_from(
        _path(powered_champion, "metrics"), "realized_utility"
    )
    utility_original = _metric_from(
        _path(powered_original, "metrics"), "realized_utility"
    )
    win_champion = _metric_from(_path(powered_champion, "metrics"), "win_rate")
    win_original = _metric_from(_path(powered_original, "metrics"), "win_rate")
    standard_win = _metric_from(_path(standard_original, "metrics"), "win_rate")

    combined_artifact = evidence.get("combined_lead_artifact", _MISSING)
    combined_artifact_hash = evidence.get(
        "combined_lead_artifact_hash", _MISSING
    )
    require_object_hash(
        "COMBINED_LEAD_ARTIFACT_HASH",
        combined_artifact,
        combined_artifact_hash,
    )
    combined_manifest = evidence.get("combined_lead_manifest", _MISSING)
    require_object_hash(
        "COMBINED_LEAD_MANIFEST_HASH",
        combined_manifest,
        evidence.get("combined_lead_manifest_hash", _MISSING),
    )
    lead_source_hashes = {
        name: _path(cells.get(name, _MISSING), "statistics_artifact_hash")
        for name in ("lead_40", "lead_80")
    }
    lead_position_ids = {
        name: _path(cell_statistics_manifests.get(name), "position_ids")
        for name in ("lead_40", "lead_80")
    }
    if all(
        isinstance(value, list)
        and all(_nonempty(position_id) for position_id in value)
        for value in lead_position_ids.values()
    ):
        combined_position_ids = sorted(
            set(lead_position_ids["lead_40"]).union(lead_position_ids["lead_80"])
        )
    else:
        combined_position_ids = _MISSING
    expected_combined_pairs = (
        expected_pair_counts.get("lead_40", 0)
        + expected_pair_counts.get("lead_80", 0)
        if expected_pair_counts.get("lead_40") is not None
        and expected_pair_counts.get("lead_80") is not None
        else None
    )
    expected_combined_clusters = (
        expected_cluster_counts.get("lead_40", 0)
        + expected_cluster_counts.get("lead_80", 0)
        if expected_cluster_counts.get("lead_40") is not None
        and expected_cluster_counts.get("lead_80") is not None
        else None
    )
    expected_combined_manifest = {
        "schema_version": 1,
        "finalized": True,
        "source_cells": ["lead_40", "lead_80"],
        "source_statistics_artifact_hashes": lead_source_hashes,
        "color_pairs": expected_combined_pairs,
        "position_ids": combined_position_ids,
        "metric_names": ["combined_lead_realized_utility"],
        "statistics_artifact_hash": combined_artifact_hash,
    }
    require_equal(
        "COMBINED_LEAD_MANIFEST_BINDING",
        combined_manifest,
        expected_combined_manifest,
    )
    if isinstance(combined_artifact, dict):
        require_boolean(
            "COMBINED_LEAD_FINALIZED",
            combined_artifact.get("finalized", _MISSING),
            True,
        )
        require_equal(
            "COMBINED_LEAD_POLICY",
            combined_artifact.get("policy_hash", _MISSING),
            computed_policy_hash,
        )
        require_equal(
            "COMBINED_LEAD_LOOK",
            _path(combined_artifact, "look", "number"),
            look_number,
        )
        require_equal(
            "COMBINED_LEAD_SOURCE_BINDING",
            combined_artifact.get("source_statistics_artifact_hashes", _MISSING),
            lead_source_hashes,
        )
        require_equal(
            "COMBINED_LEAD_PAIR_COUNT",
            _path(combined_artifact, "counts", "color_pairs"),
            expected_combined_pairs,
        )
        require_equal(
            "COMBINED_LEAD_POSITION_IDS",
            combined_artifact.get("position_ids", _MISSING),
            combined_position_ids,
        )
    combined_lead = _metric_from(
        _path(combined_artifact, "metrics"),
        "combined_lead_realized_utility",
    )

    routine_alpha = look_config.get("routine_one_sided_alpha") if look_config else None
    catastrophe_alpha = (
        look_config.get("catastrophe_one_sided_alpha") if look_config else None
    )
    bootstrap_replications = active_policy["bootstrap"]["replications"]

    def inference_check(
        code: str,
        metric: Any,
        alpha: Optional[float],
        nominal_confidence: float,
        *,
        expected_metric_name: str,
        expected_color_pairs: Optional[int],
        expected_minimum_clusters: Optional[int],
        expected_position_ids: Any,
        source_artifact_hash: Any,
        source_manifest: Any,
    ) -> None:
        if metric is _MISSING or not isinstance(metric, dict):
            add(code, INCONCLUSIVE, expected="complete position-clustered metric")
            return
        required_values = (
            metric.get("metric", _MISSING),
            metric.get("available", _MISSING),
            metric.get("complete", _MISSING),
            metric.get("estimate", _MISSING),
            metric.get("lower_bound", _MISSING),
            metric.get("upper_bound", _MISSING),
            metric.get("color_pairs", _MISSING),
            metric.get("position_clusters", _MISSING),
            metric.get("degrees_of_freedom", _MISSING),
            _path(metric, "small_cluster_correction", "method"),
            _path(metric, "small_cluster_correction", "variance_multiplier"),
            metric.get("one_sided_alpha", _MISSING),
            metric.get("allocated_one_sided_confidence", _MISSING),
            metric.get("nominal_one_sided_confidence", _MISSING),
            _path(metric, "bootstrap", "method"),
            _path(metric, "bootstrap", "replications"),
            _path(metric, "bootstrap", "seed"),
            _path(metric, "bootstrap", "stratum_dimensions"),
            _path(metric, "bootstrap", "stratum_cluster_counts"),
        )
        if any(value is _MISSING or value is None for value in required_values):
            add(
                code,
                INCONCLUSIVE,
                actual=[None if value is _MISSING else value for value in required_values],
                expected="complete inference metadata",
            )
            return
        (
            metric_name,
            available,
            complete,
            estimate,
            lower_bound,
            upper_bound,
            color_pairs,
            clusters,
            degrees_of_freedom,
            correction,
            variance_multiplier,
            actual_alpha,
            allocated_confidence,
            actual_nominal_confidence,
            bootstrap_method,
            replications,
            bootstrap_seed,
            bootstrap_dimensions,
            bootstrap_stratum_counts,
        ) = required_values
        expected_bootstrap_seed = (
            int.from_bytes(
                hashlib.sha256(
                    f"{active_policy['bootstrap']['seed']}:{metric_name}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            if isinstance(metric_name, str)
            else None
        )
        valid_bootstrap_counts = (
            isinstance(bootstrap_stratum_counts, list)
            and all(
                isinstance(entry, dict)
                and _is_nonnegative_integer(entry.get("position_clusters"))
                for entry in bootstrap_stratum_counts
            )
        )
        bootstrap_cluster_total = (
            sum(entry["position_clusters"] for entry in bootstrap_stratum_counts)
            if valid_bootstrap_counts
            else None
        )
        cluster_count_valid = (
            clusters == expected_minimum_clusters
            if v2_policy and expected_minimum_clusters is not None
            else isinstance(clusters, int)
            and not isinstance(clusters, bool)
            and clusters >= 2
        )
        well_formed = (
            isinstance(metric_name, str)
            and metric_name == expected_metric_name
            and available is True
            and complete is True
            and _is_finite_number(estimate)
            and _is_finite_number(lower_bound)
            and _is_finite_number(upper_bound)
            and float(lower_bound) <= float(estimate) <= float(upper_bound)
            and isinstance(color_pairs, int)
            and not isinstance(color_pairs, bool)
            and color_pairs == expected_color_pairs
            and isinstance(clusters, int)
            and not isinstance(clusters, bool)
            and cluster_count_valid
            and isinstance(expected_position_ids, list)
            and clusters == len(expected_position_ids)
            and degrees_of_freedom == clusters - 1
            and correction == "CR1_BESSEL_WITH_STUDENT_T"
            and _is_finite_number(variance_multiplier)
            and math.isclose(
                float(variance_multiplier),
                clusters / (clusters - 1.0),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and _is_finite_number(actual_alpha)
            and alpha is not None
            and math.isclose(float(actual_alpha), float(alpha), rel_tol=0.0, abs_tol=1.0e-15)
            and _is_finite_number(allocated_confidence)
            and math.isclose(
                float(allocated_confidence),
                1.0 - float(alpha),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            and _is_finite_number(actual_nominal_confidence)
            and math.isclose(
                float(actual_nominal_confidence),
                nominal_confidence,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            and bootstrap_method
            == (
                "stratified_wild_position_cluster_rademacher_"
                "centered_within_declared_strata"
            )
            and replications == bootstrap_replications
            and bootstrap_seed == expected_bootstrap_seed
            and bootstrap_dimensions == ["schedule_id", "suite"]
            and bootstrap_cluster_total == clusters
        )
        position_values = metric.get("position_values")
        valid_position_values = (
            isinstance(position_values, list)
            and all(
                isinstance(row, dict) and _nonempty(row.get("position_id"))
                for row in position_values
            )
        )
        metric_position_ids = (
            sorted(row["position_id"] for row in position_values)
            if valid_position_values
            else None
        )
        one_pair_per_cluster = (
            valid_position_values
            and all(row.get("pair_count") == 1 for row in position_values)
            and len(position_values) == color_pairs
        )
        recomputed_strata: Dict[str, int] = {}
        if isinstance(position_values, list):
            for row in position_values:
                if not isinstance(row, dict) or not isinstance(row.get("stratum"), dict):
                    recomputed_strata = {}
                    break
                key = json.dumps(
                    row["stratum"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                recomputed_strata[key] = recomputed_strata.get(key, 0) + 1
        recomputed_stratum_counts = [
            {
                **json.loads(key),
                "position_clusters": count,
            }
            for key, count in sorted(recomputed_strata.items())
        ]
        well_formed = (
            well_formed
            and metric_position_ids == expected_position_ids
            and (not v2_policy or one_pair_per_cluster)
            and bootstrap_stratum_counts == recomputed_stratum_counts
            and _is_sha256(source_artifact_hash)
            and isinstance(source_manifest, dict)
            and expected_metric_name in source_manifest.get("metric_names", [])
        )
        add(
            code,
            PASS if well_formed else FAIL,
            actual={
                "metric": metric_name,
                "source_artifact_hash": source_artifact_hash,
                "available": available,
                "complete": complete,
                "estimate": estimate,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "color_pairs": color_pairs,
                "position_clusters": clusters,
                "one_pair_per_independent_position_cluster": one_pair_per_cluster,
                "degrees_of_freedom": degrees_of_freedom,
                "small_cluster_correction": correction,
                "variance_multiplier": variance_multiplier,
                "one_sided_alpha": actual_alpha,
                "allocated_one_sided_confidence": allocated_confidence,
                "nominal_one_sided_confidence": actual_nominal_confidence,
                "bootstrap_method": bootstrap_method,
                "bootstrap_replications": replications,
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_stratum_dimensions": bootstrap_dimensions,
                "bootstrap_stratum_cluster_counts": bootstrap_stratum_counts,
                "recomputed_stratum_cluster_counts": recomputed_stratum_counts,
                "position_ids": metric_position_ids,
            },
            expected={
                "metric": expected_metric_name,
                "source_artifact_hash": "bound finalized statistics artifact",
                "available": True,
                "complete": True,
                "finite_ordered_estimate_and_bounds": True,
                "color_pairs": expected_color_pairs,
                "minimum_independent_position_clusters": expected_minimum_clusters,
                "one_pair_per_independent_position_cluster": v2_policy,
                "position_ids": expected_position_ids,
                "degrees_of_freedom": "position_clusters-1",
                "small_cluster_correction": "CR1_BESSEL_WITH_STUDENT_T",
                "variance_multiplier": "G/(G-1)",
                "one_sided_alpha": alpha,
                "allocated_one_sided_confidence": (
                    1.0 - float(alpha) if alpha is not None else None
                ),
                "nominal_one_sided_confidence": nominal_confidence,
                "bootstrap_method": (
                    "stratified_wild_position_cluster_rademacher_"
                    "centered_within_declared_strata"
                ),
                "bootstrap_replications": bootstrap_replications,
                "bootstrap_seed": expected_bootstrap_seed,
                "bootstrap_stratum_dimensions": ["schedule_id", "suite"],
                "bootstrap_stratum_cluster_counts": recomputed_stratum_counts,
            },
        )

    def lower_margin_check(
        code: str,
        metric: Any,
        threshold: float,
    ) -> None:
        lower = _path(metric, "lower_bound")
        if lower is _MISSING or lower is None:
            add(code, INCONCLUSIVE, expected=threshold, operator=">")
            return
        if not _is_finite_number(lower):
            add(code, FAIL, actual=lower, expected=threshold, operator=">")
            return
        lower_value = float(lower)
        if lower_value > threshold:
            add(
                code,
                PASS,
                actual=lower_value,
                expected=threshold,
                operator=">",
            )
            return
        if not v2_policy:
            add(
                code,
                FAIL,
                actual=lower_value,
                expected=threshold,
                operator=">",
            )
            return

        upper = _path(metric, "upper_bound")
        conditional_power = _first_present(
            _path(metric, "conditional_power_to_final_look"),
            _path(metric, "conditional_power"),
        )
        futility_threshold = stage_3.get(
            "low_conditional_power_stop_threshold"
        )
        futility_proven = False
        if conditional_power is not _MISSING:
            futility_code = code.removesuffix("_LOWER_BOUND") + "_CONDITIONAL_POWER"
            if not (
                _is_finite_number(conditional_power)
                and 0.0 <= float(conditional_power) <= 1.0
                and _is_finite_number(futility_threshold)
            ):
                add(
                    futility_code,
                    FAIL,
                    actual=conditional_power,
                    expected="finite probability in [0,1]",
                )
            else:
                futility_proven = (
                    look_number == 1
                    and float(conditional_power) <= float(futility_threshold)
                )
                add(
                    futility_code,
                    FAIL if futility_proven else PASS,
                    actual=float(conditional_power),
                    expected=float(futility_threshold),
                    operator=">",
                    fail_reason=futility_code + "_STOP",
                )

        if upper is not _MISSING and upper is not None and not _is_finite_number(upper):
            add(
                code,
                FAIL,
                actual={"lower_bound": lower_value, "upper_bound": upper},
                expected=threshold,
                operator=">",
            )
        elif (
            futility_proven
            or (
                _is_finite_number(upper)
                and float(upper) <= threshold
            )
        ):
            add(
                code,
                FAIL,
                actual={
                    "lower_bound": lower_value,
                    "upper_bound": (
                        float(upper) if _is_finite_number(upper) else None
                    ),
                    "conditional_power": (
                        float(conditional_power)
                        if _is_finite_number(conditional_power)
                        else None
                    ),
                },
                expected=threshold,
                operator=">",
                fail_reason=code + "_FUTILITY_OR_HARM_PROVEN",
            )
        else:
            add(
                code,
                INCONCLUSIVE,
                actual={
                    "lower_bound": lower_value,
                    "upper_bound": (
                        float(upper) if _is_finite_number(upper) else None
                    ),
                },
                expected=threshold,
                operator=">",
                missing_reason=code + "_MARGIN_NOT_ESTABLISHED",
            )
            if _is_finite_number(upper):
                continuation_inconclusive_codes.add(code)

    optimization_metrics = (
        (
            "POWERED_UTILITY_VS_CHAMPION",
            utility_champion,
            thresholds["powered_utility_vs_champion_lower_bound_strictly_above"],
            "powered_candidate_vs_champion",
            "realized_utility",
        ),
        (
            "POWERED_UTILITY_VS_ORIGINAL",
            utility_original,
            thresholds["powered_utility_vs_original_lower_bound_strictly_above"],
            "powered_candidate_vs_original",
            "realized_utility",
        ),
        (
            "COMBINED_LEAD_UTILITY",
            combined_lead,
            thresholds["combined_lead_utility_lower_bound_strictly_above"],
            "combined_lead",
            "combined_lead_realized_utility",
        ),
        (
            "POWERED_WIN_RATE_VS_CHAMPION",
            win_champion,
            thresholds["powered_win_rate_vs_champion_lower_bound_strictly_above"],
            "powered_candidate_vs_champion",
            "win_rate",
        ),
        (
            "POWERED_WIN_RATE_VS_ORIGINAL",
            win_original,
            thresholds["powered_win_rate_vs_original_lower_bound_strictly_above"],
            "powered_candidate_vs_original",
            "win_rate",
        ),
        (
            "STANDARD_WIN_RATE_VS_ORIGINAL",
            standard_win,
            thresholds["standard_win_rate_vs_original_lower_bound_strictly_above"],
            "standard_candidate_vs_original",
            "win_rate",
        ),
    )
    for code, metric, threshold, source_cell, metric_name in optimization_metrics:
        if source_cell == "combined_lead":
            source_hash = combined_artifact_hash
            source_manifest = combined_manifest
            source_pairs = expected_combined_pairs
            source_minimum_clusters = expected_combined_clusters
            source_positions = combined_position_ids
        else:
            source_hash = _path(cells.get(source_cell, _MISSING), "statistics_artifact_hash")
            source_manifest = cell_statistics_manifests.get(source_cell)
            source_pairs = expected_pair_counts.get(source_cell)
            source_minimum_clusters = expected_cluster_counts.get(source_cell)
            source_positions = _path(source_manifest, "position_ids")
        inference_check(
            code + "_INFERENCE",
            metric,
            routine_alpha,
            0.95,
            expected_metric_name=metric_name,
            expected_color_pairs=source_pairs,
            expected_minimum_clusters=source_minimum_clusters,
            expected_position_ids=source_positions,
            source_artifact_hash=source_hash,
            source_manifest=source_manifest,
        )
        lower_margin_check(code + "_LOWER_BOUND", metric, threshold)

    discovery = evidence.get("discovery", _MISSING)
    require_boolean(
        "STAGE_1_DISCOVERY_PASSED",
        _path(discovery, "stage_1_passed"),
        True,
    )
    require_boolean(
        "STAGE_2_FINALIST_SELECTION_PASSED",
        _path(discovery, "stage_2_passed"),
        True,
    )
    require_boolean(
        "NOT_DOMINATED_BY_LATER_SAFE_FINALIST",
        _path(discovery, "dominated_by_later_safe_finalist"),
        False,
    )
    require_boolean(
        "CONFIRMATION_SCHEDULE_INDEPENDENT",
        _path(discovery, "confirmation_schedule_independent"),
        True,
    )
    require_equal(
        "CONFIRMATION_CANDIDATE_COUNT",
        _path(discovery, "confirmation_candidate_count"),
        1,
    )

    risk_container = evidence.get("risk_differences", _MISSING)
    risk_thresholds = thresholds["candidate_minus_reference_risk_upper_bounds"]
    for risk_name in sorted(risk_thresholds):
        source_cell = RISK_CELL_BINDINGS[risk_name]
        source_artifact = cell_artifacts.get(source_cell, _MISSING)
        risk_metric = _metric_from(
            _path(source_artifact, "risk_differences"),
            risk_name,
        )
        published_risk_metric = _metric_from(risk_container, risk_name)
        require_equal(
            "RISK_" + risk_name.upper() + "_PUBLISHED_BINDING",
            published_risk_metric,
            risk_metric,
        )
        code = "RISK_" + risk_name.upper()
        source_manifest = cell_statistics_manifests.get(source_cell)
        inference_check(
            code + "_INFERENCE",
            risk_metric,
            catastrophe_alpha,
            0.99,
            expected_metric_name=risk_name,
            expected_color_pairs=expected_pair_counts.get(source_cell),
            expected_minimum_clusters=expected_cluster_counts.get(source_cell),
            expected_position_ids=_path(source_manifest, "position_ids"),
            source_artifact_hash=_path(
                cells.get(source_cell, _MISSING), "statistics_artifact_hash"
            ),
            source_manifest=source_manifest,
        )
        risk_upper = _path(risk_metric, "upper_bound")
        risk_lower = _path(risk_metric, "lower_bound")
        risk_limit = risk_thresholds[risk_name]
        if risk_upper is _MISSING or risk_upper is None:
            add(
                code + "_UPPER_BOUND",
                INCONCLUSIVE,
                expected=risk_limit,
                operator="<=",
            )
        elif not _is_finite_number(risk_upper):
            add(
                code + "_UPPER_BOUND",
                FAIL,
                actual=risk_upper,
                expected=risk_limit,
                operator="<=",
            )
        elif float(risk_upper) <= risk_limit:
            add(
                code + "_UPPER_BOUND",
                PASS,
                actual=float(risk_upper),
                expected=risk_limit,
                operator="<=",
            )
        else:
            proven_violation = (
                _is_finite_number(risk_lower) and float(risk_lower) > risk_limit
            )
            add(
                code + "_UPPER_BOUND",
                FAIL if proven_violation else INCONCLUSIVE,
                actual=float(risk_upper),
                expected=risk_limit,
                operator="<=",
                missing_reason=code + "_SAFETY_MARGIN_NOT_ESTABLISHED",
            )
            if not proven_violation:
                continuation_inconclusive_codes.add(code + "_UPPER_BOUND")
        require_equal(
            code + "_DIRECTION",
            _path(risk_metric, "direction"),
            "candidate_minus_reference",
        )
        require_boolean(
            code + "_MATCHED_WITHIN_GAME",
            _path(risk_metric, "matched_within_game"),
            True,
        )
        candidate_events = _path(risk_metric, "candidate_events")
        reference_events = _path(risk_metric, "reference_events")
        for role, event_count in (
            ("CANDIDATE", candidate_events),
            ("REFERENCE", reference_events),
        ):
            if event_count is _MISSING or event_count is None:
                add(
                    f"{code}_{role}_EVENT_COUNT",
                    INCONCLUSIVE,
                    expected="nonnegative integer",
                )
            elif not _is_nonnegative_integer(event_count):
                add(
                    f"{code}_{role}_EVENT_COUNT",
                    FAIL,
                    actual=event_count,
                    expected="nonnegative integer",
                )
            else:
                add(
                    f"{code}_{role}_EVENT_COUNT",
                    PASS,
                    actual=event_count,
                    expected="nonnegative integer",
                )
        if (
            _is_nonnegative_integer(candidate_events)
            and _is_nonnegative_integer(reference_events)
            and candidate_events == 0
            and reference_events == 0
        ):
            clusters = _path(risk_metric, "position_clusters")
            if (
                isinstance(clusters, int)
                and not isinstance(clusters, bool)
                and clusters > 0
                and _is_finite_number(catastrophe_alpha)
            ):
                exact_zero_bound = exact_zero_event_upper_bound(
                    float(catastrophe_alpha),
                    clusters,
                )
                stored_zero_bound = _path(
                    risk_metric, "zero_event_uncertainty_upper_bound"
                )
                if not _is_finite_number(stored_zero_bound):
                    add(
                        code + "_ZERO_EVENT_UNCERTAINTY",
                        INCONCLUSIVE,
                        expected=exact_zero_bound,
                    )
                else:
                    add(
                        code + "_ZERO_EVENT_UNCERTAINTY",
                        PASS
                        if math.isclose(
                            float(stored_zero_bound),
                            exact_zero_bound,
                            rel_tol=0.0,
                            abs_tol=1.0e-15,
                        )
                        else FAIL,
                        actual=float(stored_zero_bound),
                        expected=exact_zero_bound,
                        operator="==",
                    )
                for suffix, bound in (
                    ("ANALYTIC", _path(risk_metric, "upper_bound")),
                    ("BOOTSTRAP", _path(risk_metric, "bootstrap", "upper_bound")),
                ):
                    if not _is_finite_number(bound):
                        add(
                            code + "_ZERO_EVENT_" + suffix + "_CORRECTION",
                            INCONCLUSIVE,
                            expected=f">={exact_zero_bound}",
                        )
                    else:
                        add(
                            code + "_ZERO_EVENT_" + suffix + "_CORRECTION",
                            PASS if float(bound) + 1.0e-15 >= exact_zero_bound else FAIL,
                            actual=float(bound),
                            expected=exact_zero_bound,
                            operator=">=",
                        )
                require_equal(
                    code + "_ZERO_EVENT_CLUSTER_COUNT",
                    _path(
                        risk_metric,
                        "zero_event_independent_position_clusters",
                    ),
                    clusters,
                )
            else:
                add(code + "_ZERO_EVENT_UNCERTAINTY", INCONCLUSIVE)
                add(code + "_ZERO_EVENT_ANALYTIC_CORRECTION", INCONCLUSIVE)
                add(code + "_ZERO_EVENT_BOOTSTRAP_CORRECTION", INCONCLUSIVE)
                add(code + "_ZERO_EVENT_CLUSTER_COUNT", INCONCLUSIVE)
            require_equal(
                code + "_ZERO_EVENT_METHOD",
                _path(risk_metric, "zero_event_uncertainty_method"),
                "one_sided_exact_no_event_bound_using_independent_position_clusters",
            )
        else:
            add(
                code + "_ZERO_EVENT_UNCERTAINTY",
                PASS,
                actual="not_applicable",
                expected="nonzero bound when both event counts are zero",
            )
            add(
                code + "_ZERO_EVENT_ANALYTIC_CORRECTION",
                PASS,
                actual="not_applicable",
                expected="nonzero bound when both event counts are zero",
            )
            add(
                code + "_ZERO_EVENT_BOOTSTRAP_CORRECTION",
                PASS,
                actual="not_applicable",
                expected="nonzero bound when both event counts are zero",
            )
            add(
                code + "_ZERO_EVENT_CLUSTER_COUNT",
                PASS,
                actual="not_applicable",
                expected="independent cluster count for zero events",
            )
            add(
                code + "_ZERO_EVENT_METHOD",
                PASS,
                actual="not_applicable",
                expected="exact no-event bound when both event counts are zero",
            )

    validity = evidence.get("validity", _MISSING)
    require_boolean(
        "VALIDITY_PROMOTION_VALID",
        _path(validity, "promotion_valid"),
        True,
    )
    zero_fields = (
        "missing_games",
        "duplicate_game_ids",
        "incomplete_pairs",
        "duplicate_pair_members",
        "resignations",
        "turn_limits",
        "unresolved_rows",
        "structural_errors",
        "perspective_violations",
        "clamp_violations",
        "endpoint_violations",
        "nonfinite_violations",
        "decomposition_violations",
    )
    for field in zero_fields:
        value = _path(validity, field)
        code = "VALIDITY_" + field.upper()
        if value is _MISSING or value is None:
            add(code, INCONCLUSIVE, expected=0, operator="==")
        elif not _is_nonnegative_integer(value):
            add(code, FAIL, actual=value, expected="nonnegative integer")
        else:
            add(code, PASS if value == 0 else FAIL, actual=value, expected=0, operator="==")

    missing_numeric_scores = _path(validity, "resolved_missing_numeric_scores")
    if missing_numeric_scores is _MISSING or missing_numeric_scores is None:
        add("VALIDITY_MISSING_NUMERIC_SCORE_REPORTED", INCONCLUSIVE, expected="nonnegative integer")
    elif _is_nonnegative_integer(missing_numeric_scores):
        add(
            "VALIDITY_MISSING_NUMERIC_SCORE_REPORTED",
            PASS,
            actual=missing_numeric_scores,
            expected="nonnegative integer",
        )
    else:
        add(
            "VALIDITY_MISSING_NUMERIC_SCORE_REPORTED",
            FAIL,
            actual=missing_numeric_scores,
            expected="nonnegative integer",
        )

    true_no_results = _path(validity, "true_no_results")
    total_games = _path(validity, "total_games")
    if (
        true_no_results is _MISSING
        or total_games is _MISSING
        or not _is_nonnegative_integer(true_no_results)
        or not isinstance(total_games, int)
        or isinstance(total_games, bool)
        or total_games <= 0
        or true_no_results > total_games
    ):
        if true_no_results is _MISSING or total_games is _MISSING:
            add("TRUE_NO_RESULT_RATE", INCONCLUSIVE, expected="valid counts")
        else:
            add(
                "TRUE_NO_RESULT_RATE",
                FAIL,
                actual={"true_no_results": true_no_results, "total_games": total_games},
                expected="0 <= true_no_results <= total_games and total_games > 0",
            )
    else:
        no_result_rate = true_no_results / total_games
        numeric_threshold(
            "TRUE_NO_RESULT_RATE",
            no_result_rate,
            "<",
            thresholds["true_no_result_rate_strictly_below"],
        )
        supplied_rate = _path(validity, "true_no_result_rate")
        if supplied_rate is not _MISSING:
            if not _is_finite_number(supplied_rate) or not math.isclose(
                float(supplied_rate), no_result_rate, rel_tol=0.0, abs_tol=1.0e-15
            ):
                add(
                    "TRUE_NO_RESULT_RATE_CONSISTENCY",
                    FAIL,
                    actual=supplied_rate,
                    expected=no_result_rate,
                    operator="==",
                )
            else:
                add(
                    "TRUE_NO_RESULT_RATE_CONSISTENCY",
                    PASS,
                    actual=float(supplied_rate),
                    expected=no_result_rate,
                    operator="==",
                )
    require_boolean(
        "FULL_MOVE_DIAGNOSTICS",
        _path(validity, "full_move_diagnostics"),
        True,
    )

    exploitability = evidence.get("exploitability", _MISSING)
    stage_0_policy = active_policy["evaluation_stages"][
        "stage_0_integrity_and_fixed_probes"
    ]
    require_boolean(
        "STAGE_0_PASSED",
        _path(exploitability, "stage_0_passed"),
        True,
    )
    stage_0_bound_fields = (
        ("fixed_analysis_positions", "fixed_analysis_visits")
        if machine_review_v3
        else (
            "fixed_analysis_positions",
            "fixed_analysis_visits",
            "exploitability_sentinel_positions",
            "exploitability_sentinel_visits",
        )
    )
    for field in stage_0_bound_fields:
        require_equal(
            "STAGE_0_" + field.upper(),
            _path(exploitability, field),
            stage_0_policy[field],
        )
    for field in (
        "hard_tactical_failures",
        "hard_exploitability_failures",
        "unresolved_failures",
        "model_runtime_errors",
    ):
        value = _path(exploitability, field)
        code = "EXPLOITABILITY_" + field.upper()
        if value is _MISSING or value is None:
            add(code, INCONCLUSIVE, expected=0, operator="==")
        elif not _is_nonnegative_integer(value):
            add(code, FAIL, actual=value, expected="nonnegative integer")
        else:
            add(code, PASS if value == 0 else FAIL, actual=value, expected=0, operator="==")
    require_boolean(
        "ENDPOINT_MASS_NOT_DOMINATING_SELECTED_MOVE",
        _path(exploitability, "selected_move_endpoint_mass_dominated"),
        False,
    )
    require_boolean(
        "VISIT_STABILITY_ACCEPTABLE",
        _path(exploitability, "visit_stability_acceptable"),
        True,
    )

    provenance = evidence.get("provenance", _MISSING)
    require_boolean("PROVENANCE_COMPLETE", _path(provenance, "complete"), True)
    require_boolean("PROVENANCE_INPUTS_IMMUTABLE", _path(provenance, "immutable_inputs"), True)
    require_boolean(
        "PROVENANCE_ORIGINAL_IMMUTABLE",
        _path(provenance, "immutable_original"),
        True,
    )
    require_sha(
        "PROVENANCE_CANDIDATE_HASH",
        _path(provenance, "candidate_hash"),
        candidate_hash,
    )
    require_sha(
        "PROVENANCE_CHAMPION_HASH",
        _path(provenance, "champion_hash"),
        champion_hash,
    )
    require_sha(
        "PROVENANCE_ORIGINAL_HASH",
        _path(provenance, "original_hash"),
        original_hash,
    )
    require_sha(
        "PROVENANCE_POLICY_HASH",
        _path(provenance, "policy_hash"),
        computed_policy_hash,
    )
    source_revision_hash = _path(provenance, "source_revision_hash")
    require_equal(
        "PROVENANCE_SOURCE_REVISION_HASH",
        source_revision_hash,
        active_policy["frozen_plan"]["source_revision"],
    )
    require_sha("PROVENANCE_BINARY_HASH", _path(provenance, "binary_hash"))
    for cell_name in sorted(cell_specs):
        require_sha(
            "PROVENANCE_CELL_BINARY_" + cell_name.upper(),
            _path(cells.get(cell_name, _MISSING), "katago_binary_hash"),
            _path(provenance, "binary_hash")
            if _is_sha256(_path(provenance, "binary_hash"))
            else None,
        )

    for config_name in ("powered_match", "standard_match"):
        require_sha(
            "PROVENANCE_CONFIG_" + config_name.upper(),
            _path(provenance, "config_hashes", config_name),
        )
    for cell_name in sorted(cell_specs):
        config_name = (
            "powered_match"
            if cell_specs[cell_name]["search_mode"] == "powered"
            else "standard_match"
        )
        provenance_config = _path(provenance, "config_hashes", config_name)
        cell_config = _path(cells.get(cell_name, _MISSING), "config_hash")
        if provenance_config is _MISSING:
            add(
                "PROVENANCE_CELL_CONFIG_" + cell_name.upper(),
                INCONCLUSIVE,
                expected="matching provenance config hash",
            )
        else:
            require_sha(
                "PROVENANCE_CELL_CONFIG_" + cell_name.upper(),
                cell_config,
                provenance_config,
            )
    for suite_name in (() if machine_review_v3 else ("tactical", "exploitability")):
        require_sha(
            "PROVENANCE_SUITE_" + suite_name.upper(),
            _path(provenance, "suite_hashes", suite_name),
        )
    for cell_name in sorted(cell_specs):
        cell_suite_hash = _path(cells.get(cell_name, _MISSING), "suite_hash")
        require_sha(
            "PROVENANCE_SUITE_BINDING_" + cell_name.upper(),
            _path(provenance, "suite_hashes", cell_name),
            cell_suite_hash if _is_sha256(cell_suite_hash) else None,
        )
    for cell_name in sorted(cell_specs):
        provenance_schedule = _path(provenance, "schedule_hashes", cell_name)
        cell_schedule = _path(cells.get(cell_name, _MISSING), "schedule_hash")
        if provenance_schedule is _MISSING:
            add(
                "PROVENANCE_SCHEDULE_" + cell_name.upper(),
                INCONCLUSIVE,
                expected=cell_schedule if cell_schedule is not _MISSING else "sha256",
            )
        else:
            require_sha(
                "PROVENANCE_SCHEDULE_" + cell_name.upper(),
                provenance_schedule,
                cell_schedule if cell_schedule is not _MISSING else None,
            )
    discovery_schedule_hash = _path(provenance, "discovery_schedule_hash")
    require_sha("PROVENANCE_DISCOVERY_SCHEDULE_HASH", discovery_schedule_hash)
    require_equal(
        "PROVENANCE_DISCOVERY_SCHEDULE_MANIFEST_BINDING",
        discovery_schedule_hash,
        _path(suite_manifest, "discovery_schedule_hash"),
    )
    confirmation_schedule_hashes = [
        _path(cells.get(cell_name, _MISSING), "schedule_hash")
        for cell_name in sorted(cell_specs)
    ]
    if _is_sha256(discovery_schedule_hash) and all(
        _is_sha256(value) for value in confirmation_schedule_hashes
    ):
        independent = all(
            discovery_schedule_hash != value for value in confirmation_schedule_hashes
        )
        add(
            "DISCOVERY_CONFIRMATION_HASH_INDEPENDENCE",
            PASS if independent else FAIL,
            actual=independent,
            expected=True,
            operator="==",
        )
    else:
        add("DISCOVERY_CONFIRMATION_HASH_INDEPENDENCE", INCONCLUSIVE, expected=True)

    if v2_policy:
        for code, check in checks.items():
            if (
                check["status"] == INCONCLUSIVE
                and code not in continuation_inconclusive_codes
            ):
                check["status"] = FAIL
                reason_codes.append(code + "_FAIL_CLOSED")

    statuses = {check["status"] for check in checks.values()}
    if FAIL in statuses:
        decision = FAIL
    elif INCONCLUSIVE in statuses:
        decision = INCONCLUSIVE
    else:
        decision = PASS
    inconclusive_codes = {
        code for code, check in checks.items() if check["status"] == INCONCLUSIVE
    }
    continuation_eligible = (
        decision == INCONCLUSIVE
        and v2_policy
        and look_number == 1
        and bool(inconclusive_codes)
        and inconclusive_codes.issubset(continuation_inconclusive_codes)
    )
    if decision == PASS:
        next_action = PROMOTE
    elif decision == FAIL:
        next_action = STOP_HARM
    elif continuation_eligible:
        next_action = CONTINUE_TO_LOOK_2
    else:
        next_action = STOP_MAXIMUM_INCONCLUSIVE

    final_50_metric = _metric_from(
        _path(powered_champion, "risk_differences"),
        "final_50",
    )
    ranking_source_cell = "powered_candidate_vs_champion"
    ranking_source = cells.get(ranking_source_cell, _MISSING)
    ranking_source_bound = all(
        checks.get(code, {}).get("status") == PASS
        for code in (
            "MATRIX_POWERED_CANDIDATE_VS_CHAMPION_STATISTICS_ARTIFACT_HASH",
            "MATRIX_POWERED_CANDIDATE_VS_CHAMPION_STATISTICS_MANIFEST_HASH",
            "POWERED_UTILITY_VS_CHAMPION_INFERENCE",
            "RISK_FINAL_50_INFERENCE",
            "RISK_FINAL_50_PUBLISHED_BINDING",
        )
    )
    utility_ranking_bound = _path(utility_champion, "lower_bound")
    final_50_ranking_bound = _path(final_50_metric, "upper_bound")
    ranking_source_bound = (
        ranking_source_bound
        and _is_finite_number(utility_ranking_bound)
        and _is_finite_number(final_50_ranking_bound)
    )
    realized_powered_utility_lower_bound = (
        float(utility_ranking_bound) if ranking_source_bound else None
    )
    final50_risk_upper_bound = (
        float(final_50_ranking_bound) if ranking_source_bound else None
    )
    ranking_summary = {
        "schema_version": 1,
        "source_bound": ranking_source_bound,
        "source_cell": ranking_source_cell,
        "candidate_hash": candidate_hash,
        "look_number": look_number if isinstance(look_number, int) else None,
        "statistics_artifact_hash": report_value(
            _path(ranking_source, "statistics_artifact_hash")
        ),
        "statistics_manifest_hash": report_value(
            _path(ranking_source, "statistics_manifest_hash")
        ),
        "realized_powered_utility_lower_bound": (
            realized_powered_utility_lower_bound
        ),
        "final50_risk_upper_bound": final50_risk_upper_bound,
        "final_50_risk_upper_bound": final50_risk_upper_bound,
    }
    return {
        "schema_version": GATE_REPORT_SCHEMA_VERSION,
        "schema_name": "risk-seeking-promotion-gate-report",
        "decision": decision,
        "next_action": next_action,
        "continuation_eligible": continuation_eligible,
        "reason_codes": sorted(set(reason_codes)),
        "checks": [checks[code] for code in sorted(checks)],
        "finalized": True,
        "candidate_hash": candidate_hash,
        "champion_hash": champion_hash,
        "tested_champion_hash": champion_hash,
        "original_hash": original_hash,
        "evaluation_key": evaluation_key,
        "config_hash": config_hash,
        "schedule_hash": schedule_hash,
        "policy_hash": computed_policy_hash,
        "policy_version": active_policy["policy_version"],
        "look_number": look_number if look_number is not _MISSING else None,
        "ranking_summary": ranking_summary,
        "realized_powered_utility_lower_bound": (
            realized_powered_utility_lower_bound
        ),
        "final50_risk_upper_bound": final50_risk_upper_bound,
        "alpha_allocation": {
            "method": active_policy["confidence"]["sequential_testing"][
                "allocation_method"
            ],
            "routine_one_sided_alpha": (
                look_config.get("routine_one_sided_alpha") if look_config else None
            ),
            "catastrophe_one_sided_alpha": (
                look_config.get("catastrophe_one_sided_alpha") if look_config else None
            ),
            "routine_allocated_one_sided_confidence": (
                1.0 - look_config["routine_one_sided_alpha"] if look_config else None
            ),
            "catastrophe_allocated_one_sided_confidence": (
                1.0 - look_config["catastrophe_one_sided_alpha"]
                if look_config
                else None
            ),
        },
    }


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


evaluate_gate = evaluate_promotion_gate
gate_promotion = evaluate_promotion_gate
run_promotion_gate = evaluate_promotion_gate
compute_policy_hash = policy_hash


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a finalized checkpoint-promotion evidence JSON file."
    )
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("-o", "--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        with args.evidence.open("r", encoding="utf-8") as handle:
            evidence = json.load(handle)
        if not isinstance(evidence, dict):
            raise ValueError("evidence must be a JSON object")
        report = evaluate_promotion_gate(evidence, policy=load_policy(args.policy))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0 if report["decision"] == PASS else 1 if report["decision"] == FAIL else 2


if __name__ == "__main__":
    raise SystemExit(main())
