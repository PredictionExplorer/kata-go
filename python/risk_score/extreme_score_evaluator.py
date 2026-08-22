"""Held-out, fixed-cohort expected-maximum checkpoint evaluation.

The promotion estimand is ``J_N = mean_c(max_i score[c, i])`` over
precommitted cohorts of exactly ``N`` trials. Candidate and reference maxima
are paired within each cohort, and exact sign inference uses the precommitted
Black/White clusters. Lifetime records are retained only as non-gating diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

PLAN_REQUEST_CONTRACT = "risk-score-extreme-score-plan-request-v1"
PLAN_CONTRACT = "risk-score-extreme-score-plan-v1"
REPORT_CONTRACT = "risk-score-extreme-score-report-v1"
STATUS_CONTRACT = "risk-score-extreme-score-status-v1"
POLICY_CONTRACT = "risk-score-held-out-expected-max-policy-v1"
POLICY_VERSION = "risk-score-held-out-expected-max-v1"
DEFAULT_POLICY_PATH = Path(__file__).with_name("extreme_score_policy_v1.json")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARMS = ("candidate", "reference")
_COLORS = ("B", "W")
_MAX_COHORT_SIZE = 100_000
_PROMOTION_RULE = (
    "positive_paired_delta_and_one_sided_exact_sign_p_value_at_or_below_alpha_"
    "in_every_required_slice"
)

_REQUEST_KEYS = {
    "schema_version",
    "contract",
    "candidate_model",
    "reference_model",
    "config",
    "cohort_size",
    "legal_score_bounds",
    "cohorts",
    "rollback_recommendation",
}
_PLAN_KEYS = {
    *_REQUEST_KEYS,
    "policy",
    "execution_matrix_sha256",
    "plan_sha256",
}
_MODEL_KEYS = {"model_id", "sha256"}
_CONFIG_KEYS = {"config_id", "sha256"}
_BOUNDS_KEYS = {"minimum", "maximum"}
_ARTIFACT_BINDING_KEYS = {"path", "file_sha256"}
_ROLLBACK_KEYS = {
    "action",
    "reference_model",
    "reference_model_artifact",
    "trainer_checkpoint_artifact",
    "quarantine_candidate_on_failure",
}
_COHORT_REQUEST_KEYS = {
    "cohort_id",
    "cluster_id",
    "league_cell",
    "opponent_snapshot_id",
    "opponent_model_sha256",
    "focal_color",
    "seeds",
}
_COHORT_PLAN_KEYS = {*_COHORT_REQUEST_KEYS, "cohort_sha256"}
_POLICY_BINDING_KEYS = {
    "path",
    "file_sha256",
    "canonical_sha256",
    "policy_version",
}
_JOB_KEYS = {
    "schema_version",
    "plan_sha256",
    "arm",
    "model_sha256",
    "cohort_id",
    "cohort_sha256",
    "trial_index",
    "seed",
    "config_sha256",
    "league_cell",
    "opponent_snapshot_id",
    "opponent_model_sha256",
    "focal_color",
}
_RESULT_KEYS = {*_JOB_KEYS, "score", "no_result", "hit_turn_limit"}
_REPORT_KEYS = {
    "schema_version",
    "contract",
    "finalized",
    "plan_binding",
    "policy_binding",
    "objective",
    "result_bindings",
    "integrity",
    "cohort_maxima",
    "statistics",
    "decision",
    "reason_codes",
    "promotion_recommended",
    "rollback_recommendation",
    "rollback_recommendation_sha256",
    "non_gating_diagnostics",
    "report_sha256",
}


class ExtremeScoreEvaluatorError(ValueError):
    """An expected-max artifact or invocation violates its frozen contract."""


class ExtremeScoreIntegrityError(ExtremeScoreEvaluatorError):
    """Result rows cannot support a promotion-valid fixed-cohort evaluation."""

    def __init__(self, issues: Sequence[Mapping[str, Any]]):
        self.issues = tuple(dict(issue) for issue in issues)
        codes = sorted({str(issue.get("code", "UNKNOWN")) for issue in issues})
        super().__init__("result integrity failed: " + ", ".join(codes))


def canonical_json(value: Any) -> str:
    """Return the sole canonical JSON encoding used by evaluator artifacts."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ExtremeScoreEvaluatorError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ExtremeScoreEvaluatorError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _ensure_finite_json(value: Any, role: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExtremeScoreEvaluatorError(f"{role} contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _ensure_finite_json(item, role)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExtremeScoreEvaluatorError(f"{role} contains a non-string key")
            _ensure_finite_json(item, role)
        return
    raise ExtremeScoreEvaluatorError(
        f"{role} contains a non-JSON value of type {type(value).__name__}"
    )


def _decode_json(data: bytes, role: str) -> Any:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtremeScoreEvaluatorError(f"cannot decode {role}: {exc}") from exc
    _ensure_finite_json(value, role)
    return value


def _regular_file(path: Path, role: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ExtremeScoreEvaluatorError(f"{role} must be a regular non-symlink file")
    return source


def _load_json(
    path: Path,
    role: str,
    *,
    canonical: bool,
) -> dict[str, Any]:
    source = _regular_file(path, role)
    data = source.read_bytes()
    value = _decode_json(data, role)
    if not isinstance(value, dict):
        raise ExtremeScoreEvaluatorError(f"{role} must have an object root")
    if canonical and data != (canonical_json(value) + "\n").encode("utf-8"):
        raise ExtremeScoreEvaluatorError(
            f"{role} must be canonical newline-terminated JSON"
        )
    return value


def _load_jsonl(path: Path, role: str) -> tuple[dict[str, Any], ...]:
    source = _regular_file(path, role)
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(source.read_bytes().splitlines(), start=1):
        if not raw.strip():
            continue
        value = _decode_json(raw, f"{role} line {line_number}")
        if not isinstance(value, dict):
            raise ExtremeScoreEvaluatorError(
                f"{role} line {line_number} must be an object"
            )
        rows.append(value)
    return tuple(rows)


def _clone_json(value: Any, role: str) -> Any:
    _ensure_finite_json(value, role)
    return json.loads(canonical_json(value))


def _require_exact_keys(value: Any, expected: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExtremeScoreEvaluatorError(f"{role} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        raise ExtremeScoreEvaluatorError(
            f"{role} keys differ from contract; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return value


def _text(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise ExtremeScoreEvaluatorError(
            f"{role} must be a nonempty trimmed single-line string"
        )
    return value


def _sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ExtremeScoreEvaluatorError(
            f"{role} must be a lowercase 64-character SHA-256"
        )
    return value


def _positive_int(value: Any, role: str, maximum: int | None = None) -> int:
    if (
        type(value) is not int
        or value <= 0
        or (maximum is not None and value > maximum)
    ):
        suffix = f" no greater than {maximum}" if maximum is not None else ""
        raise ExtremeScoreEvaluatorError(f"{role} must be a positive integer{suffix}")
    return value


def _finite_number(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExtremeScoreEvaluatorError(f"{role} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ExtremeScoreEvaluatorError(f"{role} must be a finite number") from exc
    if not math.isfinite(number):
        raise ExtremeScoreEvaluatorError(f"{role} must be a finite number")
    return number


def _model_binding(value: Any, role: str) -> dict[str, str]:
    raw = _require_exact_keys(value, _MODEL_KEYS, role)
    return {
        "model_id": _text(raw["model_id"], f"{role}.model_id"),
        "sha256": _sha256(raw["sha256"], f"{role}.sha256"),
    }


def _config_binding(value: Any) -> dict[str, str]:
    raw = _require_exact_keys(value, _CONFIG_KEYS, "config")
    return {
        "config_id": _text(raw["config_id"], "config.config_id"),
        "sha256": _sha256(raw["sha256"], "config.sha256"),
    }


def _artifact_binding(
    value: Any,
    role: str,
    *,
    expected_file_sha256: str | None = None,
) -> dict[str, str]:
    raw = _require_exact_keys(value, _ARTIFACT_BINDING_KEYS, role)
    path_text = _text(raw["path"], f"{role}.path")
    path = Path(path_text)
    if not path.is_absolute() or path != Path(os.path.abspath(path)):
        raise ExtremeScoreEvaluatorError(f"{role}.path must be absolute and normalized")
    digest = _sha256(raw["file_sha256"], f"{role}.file_sha256")
    if expected_file_sha256 is not None and digest != expected_file_sha256:
        raise ExtremeScoreEvaluatorError(
            f"{role} hash must equal the bound reference model hash"
        )
    source = _regular_file(path, role)
    observed = file_sha256(source)
    if observed != digest:
        raise ExtremeScoreEvaluatorError(f"{role} is missing or changed")
    if file_sha256(source) != observed:
        raise ExtremeScoreEvaluatorError(f"{role} changed while being validated")
    return {"path": path_text, "file_sha256": digest}


def _rollback_recommendation(
    value: Any,
    reference_model: Mapping[str, str],
) -> dict[str, Any]:
    raw = _require_exact_keys(value, _ROLLBACK_KEYS, "rollback recommendation")
    if raw["action"] != "retain_reference":
        raise ExtremeScoreEvaluatorError(
            "rollback recommendation action must be retain_reference"
        )
    reference = _model_binding(
        raw["reference_model"], "rollback recommendation reference model"
    )
    if reference != reference_model:
        raise ExtremeScoreEvaluatorError(
            "rollback recommendation must exactly bind the plan reference model"
        )
    reference_artifact = _artifact_binding(
        raw["reference_model_artifact"],
        "rollback recommendation reference model artifact",
        expected_file_sha256=reference["sha256"],
    )
    checkpoint_artifact = _artifact_binding(
        raw["trainer_checkpoint_artifact"],
        "rollback recommendation trainer checkpoint artifact",
    )
    if reference_artifact["path"] == checkpoint_artifact["path"]:
        raise ExtremeScoreEvaluatorError(
            "rollback model and trainer checkpoint artifacts must be distinct"
        )
    if raw["quarantine_candidate_on_failure"] is not True:
        raise ExtremeScoreEvaluatorError(
            "rollback recommendation must quarantine the failed candidate"
        )
    return {
        "action": "retain_reference",
        "reference_model": reference,
        "reference_model_artifact": reference_artifact,
        "trainer_checkpoint_artifact": checkpoint_artifact,
        "quarantine_candidate_on_failure": True,
    }


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    """Load and validate a frozen held-out expected-max policy."""

    policy = _load_json(Path(path), "expected-max policy", canonical=False)
    _require_exact_keys(
        policy,
        {
            "schema_version",
            "contract",
            "policy_version",
            "status",
            "objective",
            "inference",
            "integrity",
        },
        "expected-max policy",
    )
    if (
        policy["schema_version"] != 1
        or policy["contract"] != POLICY_CONTRACT
        or policy["policy_version"] != POLICY_VERSION
        or policy["status"] != "frozen"
    ):
        raise ExtremeScoreEvaluatorError("expected-max policy identity is invalid")

    objective = _require_exact_keys(
        policy["objective"],
        {
            "name",
            "score_perspective",
            "cohort_statistic",
            "estimand",
            "comparison",
            "raw_lifetime_records",
        },
        "policy objective",
    )
    expected_objective = {
        "name": "held_out_expected_max",
        "score_perspective": "focal_model",
        "cohort_statistic": "maximum_finite_terminal_score",
        "estimand": "mean_over_precommitted_fixed_n_cohort_maxima",
        "comparison": "paired_candidate_minus_reference",
        "raw_lifetime_records": "diagnostic_only",
    }
    if dict(objective) != expected_objective:
        raise ExtremeScoreEvaluatorError("policy objective contract is unsupported")

    inference = _require_exact_keys(
        policy["inference"],
        {
            "method",
            "one_sided_alpha",
            "minimum_clusters",
            "required_slices",
            "promotion_rule",
        },
        "policy inference",
    )
    if (
        inference["method"] != "exact_paired_cluster_sign_test"
        or inference["promotion_rule"] != _PROMOTION_RULE
        or inference["required_slices"] != ["overall", "B", "W"]
    ):
        raise ExtremeScoreEvaluatorError("policy inference contract is unsupported")
    alpha = _finite_number(inference["one_sided_alpha"], "one-sided alpha")
    if not 0.0 < alpha < 0.5:
        raise ExtremeScoreEvaluatorError("one-sided alpha must be in (0, 0.5)")
    minimum_clusters = _positive_int(
        inference["minimum_clusters"], "minimum clusters"
    )
    if minimum_clusters < 8:
        raise ExtremeScoreEvaluatorError(
            "exact sign inference requires at least 8 independent clusters"
        )

    integrity = _require_exact_keys(
        policy["integrity"],
        {
            "require_precommitted_plan",
            "require_complete_fixed_n_cohorts",
            "require_candidate_reference_pairing",
            "require_identical_seed_config_opponent_and_cohort_identities",
            "require_alternating_balanced_focal_colors",
            "require_finite_legal_numeric_scores",
            "reject_no_results",
            "reject_turn_limits",
        },
        "policy integrity",
    )
    if any(value is not True for value in integrity.values()):
        raise ExtremeScoreEvaluatorError(
            "policy integrity controls must all remain enabled"
        )
    return policy


def _policy_binding(path: Path, policy: Mapping[str, Any]) -> dict[str, str]:
    source = _regular_file(Path(path), "expected-max policy").resolve()
    return {
        "path": str(source),
        "file_sha256": file_sha256(source),
        "canonical_sha256": canonical_sha256(policy),
        "policy_version": str(policy["policy_version"]),
    }


def _normalize_request(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = _require_exact_keys(request, _REQUEST_KEYS, "plan request")
    if raw["schema_version"] != 1 or raw["contract"] != PLAN_REQUEST_CONTRACT:
        raise ExtremeScoreEvaluatorError("plan request identity is invalid")

    candidate = _model_binding(raw["candidate_model"], "candidate model")
    reference = _model_binding(raw["reference_model"], "reference model")
    if candidate["sha256"] == reference["sha256"]:
        raise ExtremeScoreEvaluatorError(
            "candidate and reference model hashes must differ"
        )
    config = _config_binding(raw["config"])
    cohort_size = _positive_int(raw["cohort_size"], "cohort_size", _MAX_COHORT_SIZE)
    bounds = _require_exact_keys(
        raw["legal_score_bounds"], _BOUNDS_KEYS, "legal score bounds"
    )
    minimum = _finite_number(bounds["minimum"], "legal score minimum")
    maximum = _finite_number(bounds["maximum"], "legal score maximum")
    if minimum >= maximum:
        raise ExtremeScoreEvaluatorError(
            "legal score minimum must be strictly below maximum"
        )

    source_cohorts = raw["cohorts"]
    if not isinstance(source_cohorts, list) or not source_cohorts:
        raise ExtremeScoreEvaluatorError("cohorts must be a nonempty array")
    if len(source_cohorts) % 2 != 0:
        raise ExtremeScoreEvaluatorError(
            "cohorts must contain an even, color-balanced count"
        )

    cohorts: list[dict[str, Any]] = []
    seen_cohort_ids: set[str] = set()
    seen_seeds: set[str] = set()
    clusters: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, item in enumerate(source_cohorts):
        cohort = _require_exact_keys(item, _COHORT_REQUEST_KEYS, f"cohort {index}")
        cohort_id = _text(cohort["cohort_id"], f"cohort {index}.cohort_id")
        cluster_id = _text(cohort["cluster_id"], f"cohort {index}.cluster_id")
        league_cell = _text(cohort["league_cell"], f"cohort {index}.league_cell")
        opponent_snapshot_id = _text(
            cohort["opponent_snapshot_id"],
            f"cohort {index}.opponent_snapshot_id",
        )
        opponent_hash = _sha256(
            cohort["opponent_model_sha256"],
            f"cohort {index}.opponent_model_sha256",
        )
        color = cohort["focal_color"]
        if color not in _COLORS:
            raise ExtremeScoreEvaluatorError(
                f"cohort {index}.focal_color must be B or W"
            )
        seeds = cohort["seeds"]
        if not isinstance(seeds, list) or len(seeds) != cohort_size:
            raise ExtremeScoreEvaluatorError(
                f"cohort {cohort_id!r} must contain exactly {cohort_size} seeds"
            )
        checked_seeds = [
            _text(seed, f"cohort {cohort_id!r} seed {seed_index}")
            for seed_index, seed in enumerate(seeds)
        ]
        if len(set(checked_seeds)) != cohort_size:
            raise ExtremeScoreEvaluatorError(
                f"cohort {cohort_id!r} contains duplicate seeds"
            )
        duplicate_global = sorted(set(checked_seeds).intersection(seen_seeds))
        if duplicate_global:
            raise ExtremeScoreEvaluatorError(
                f"seeds are reused across cohorts: {duplicate_global}"
            )
        if cohort_id in seen_cohort_ids:
            raise ExtremeScoreEvaluatorError(f"duplicate cohort_id {cohort_id!r}")
        seen_cohort_ids.add(cohort_id)
        seen_seeds.update(checked_seeds)
        payload = {
            "cohort_id": cohort_id,
            "cluster_id": cluster_id,
            "league_cell": league_cell,
            "opponent_snapshot_id": opponent_snapshot_id,
            "opponent_model_sha256": opponent_hash,
            "focal_color": color,
            "seeds": checked_seeds,
        }
        finalized = {**payload, "cohort_sha256": canonical_sha256(payload)}
        cohorts.append(finalized)
        clusters[cluster_id].append((index, finalized))

    colors = [cohort["focal_color"] for cohort in cohorts]
    if any(left == right for left, right in pairwise(colors)):
        raise ExtremeScoreEvaluatorError(
            "focal colors must alternate across precommitted cohorts"
        )
    if colors.count("B") != colors.count("W"):
        raise ExtremeScoreEvaluatorError(
            "precommitted cohorts must be exactly balanced by focal color"
        )
    for cluster_id, members in sorted(clusters.items()):
        if len(members) != 2:
            raise ExtremeScoreEvaluatorError(
                f"cluster {cluster_id!r} must contain exactly one B and one W cohort"
            )
        indices = sorted(index for index, _ in members)
        if indices[1] != indices[0] + 1 or indices[0] % 2 != 0:
            raise ExtremeScoreEvaluatorError(
                f"cluster {cluster_id!r} cohorts must be one adjacent color pair"
            )
        pair = [cohort for _, cohort in sorted(members)]
        if {cohort["focal_color"] for cohort in pair} != {"B", "W"}:
            raise ExtremeScoreEvaluatorError(
                f"cluster {cluster_id!r} must contain both focal colors"
            )
        identity_fields = (
            "league_cell",
            "opponent_snapshot_id",
            "opponent_model_sha256",
        )
        if any(pair[0][field] != pair[1][field] for field in identity_fields):
            raise ExtremeScoreEvaluatorError(
                f"cluster {cluster_id!r} changes its frozen opponent or league cell"
            )

    rollback = _rollback_recommendation(raw["rollback_recommendation"], reference)
    return {
        "candidate_model": candidate,
        "reference_model": reference,
        "config": config,
        "cohort_size": cohort_size,
        "legal_score_bounds": {"minimum": minimum, "maximum": maximum},
        "cohorts": cohorts,
        "rollback_recommendation": rollback,
    }


def _matrix_entries(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for arm in _ARMS:
        model_hash = plan[f"{arm}_model"]["sha256"]
        for cohort in plan["cohorts"]:
            for trial_index, seed in enumerate(cohort["seeds"]):
                entries.append(
                    {
                        "arm": arm,
                        "model_sha256": model_hash,
                        "cohort_id": cohort["cohort_id"],
                        "cohort_sha256": cohort["cohort_sha256"],
                        "trial_index": trial_index,
                        "seed": seed,
                        "config_sha256": plan["config"]["sha256"],
                        "league_cell": cohort["league_cell"],
                        "opponent_snapshot_id": cohort["opponent_snapshot_id"],
                        "opponent_model_sha256": cohort["opponent_model_sha256"],
                        "focal_color": cohort["focal_color"],
                    }
                )
    return entries


def build_plan(
    request: Mapping[str, Any],
    *,
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    """Finalize a precommit request into a self-hashed immutable plan."""

    normalized = _normalize_request(request)
    policy = load_policy(policy_path)
    plan: dict[str, Any] = {
        "schema_version": 1,
        "contract": PLAN_CONTRACT,
        "policy": _policy_binding(policy_path, policy),
        **normalized,
    }
    plan["execution_matrix_sha256"] = canonical_sha256(_matrix_entries(plan))
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def _validate_policy_binding(value: Any) -> tuple[dict[str, str], dict[str, Any]]:
    binding = _require_exact_keys(value, _POLICY_BINDING_KEYS, "plan policy binding")
    path_text = _text(binding["path"], "policy path")
    path = Path(path_text)
    if not path.is_absolute() or path != Path(os.path.abspath(path)):
        raise ExtremeScoreEvaluatorError(
            "plan policy path must be absolute and normalized"
        )
    checked = {
        "path": path_text,
        "file_sha256": _sha256(binding["file_sha256"], "policy file SHA-256"),
        "canonical_sha256": _sha256(
            binding["canonical_sha256"], "policy canonical SHA-256"
        ),
        "policy_version": _text(binding["policy_version"], "policy version"),
    }
    policy = load_policy(path)
    if (
        file_sha256(path) != checked["file_sha256"]
        or canonical_sha256(policy) != checked["canonical_sha256"]
        or policy["policy_version"] != checked["policy_version"]
    ):
        raise ExtremeScoreEvaluatorError(
            "expected-max policy changed after plan precommit"
        )
    return checked, policy


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical copy after checking every self-hash and identity."""

    raw = _require_exact_keys(plan, _PLAN_KEYS, "expected-max plan")
    if raw["schema_version"] != 1 or raw["contract"] != PLAN_CONTRACT:
        raise ExtremeScoreEvaluatorError("expected-max plan identity is invalid")
    supplied_hash = _sha256(raw["plan_sha256"], "plan SHA-256")
    payload = dict(raw)
    payload.pop("plan_sha256")
    if supplied_hash != canonical_sha256(payload):
        raise ExtremeScoreEvaluatorError("expected-max plan self-hash is invalid")

    policy_binding, _ = _validate_policy_binding(raw["policy"])
    request_cohorts = []
    for index, cohort in enumerate(
        raw["cohorts"] if isinstance(raw["cohorts"], list) else []
    ):
        checked = _require_exact_keys(cohort, _COHORT_PLAN_KEYS, f"plan cohort {index}")
        payload_cohort = {key: checked[key] for key in _COHORT_REQUEST_KEYS}
        supplied_cohort_hash = _sha256(
            checked["cohort_sha256"], f"plan cohort {index} SHA-256"
        )
        if supplied_cohort_hash != canonical_sha256(payload_cohort):
            raise ExtremeScoreEvaluatorError(
                f"plan cohort {index} self-hash is invalid"
            )
        request_cohorts.append(payload_cohort)
    synthetic_request = {
        "schema_version": 1,
        "contract": PLAN_REQUEST_CONTRACT,
        "candidate_model": raw["candidate_model"],
        "reference_model": raw["reference_model"],
        "config": raw["config"],
        "cohort_size": raw["cohort_size"],
        "legal_score_bounds": raw["legal_score_bounds"],
        "cohorts": request_cohorts,
        "rollback_recommendation": raw["rollback_recommendation"],
    }
    normalized = _normalize_request(synthetic_request)
    for key, expected in normalized.items():
        if raw[key] != expected:
            raise ExtremeScoreEvaluatorError(
                f"expected-max plan {key} is not canonical"
            )
    checked_plan = {
        "schema_version": 1,
        "contract": PLAN_CONTRACT,
        "policy": policy_binding,
        **normalized,
        "execution_matrix_sha256": _sha256(
            raw["execution_matrix_sha256"], "execution matrix SHA-256"
        ),
        "plan_sha256": supplied_hash,
    }
    if checked_plan["execution_matrix_sha256"] != canonical_sha256(
        _matrix_entries(checked_plan)
    ):
        raise ExtremeScoreEvaluatorError(
            "execution matrix hash does not match precommitted cohorts"
        )
    if checked_plan != raw:
        raise ExtremeScoreEvaluatorError("expected-max plan is not canonical")
    return _clone_json(checked_plan, "expected-max plan")


def load_plan(path: Path) -> dict[str, Any]:
    return validate_plan(_load_json(Path(path), "expected-max plan", canonical=True))


def build_runner_jobs(plan: Mapping[str, Any], arm: str) -> tuple[dict[str, Any], ...]:
    """Expand one arm of a plan into immutable runner job identities."""

    checked = validate_plan(plan)
    if arm not in _ARMS:
        raise ExtremeScoreEvaluatorError("runner arm must be candidate or reference")
    model_hash = checked[f"{arm}_model"]["sha256"]
    jobs: list[dict[str, Any]] = []
    for cohort in checked["cohorts"]:
        for trial_index, seed in enumerate(cohort["seeds"]):
            jobs.append(
                {
                    "schema_version": 1,
                    "plan_sha256": checked["plan_sha256"],
                    "arm": arm,
                    "model_sha256": model_hash,
                    "cohort_id": cohort["cohort_id"],
                    "cohort_sha256": cohort["cohort_sha256"],
                    "trial_index": trial_index,
                    "seed": seed,
                    "config_sha256": checked["config"]["sha256"],
                    "league_cell": cohort["league_cell"],
                    "opponent_snapshot_id": cohort["opponent_snapshot_id"],
                    "opponent_model_sha256": cohort["opponent_model_sha256"],
                    "focal_color": cohort["focal_color"],
                }
            )
    return tuple(jobs)


def _issue(
    issues: list[dict[str, Any]],
    code: str,
    *,
    arm: str | None = None,
    row_index: int | None = None,
    cohort_id: str | None = None,
    trial_index: int | None = None,
    detail: str,
) -> None:
    issue: dict[str, Any] = {"code": code, "detail": detail}
    if arm is not None:
        issue["arm"] = arm
    if row_index is not None:
        issue["row_index"] = row_index
    if cohort_id is not None:
        issue["cohort_id"] = cohort_id
    if trial_index is not None:
        issue["trial_index"] = trial_index
    issues.append(issue)


def _semantic_rows_hash(rows: Sequence[Any]) -> str | None:
    try:
        _ensure_finite_json(list(rows), "result rows")
        return canonical_sha256(rows)
    except (ExtremeScoreEvaluatorError, TypeError, ValueError):
        return None


def _focal_legal_score_bounds(
    plan: Mapping[str, Any],
    focal_color: str,
) -> tuple[float, float]:
    white_minimum = float(plan["legal_score_bounds"]["minimum"])
    white_maximum = float(plan["legal_score_bounds"]["maximum"])
    if focal_color == "W":
        return white_minimum, white_maximum
    if focal_color == "B":
        return -white_maximum, -white_minimum
    raise ExtremeScoreEvaluatorError("scheduled focal color must be B or W")


def _validate_result_arm(
    plan: Mapping[str, Any],
    arm: str,
    rows: Sequence[Any],
    issues: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    jobs = build_runner_jobs(plan, arm)
    expected = {(job["cohort_id"], job["trial_index"]): job for job in jobs}
    observed_coordinates: set[tuple[str, int]] = set()
    normalized: dict[tuple[str, int], dict[str, Any]] = {}

    for row_index, value in enumerate(rows):
        if not isinstance(value, Mapping):
            _issue(
                issues,
                "ROW_NOT_OBJECT",
                arm=arm,
                row_index=row_index,
                detail="result row must be an object",
            )
            continue
        actual_keys = set(value)
        if actual_keys != _RESULT_KEYS:
            _issue(
                issues,
                "RESULT_KEYS_MISMATCH",
                arm=arm,
                row_index=row_index,
                detail=(
                    f"missing={sorted(_RESULT_KEYS.difference(actual_keys))}, "
                    f"unexpected={sorted(actual_keys.difference(_RESULT_KEYS))}"
                ),
            )

        cohort_id_value = value.get("cohort_id")
        cohort_id = cohort_id_value if isinstance(cohort_id_value, str) else None
        trial_value = value.get("trial_index")
        trial_index = (
            trial_value if type(trial_value) is int and trial_value >= 0 else None
        )
        if cohort_id is None or trial_index is None:
            _issue(
                issues,
                "INVALID_RESULT_COORDINATE",
                arm=arm,
                row_index=row_index,
                detail="cohort_id and nonnegative integer trial_index are required",
            )
            continue
        coordinate = (cohort_id, trial_index)
        job = expected.get(coordinate)
        if job is None:
            _issue(
                issues,
                "UNPLANNED_RESULT",
                arm=arm,
                row_index=row_index,
                cohort_id=cohort_id,
                trial_index=trial_index,
                detail="result coordinate was not precommitted",
            )
            continue
        if coordinate in observed_coordinates:
            _issue(
                issues,
                "DUPLICATE_RESULT",
                arm=arm,
                row_index=row_index,
                cohort_id=cohort_id,
                trial_index=trial_index,
                detail="result coordinate appears more than once",
            )
            continue
        observed_coordinates.add(coordinate)
        legal_minimum, legal_maximum = _focal_legal_score_bounds(
            plan, str(job["focal_color"])
        )

        valid = actual_keys == _RESULT_KEYS
        for key in sorted(_JOB_KEYS):
            if value.get(key) != job[key]:
                valid = False
                _issue(
                    issues,
                    "IDENTITY_MISMATCH",
                    arm=arm,
                    row_index=row_index,
                    cohort_id=cohort_id,
                    trial_index=trial_index,
                    detail=f"{key} does not match the precommitted runner job",
                )

        score_value = value.get("score")
        try:
            score = _finite_number(score_value, "score")
        except ExtremeScoreEvaluatorError:
            valid = False
            _issue(
                issues,
                "INVALID_SCORE",
                arm=arm,
                row_index=row_index,
                cohort_id=cohort_id,
                trial_index=trial_index,
                detail="score must be finite numeric",
            )
            score = None
        if score is not None and not legal_minimum <= score <= legal_maximum:
            valid = False
            _issue(
                issues,
                "ILLEGAL_SCORE",
                arm=arm,
                row_index=row_index,
                cohort_id=cohort_id,
                trial_index=trial_index,
                detail=(
                    f"{job['focal_color']}-focal score {score} is outside "
                    f"[{legal_minimum}, {legal_maximum}]"
                ),
            )

        no_result = value.get("no_result")
        if not isinstance(no_result, bool):
            valid = False
            _issue(
                issues,
                "INVALID_TERMINAL_FLAG",
                arm=arm,
                row_index=row_index,
                cohort_id=cohort_id,
                trial_index=trial_index,
                detail="no_result must be explicitly boolean",
            )
        elif no_result:
            valid = False
            _issue(
                issues,
                "NO_RESULT",
                arm=arm,
                row_index=row_index,
                cohort_id=cohort_id,
                trial_index=trial_index,
                detail="no-result trials are integrity failures",
            )
        hit_turn_limit = value.get("hit_turn_limit")
        if not isinstance(hit_turn_limit, bool):
            valid = False
            _issue(
                issues,
                "INVALID_TERMINAL_FLAG",
                arm=arm,
                row_index=row_index,
                cohort_id=cohort_id,
                trial_index=trial_index,
                detail="hit_turn_limit must be explicitly boolean",
            )
        elif hit_turn_limit:
            valid = False
            _issue(
                issues,
                "TURN_LIMIT",
                arm=arm,
                row_index=row_index,
                cohort_id=cohort_id,
                trial_index=trial_index,
                detail="turn-limit trials are integrity failures",
            )
        if valid and score is not None:
            normalized[coordinate] = {**job, "score": score}

    for cohort_id, trial_index in sorted(
        set(expected).difference(observed_coordinates)
    ):
        _issue(
            issues,
            "MISSING_RESULT",
            arm=arm,
            cohort_id=cohort_id,
            trial_index=trial_index,
            detail="precommitted trial has no result",
        )
    return normalized


def _sorted_issues(issues: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(issue) for issue in issues),
        key=lambda issue: (
            str(issue.get("code", "")),
            str(issue.get("arm", "")),
            str(issue.get("cohort_id", "")),
            int(issue.get("trial_index", -1)),
            int(issue.get("row_index", -1)),
            str(issue.get("detail", "")),
        ),
    )


def _cohort_maxima(
    plan: Mapping[str, Any],
    rows_by_arm: Mapping[str, Mapping[tuple[str, int], Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    maxima: list[dict[str, Any]] = []
    for cohort in plan["cohorts"]:
        cohort_id = cohort["cohort_id"]
        candidate_scores = [
            float(rows_by_arm["candidate"][(cohort_id, index)]["score"])
            for index in range(plan["cohort_size"])
        ]
        reference_scores = [
            float(rows_by_arm["reference"][(cohort_id, index)]["score"])
            for index in range(plan["cohort_size"])
        ]
        candidate_maximum = max(candidate_scores)
        reference_maximum = max(reference_scores)
        maxima.append(
            {
                "cohort_id": cohort_id,
                "cohort_sha256": cohort["cohort_sha256"],
                "cluster_id": cohort["cluster_id"],
                "league_cell": cohort["league_cell"],
                "opponent_snapshot_id": cohort["opponent_snapshot_id"],
                "opponent_model_sha256": cohort["opponent_model_sha256"],
                "focal_color": cohort["focal_color"],
                "N": plan["cohort_size"],
                "candidate_maximum": candidate_maximum,
                "reference_maximum": reference_maximum,
                "paired_delta": candidate_maximum - reference_maximum,
            }
        )
    return maxima


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ExtremeScoreEvaluatorError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _exact_one_sided_sign_p_value(
    informative_clusters: int,
    positive_clusters: int,
) -> float:
    if informative_clusters <= 0:
        raise ExtremeScoreEvaluatorError(
            "exact sign inference requires informative clusters"
        )
    numerator = sum(
        math.comb(informative_clusters, positives)
        for positives in range(positive_clusters, informative_clusters + 1)
    )
    return float(Fraction(numerator, 1 << informative_clusters))


def _slice_statistics(
    maxima: Sequence[Mapping[str, Any]],
    *,
    slice_name: str,
    alpha: float,
    minimum_clusters: int,
) -> dict[str, Any]:
    selected = [
        row
        for row in maxima
        if slice_name == "overall" or row["focal_color"] == slice_name
    ]
    candidate_values = [float(row["candidate_maximum"]) for row in selected]
    reference_values = [float(row["reference_maximum"]) for row in selected]
    deltas = [float(row["paired_delta"]) for row in selected]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        grouped[str(row["cluster_id"])].append(float(row["paired_delta"]))
    cluster_ids = sorted(grouped)
    cluster_count = len(cluster_ids)
    cluster_effects = [
        {
            "cluster_id": cluster_id,
            "paired_delta": _mean(grouped[cluster_id]),
        }
        for cluster_id in cluster_ids
    ]
    positive_clusters = sum(
        effect["paired_delta"] > 0.0 for effect in cluster_effects
    )
    negative_clusters = sum(
        effect["paired_delta"] < 0.0 for effect in cluster_effects
    )
    zero_clusters = cluster_count - positive_clusters - negative_clusters
    informative_clusters = positive_clusters + negative_clusters
    estimate = _mean(deltas)
    available = (
        cluster_count >= minimum_clusters
        and informative_clusters >= minimum_clusters
    )
    p_value: float | None = None
    if available:
        p_value = _exact_one_sided_sign_p_value(
            informative_clusters, positive_clusters
        )
    return {
        "metric": "J_N",
        "slice": slice_name,
        "N": int(selected[0]["N"]),
        "cohorts": len(selected),
        "clusters": cluster_count,
        "cluster_ids_sha256": canonical_sha256(cluster_ids),
        "candidate_J_N": _mean(candidate_values),
        "reference_J_N": _mean(reference_values),
        "paired_delta_estimate": estimate,
        "inference_available": available,
        "one_sided_p_value": p_value,
        "positive_clusters": positive_clusters,
        "negative_clusters": negative_clusters,
        "zero_clusters": zero_clusters,
        "informative_clusters": informative_clusters,
        "one_sided_alpha": alpha,
        "inference": {
            "method": "exact_paired_cluster_sign_test",
            "inference_unit": "precommitted_black_white_cluster",
            "pairing_unit": "precommitted_fixed_n_cohort",
            "minimum_clusters": minimum_clusters,
            "null_positive_probability": 0.5,
            "zero_difference_handling": "excluded",
            "cluster_effects_sha256": canonical_sha256(cluster_effects),
        },
    }


def _derive_decision(
    integrity_valid: bool,
    statistics: Mapping[str, Any],
) -> tuple[str, list[str]]:
    if not isinstance(integrity_valid, bool):
        raise ExtremeScoreEvaluatorError("report integrity.valid must be boolean")
    if not integrity_valid:
        if statistics:
            raise ExtremeScoreEvaluatorError(
                "integrity-failed reports must not contain statistics"
            )
        return "FAIL", ["INTEGRITY_FAILURE"]
    if not isinstance(statistics, Mapping) or set(statistics) != {
        "overall",
        "B",
        "W",
    }:
        raise ExtremeScoreEvaluatorError(
            "integrity-valid reports require overall, B, and W statistics"
        )

    unavailable: list[str] = []
    nonpositive_estimates: list[str] = []
    nonsignificant: list[str] = []
    for name in ("overall", "B", "W"):
        statistic = statistics[name]
        if not isinstance(statistic, Mapping) or statistic.get("slice") != name:
            raise ExtremeScoreEvaluatorError(
                f"report statistic {name!r} has an invalid slice identity"
            )
        estimate = _finite_number(
            statistic.get("paired_delta_estimate"),
            f"{name} paired delta estimate",
        )
        available = statistic.get("inference_available")
        if not isinstance(available, bool):
            raise ExtremeScoreEvaluatorError(
                f"{name} inference_available must be boolean"
            )
        if estimate <= 0.0:
            nonpositive_estimates.append(name)
        if not available:
            if statistic.get("one_sided_p_value") is not None:
                raise ExtremeScoreEvaluatorError(
                    f"{name} unavailable inference must have a null p-value"
                )
            unavailable.append(name)
            continue
        p_value = _finite_number(
            statistic.get("one_sided_p_value"), f"{name} one-sided p-value"
        )
        alpha = _finite_number(
            statistic.get("one_sided_alpha"), f"{name} one-sided alpha"
        )
        if not 0.0 <= p_value <= 1.0 or not 0.0 < alpha < 0.5:
            raise ExtremeScoreEvaluatorError(
                f"{name} exact sign inference probabilities are invalid"
            )
        if p_value > alpha:
            nonsignificant.append(name)

    if nonpositive_estimates:
        return (
            "FAIL",
            [
                f"NONPOSITIVE_PAIRED_DELTA_{name.upper()}"
                for name in nonpositive_estimates
            ],
        )
    if unavailable:
        return (
            "INCONCLUSIVE",
            [f"INSUFFICIENT_CLUSTERS_{name.upper()}" for name in unavailable],
        )
    if nonsignificant:
        return (
            "INCONCLUSIVE",
            [
                f"EXACT_SIGN_TEST_NOT_SIGNIFICANT_{name.upper()}"
                for name in nonsignificant
            ],
        )
    return "PASS", ["ALL_REQUIRED_EXACT_SIGN_TESTS_SIGNIFICANT"]


def _normalize_lifetime_records(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ExtremeScoreEvaluatorError(
            "raw lifetime records must be an object when supplied"
        )
    return _clone_json(value, "raw lifetime records")


def _runner_source_bindings(runner: Any) -> dict[str, Any]:
    raw = getattr(runner, "source_bindings", None)
    if not isinstance(raw, Mapping):
        return {}
    bindings: dict[str, Any] = {}
    for arm in _ARMS:
        item = raw.get(arm)
        if not isinstance(item, Mapping):
            continue
        if set(item) != {"path", "file_sha256"}:
            continue
        path = item["path"]
        digest = item["file_sha256"]
        if (
            isinstance(path, str)
            and path
            and isinstance(digest, str)
            and _SHA256_RE.fullmatch(digest)
        ):
            bindings[arm] = {"path": path, "file_sha256": digest}
    return bindings


def evaluate_with_runner(
    plan: Mapping[str, Any],
    *,
    runner: Callable[[str, tuple[dict[str, Any], ...]], Iterable[Mapping[str, Any]]],
    plan_binding: Mapping[str, Any] | None = None,
    raw_lifetime_records: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute both frozen arms through an injected runner and derive a report."""

    checked_plan = validate_plan(plan)
    _, policy = _validate_policy_binding(checked_plan["policy"])
    lifetime_records = _normalize_lifetime_records(raw_lifetime_records)
    rows: dict[str, list[Any]] = {}
    runner_issues: list[dict[str, Any]] = []
    for arm in _ARMS:
        jobs = build_runner_jobs(checked_plan, arm)
        try:
            produced = runner(arm, jobs)
            rows[arm] = list(produced)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            rows[arm] = []
            _issue(
                runner_issues,
                "RUNNER_ERROR",
                arm=arm,
                detail=f"{type(exc).__name__}: {exc}",
            )

    issues = list(runner_issues)
    normalized_by_arm = {
        arm: _validate_result_arm(checked_plan, arm, rows[arm], issues) for arm in _ARMS
    }
    issues = _sorted_issues(issues)
    source_bindings = _runner_source_bindings(runner)
    result_bindings = {
        arm: {
            "rows": len(rows[arm]),
            "rows_canonical_sha256": _semantic_rows_hash(rows[arm]),
            "source": source_bindings.get(arm),
        }
        for arm in _ARMS
    }

    if plan_binding is None:
        checked_plan_binding = {
            "path": None,
            "file_sha256": None,
            "plan_sha256": checked_plan["plan_sha256"],
            "execution_matrix_sha256": checked_plan["execution_matrix_sha256"],
        }
    else:
        expected_plan_binding_keys = {
            "path",
            "file_sha256",
            "plan_sha256",
            "execution_matrix_sha256",
        }
        raw_binding = _require_exact_keys(
            plan_binding, expected_plan_binding_keys, "report plan binding"
        )
        checked_plan_binding = dict(raw_binding)
        if (
            checked_plan_binding["plan_sha256"] != checked_plan["plan_sha256"]
            or checked_plan_binding["execution_matrix_sha256"]
            != checked_plan["execution_matrix_sha256"]
        ):
            raise ExtremeScoreEvaluatorError(
                "report plan binding contradicts the evaluated plan"
            )
        _sha256(checked_plan_binding["file_sha256"], "plan file SHA-256")
        _text(checked_plan_binding["path"], "plan path")

    maxima: list[dict[str, Any]] = []
    statistics: dict[str, Any] = {}
    if not issues:
        maxima = _cohort_maxima(checked_plan, normalized_by_arm)
        inference = policy["inference"]
        statistics = {
            slice_name: _slice_statistics(
                maxima,
                slice_name=slice_name,
                alpha=float(inference["one_sided_alpha"]),
                minimum_clusters=int(inference["minimum_clusters"]),
            )
            for slice_name in inference["required_slices"]
        }

    decision, reason_codes = _derive_decision(not issues, statistics)

    rollback = _clone_json(
        checked_plan["rollback_recommendation"], "rollback recommendation"
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "contract": REPORT_CONTRACT,
        "finalized": True,
        "plan_binding": checked_plan_binding,
        "policy_binding": checked_plan["policy"],
        "objective": {
            "name": "held_out_expected_max",
            "formula": "J_N=mean_c(max_i(score[c,i]))",
            "N": checked_plan["cohort_size"],
            "comparison": "paired_candidate_minus_reference",
            "required_slices": ["overall", "B", "W"],
            "promotion_rule": _PROMOTION_RULE,
        },
        "result_bindings": result_bindings,
        "integrity": {
            "valid": not issues,
            "issue_codes": sorted({issue["code"] for issue in issues}),
            "issues": issues,
            "expected_rows_per_arm": (
                len(checked_plan["cohorts"]) * checked_plan["cohort_size"]
            ),
            "complete_fixed_n_cohorts": not issues,
            "candidate_reference_paired": not issues,
            "alternating_balanced_focal_colors": True,
        },
        "cohort_maxima": maxima,
        "statistics": statistics,
        "decision": decision,
        "reason_codes": reason_codes,
        "promotion_recommended": decision == "PASS",
        "rollback_recommendation": rollback,
        "rollback_recommendation_sha256": canonical_sha256(rollback),
        "non_gating_diagnostics": {
            "raw_lifetime_records": lifetime_records,
            "used_for_decision": False,
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def evaluate_expected_max(
    plan: Mapping[str, Any],
    candidate_rows: Iterable[Mapping[str, Any]],
    reference_rows: Iterable[Mapping[str, Any]],
    *,
    raw_lifetime_records: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience API for already-produced in-memory result rows."""

    rows = {
        "candidate": tuple(candidate_rows),
        "reference": tuple(reference_rows),
    }

    def runner(
        arm: str, _jobs: tuple[dict[str, Any], ...]
    ) -> Iterable[Mapping[str, Any]]:
        return rows[arm]

    return evaluate_with_runner(
        plan,
        runner=runner,
        raw_lifetime_records=raw_lifetime_records,
    )


class JsonlResultRunner:
    """A GPU-free CLI runner that reads one finalized JSONL file per arm."""

    def __init__(self, candidate_path: Path, reference_path: Path) -> None:
        self.paths = {
            "candidate": _regular_file(candidate_path, "candidate results"),
            "reference": _regular_file(reference_path, "reference results"),
        }
        self._source_hashes = {
            arm: file_sha256(path) for arm, path in self.paths.items()
        }

    @property
    def source_bindings(self) -> dict[str, dict[str, str]]:
        bindings: dict[str, dict[str, str]] = {}
        for arm, path in self.paths.items():
            if file_sha256(path) != self._source_hashes[arm]:
                raise ExtremeScoreEvaluatorError(
                    f"{arm} results changed during evaluation"
                )
            bindings[arm] = {
                "path": str(path.resolve()),
                "file_sha256": self._source_hashes[arm],
            }
        return bindings

    def __call__(
        self, arm: str, _jobs: tuple[dict[str, Any], ...]
    ) -> tuple[dict[str, Any], ...]:
        if arm not in _ARMS:
            raise ExtremeScoreEvaluatorError(
                "result runner arm must be candidate or reference"
            )
        path = self.paths[arm]
        if file_sha256(path) != self._source_hashes[arm]:
            raise ExtremeScoreEvaluatorError(f"{arm} results changed before evaluation")
        rows = _load_jsonl(path, f"{arm} results")
        if file_sha256(path) != self._source_hashes[arm]:
            raise ExtremeScoreEvaluatorError(f"{arm} results changed while being read")
        return rows


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_immutable_bytes(path: Path, data: bytes, role: str) -> None:
    target = Path(path)
    try:
        metadata = target.lstat()
    except FileNotFoundError as exc:
        raise ExtremeScoreEvaluatorError(f"{role} is missing: {target}") from exc
    if target.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ExtremeScoreEvaluatorError(f"{role} must be a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) & 0o222:
        raise ExtremeScoreEvaluatorError(f"{role} must be read-only")
    observed = target.read_bytes()
    expected_hash = hashlib.sha256(data).hexdigest()
    if observed != data or file_sha256(target) != expected_hash:
        raise ExtremeScoreEvaluatorError(
            f"{role} bytes or SHA-256 contradict immutable content: {target}"
        )


def _publish_immutable_bytes(path: Path, data: bytes) -> None:
    """Atomically create read-only bytes, allowing identical reuse only."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ExtremeScoreEvaluatorError(
            "artifact parent must be a regular non-symlink directory"
        )
    if target.exists() or target.is_symlink():
        _verify_immutable_bytes(target, data, "existing immutable artifact")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            _verify_immutable_bytes(
                target, data, "concurrently published immutable artifact"
            )
        _verify_immutable_bytes(target, data, "published immutable artifact")
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def publish_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically create a canonical read-only artifact."""

    data = (canonical_json(value) + "\n").encode("utf-8")
    _publish_immutable_bytes(path, data)


def plan_file(
    request_path: Path,
    output_path: Path,
    *,
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    request = _load_json(request_path, "expected-max plan request", canonical=False)
    plan = build_plan(request, policy_path=policy_path)
    publish_immutable_json(output_path, plan)
    return plan


def evaluate_plan_file(
    plan_path: Path,
    output_path: Path,
    *,
    runner: Callable[[str, tuple[dict[str, Any], ...]], Iterable[Mapping[str, Any]]],
    raw_lifetime_records: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = _regular_file(plan_path, "expected-max plan")
    source_hash = file_sha256(source)
    plan = load_plan(source)
    if file_sha256(source) != source_hash:
        raise ExtremeScoreEvaluatorError("expected-max plan changed while being loaded")
    captured_rows: dict[str, list[Any]] = {}

    def capturing_runner(
        arm: str, jobs: tuple[dict[str, Any], ...]
    ) -> Iterable[Mapping[str, Any]]:
        produced = list(runner(arm, jobs))
        frozen = _clone_json(produced, f"{arm} runner rows")
        captured_rows[arm] = frozen
        return frozen

    report = evaluate_with_runner(
        plan,
        runner=capturing_runner,
        plan_binding={
            "path": str(source.resolve()),
            "file_sha256": source_hash,
            "plan_sha256": plan["plan_sha256"],
            "execution_matrix_sha256": plan["execution_matrix_sha256"],
        },
        raw_lifetime_records=raw_lifetime_records,
    )
    if file_sha256(source) != source_hash:
        raise ExtremeScoreEvaluatorError("expected-max plan changed during evaluation")
    validate_plan(plan)
    if report["integrity"]["valid"] is not True:
        raise ExtremeScoreIntegrityError(report["integrity"]["issues"])

    output = Path(output_path)
    cohort_order = {
        cohort["cohort_id"]: index for index, cohort in enumerate(plan["cohorts"])
    }
    for arm in _ARMS:
        rows = sorted(
            captured_rows[arm],
            key=lambda row: (
                cohort_order[row["cohort_id"]],
                row["trial_index"],
            ),
        )
        data = b"".join(
            (canonical_json(row) + "\n").encode("utf-8") for row in rows
        )
        digest = hashlib.sha256(data).hexdigest()
        snapshot = (
            output.parent
            / f"{output.name}.{arm}.{digest}.results.jsonl"
        ).resolve()
        _publish_immutable_bytes(snapshot, data)
        report["result_bindings"][arm]["source"] = {
            "path": str(snapshot),
            "file_sha256": digest,
        }
        report["result_bindings"][arm]["rows_canonical_sha256"] = canonical_sha256(
            rows
        )
    report.pop("report_sha256")
    report["report_sha256"] = canonical_sha256(report)
    publish_immutable_json(output_path, report)
    return report


def load_report(path: Path) -> dict[str, Any]:
    report = _load_json(path, "expected-max report", canonical=True)
    _require_exact_keys(report, _REPORT_KEYS, "expected-max report")
    if (
        report["schema_version"] != 1
        or report["contract"] != REPORT_CONTRACT
        or report["finalized"] is not True
    ):
        raise ExtremeScoreEvaluatorError("expected-max report identity is invalid")
    supplied_hash = _sha256(report["report_sha256"], "report SHA-256")
    payload = dict(report)
    payload.pop("report_sha256")
    if supplied_hash != canonical_sha256(payload):
        raise ExtremeScoreEvaluatorError("expected-max report self-hash is invalid")

    plan_binding = _require_exact_keys(
        report["plan_binding"],
        {
            "path",
            "file_sha256",
            "plan_sha256",
            "execution_matrix_sha256",
        },
        "report plan binding",
    )
    _sha256(plan_binding["plan_sha256"], "report plan SHA-256")
    _sha256(
        plan_binding["execution_matrix_sha256"],
        "report execution matrix SHA-256",
    )
    if (plan_binding["path"] is None) != (plan_binding["file_sha256"] is None):
        raise ExtremeScoreEvaluatorError(
            "report plan path and file hash must both be present or absent"
        )
    if plan_binding["path"] is None:
        raise ExtremeScoreEvaluatorError(
            "final reports must bind a reloadable plan artifact"
        )
    plan_path_text = _text(plan_binding["path"], "report plan path")
    plan_path = Path(plan_path_text)
    if not plan_path.is_absolute() or plan_path != Path(os.path.abspath(plan_path)):
        raise ExtremeScoreEvaluatorError(
            "report plan path must be absolute and normalized"
        )
    plan_file_hash = _sha256(
        plan_binding["file_sha256"], "report plan file SHA-256"
    )
    plan_source = _regular_file(plan_path, "report-bound plan")
    if file_sha256(plan_source) != plan_file_hash:
        raise ExtremeScoreEvaluatorError("report-bound plan is missing or changed")
    bound_plan = load_plan(plan_source)
    if file_sha256(plan_source) != plan_file_hash:
        raise ExtremeScoreEvaluatorError("report-bound plan changed while being loaded")
    if (
        bound_plan["plan_sha256"] != plan_binding["plan_sha256"]
        or bound_plan["execution_matrix_sha256"]
        != plan_binding["execution_matrix_sha256"]
    ):
        raise ExtremeScoreEvaluatorError(
            "report plan binding contradicts the hash-bound plan"
        )

    policy_binding, _ = _validate_policy_binding(report["policy_binding"])
    if (
        policy_binding != report["policy_binding"]
        or report["policy_binding"] != bound_plan["policy"]
    ):
        raise ExtremeScoreEvaluatorError(
            "expected-max report policy binding is not canonical"
        )
    objective = _require_exact_keys(
        report["objective"],
        {"name", "formula", "N", "comparison", "required_slices", "promotion_rule"},
        "report objective",
    )
    if (
        objective["name"] != "held_out_expected_max"
        or objective["formula"] != "J_N=mean_c(max_i(score[c,i]))"
        or objective["comparison"] != "paired_candidate_minus_reference"
        or objective["required_slices"] != ["overall", "B", "W"]
        or objective["promotion_rule"] != _PROMOTION_RULE
    ):
        raise ExtremeScoreEvaluatorError("expected-max report objective is invalid")
    _positive_int(objective["N"], "report cohort size", _MAX_COHORT_SIZE)

    result_bindings = _require_exact_keys(
        report["result_bindings"], set(_ARMS), "report result bindings"
    )
    bound_rows: dict[str, tuple[dict[str, Any], ...]] = {}
    bound_sources: dict[str, dict[str, str]] = {}
    for arm in _ARMS:
        binding = _require_exact_keys(
            result_bindings[arm],
            {"rows", "rows_canonical_sha256", "source"},
            f"{arm} result binding",
        )
        if type(binding["rows"]) is not int or binding["rows"] < 0:
            raise ExtremeScoreEvaluatorError(
                f"{arm} result row count must be a nonnegative integer"
            )
        if binding["rows_canonical_sha256"] is not None:
            _sha256(
                binding["rows_canonical_sha256"],
                f"{arm} semantic result SHA-256",
            )
        if binding["source"] is None:
            raise ExtremeScoreEvaluatorError(
                f"final report must bind reloadable {arm} result rows"
            )
        result_source = _require_exact_keys(
            binding["source"],
            {"path", "file_sha256"},
            f"{arm} result source",
        )
        source_path_text = _text(
            result_source["path"], f"{arm} result source path"
        )
        source_path = Path(source_path_text)
        if not source_path.is_absolute() or source_path != Path(
            os.path.abspath(source_path)
        ):
            raise ExtremeScoreEvaluatorError(
                f"{arm} result source path must be absolute and normalized"
            )
        source_hash = _sha256(
            result_source["file_sha256"], f"{arm} result source SHA-256"
        )
        source_file = _regular_file(
            source_path, f"{arm} report-bound results"
        )
        if file_sha256(source_file) != source_hash:
            raise ExtremeScoreEvaluatorError(
                f"{arm} report-bound results are missing or changed"
            )
        source_rows = _load_jsonl(source_file, f"{arm} report-bound results")
        if file_sha256(source_file) != source_hash:
            raise ExtremeScoreEvaluatorError(
                f"{arm} report-bound results changed while being loaded"
            )
        if (
            len(source_rows) != binding["rows"]
            or canonical_sha256(source_rows) != binding["rows_canonical_sha256"]
        ):
            raise ExtremeScoreEvaluatorError(
                f"{arm} report result binding is inconsistent"
            )
        bound_rows[arm] = source_rows
        bound_sources[arm] = {
            "path": source_path_text,
            "file_sha256": source_hash,
        }

    integrity = _require_exact_keys(
        report["integrity"],
        {
            "valid",
            "issue_codes",
            "issues",
            "expected_rows_per_arm",
            "complete_fixed_n_cohorts",
            "candidate_reference_paired",
            "alternating_balanced_focal_colors",
        },
        "report integrity",
    )
    if (
        not isinstance(integrity["valid"], bool)
        or not isinstance(integrity["issues"], list)
        or not isinstance(integrity["issue_codes"], list)
        or any(not isinstance(issue, Mapping) for issue in integrity["issues"])
    ):
        raise ExtremeScoreEvaluatorError(
            "expected-max report integrity fields are malformed"
        )
    derived_issue_codes = sorted(
        {
            issue.get("code")
            for issue in integrity["issues"]
            if isinstance(issue.get("code"), str)
        }
    )
    if integrity["issue_codes"] != derived_issue_codes:
        raise ExtremeScoreEvaluatorError(
            "expected-max report integrity issue codes are inconsistent"
        )
    if integrity["valid"] is (bool(integrity["issues"])):
        raise ExtremeScoreEvaluatorError(
            "expected-max report integrity validity is inconsistent"
        )
    if (
        type(integrity["expected_rows_per_arm"]) is not int
        or integrity["expected_rows_per_arm"] <= 0
        or integrity["complete_fixed_n_cohorts"] is not integrity["valid"]
        or integrity["candidate_reference_paired"] is not integrity["valid"]
        or integrity["alternating_balanced_focal_colors"] is not True
    ):
        raise ExtremeScoreEvaluatorError(
            "expected-max report integrity summary is inconsistent"
        )
    if integrity["valid"] is not True:
        raise ExtremeScoreEvaluatorError(
            "final reports require a complete integrity-valid execution matrix"
        )
    if not isinstance(report["cohort_maxima"], list):
        raise ExtremeScoreEvaluatorError("report cohort_maxima must be an array")
    if integrity["valid"] and not report["cohort_maxima"]:
        raise ExtremeScoreEvaluatorError(
            "integrity-valid report must contain cohort maxima"
        )
    if not integrity["valid"] and report["cohort_maxima"]:
        raise ExtremeScoreEvaluatorError(
            "integrity-failed report must not contain cohort maxima"
        )
    if integrity["valid"]:
        expected_rows = integrity["expected_rows_per_arm"]
        if expected_rows != len(report["cohort_maxima"]) * objective["N"] or any(
            result_bindings[arm]["rows"] != expected_rows
            or result_bindings[arm]["rows_canonical_sha256"] is None
            for arm in _ARMS
        ):
            raise ExtremeScoreEvaluatorError(
                "integrity-valid report result counts or hashes are inconsistent"
            )

    decision, reason_codes = _derive_decision(integrity["valid"], report["statistics"])
    if report["decision"] != decision or report["reason_codes"] != reason_codes:
        raise ExtremeScoreEvaluatorError(
            "expected-max report decision derivation is inconsistent"
        )
    if report["promotion_recommended"] is not (decision == "PASS"):
        raise ExtremeScoreEvaluatorError(
            "expected-max report promotion recommendation is inconsistent"
        )
    diagnostics = _require_exact_keys(
        report["non_gating_diagnostics"],
        {"raw_lifetime_records", "used_for_decision"},
        "report non-gating diagnostics",
    )
    if (
        not isinstance(diagnostics["raw_lifetime_records"], Mapping)
        or diagnostics["used_for_decision"] is not False
    ):
        raise ExtremeScoreEvaluatorError(
            "raw lifetime records must remain non-gating diagnostics"
        )
    if not isinstance(report["rollback_recommendation"], Mapping):
        raise ExtremeScoreEvaluatorError(
            "report rollback recommendation must be an object"
        )
    if report["rollback_recommendation_sha256"] != canonical_sha256(
        report["rollback_recommendation"]
    ):
        raise ExtremeScoreEvaluatorError(
            "expected-max report rollback recommendation hash is invalid"
        )
    if report["rollback_recommendation"] != bound_plan["rollback_recommendation"]:
        raise ExtremeScoreEvaluatorError(
            "expected-max report rollback recommendation contradicts its plan"
        )

    class ReportBoundRunner:
        source_bindings = bound_sources

        def __call__(
            self, arm: str, _jobs: tuple[dict[str, Any], ...]
        ) -> tuple[dict[str, Any], ...]:
            return bound_rows[arm]

    recomputed = evaluate_with_runner(
        bound_plan,
        runner=ReportBoundRunner(),
        plan_binding=dict(plan_binding),
        raw_lifetime_records=diagnostics["raw_lifetime_records"],
    )
    if canonical_json(recomputed) != canonical_json(report):
        raise ExtremeScoreEvaluatorError(
            "expected-max report differs from deterministic recomputation"
        )
    if file_sha256(plan_source) != plan_file_hash:
        raise ExtremeScoreEvaluatorError(
            "report-bound plan changed during report validation"
        )
    for arm in _ARMS:
        source = bound_sources[arm]
        if file_sha256(Path(source["path"])) != source["file_sha256"]:
            raise ExtremeScoreEvaluatorError(
                f"{arm} report-bound results changed during report validation"
            )
    return report


def evaluation_status(
    plan_path: Path,
    *,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Return a self-hashed read-only status projection."""

    plan_source = _regular_file(plan_path, "expected-max plan")
    plan = load_plan(plan_source)
    report_binding: dict[str, Any] | None = None
    decision: str | None = None
    promotion_recommended = False
    state = "PLANNED"
    if report_path is not None:
        report_source = _regular_file(report_path, "expected-max report")
        report = load_report(report_source)
        expected_plan_binding = {
            "path": str(plan_source.resolve()),
            "file_sha256": file_sha256(plan_source),
            "plan_sha256": plan["plan_sha256"],
            "execution_matrix_sha256": plan["execution_matrix_sha256"],
        }
        if (
            report["plan_binding"] != expected_plan_binding
            or report["policy_binding"] != plan["policy"]
            or report["rollback_recommendation"] != plan["rollback_recommendation"]
        ):
            raise ExtremeScoreEvaluatorError(
                "expected-max report is bound to another plan"
            )
        state = "EVALUATED"
        decision = report["decision"]
        promotion_recommended = bool(report["promotion_recommended"])
        report_binding = {
            "path": str(report_source.resolve()),
            "file_sha256": file_sha256(report_source),
            "report_sha256": report["report_sha256"],
        }
    status: dict[str, Any] = {
        "schema_version": 1,
        "contract": STATUS_CONTRACT,
        "state": state,
        "plan_binding": {
            "path": str(plan_source.resolve()),
            "file_sha256": file_sha256(plan_source),
            "plan_sha256": plan["plan_sha256"],
            "execution_matrix_sha256": plan["execution_matrix_sha256"],
        },
        "report_binding": report_binding,
        "decision": decision,
        "promotion_recommended": promotion_recommended,
        "rollback_recommendation": _clone_json(
            plan["rollback_recommendation"], "rollback recommendation"
        ),
    }
    status["status_sha256"] = canonical_sha256(status)
    return status


# Small programmatic seams mirroring the CLI subcommands.
plan = plan_file
evaluate = evaluate_plan_file
status = evaluation_status


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan", help="finalize a precommitted fixed-cohort plan"
    )
    plan_parser.add_argument("--spec", required=True, type=Path)
    plan_parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    plan_parser.add_argument("--output", required=True, type=Path)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate finalized rows against a frozen plan"
    )
    evaluate_parser.add_argument("--plan", required=True, type=Path)
    evaluate_parser.add_argument("--candidate-results", type=Path)
    evaluate_parser.add_argument("--reference-results", type=Path)
    evaluate_parser.add_argument("--raw-lifetime-records", type=Path)
    evaluate_parser.add_argument("--output", required=True, type=Path)

    status_parser = subparsers.add_parser(
        "status", help="inspect a plan or finalized report"
    )
    status_parser.add_argument("--plan", required=True, type=Path)
    status_parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[str, tuple[dict[str, Any], ...]], Iterable[Mapping[str, Any]]]
    | None = None,
) -> int:
    args = parse_args(argv)
    try:
        if args.command == "plan":
            plan = plan_file(args.spec, args.output, policy_path=args.policy)
            result = {
                "output": str(args.output),
                "file_sha256": file_sha256(args.output),
                "plan_sha256": plan["plan_sha256"],
                "execution_matrix_sha256": plan["execution_matrix_sha256"],
            }
        elif args.command == "evaluate":
            active_runner = runner
            if active_runner is None:
                if args.candidate_results is None or args.reference_results is None:
                    raise ExtremeScoreEvaluatorError(
                        "evaluate requires both result files without an injected runner"
                    )
                active_runner = JsonlResultRunner(
                    args.candidate_results, args.reference_results
                )
            elif (
                args.candidate_results is not None or args.reference_results is not None
            ):
                raise ExtremeScoreEvaluatorError(
                    "result files and an injected runner are mutually exclusive"
                )
            lifetime_records = (
                _load_json(
                    args.raw_lifetime_records,
                    "raw lifetime records",
                    canonical=False,
                )
                if args.raw_lifetime_records is not None
                else None
            )
            report = evaluate_plan_file(
                args.plan,
                args.output,
                runner=active_runner,
                raw_lifetime_records=lifetime_records,
            )
            result = {
                "output": str(args.output),
                "file_sha256": file_sha256(args.output),
                "report_sha256": report["report_sha256"],
                "decision": report["decision"],
                "promotion_recommended": report["promotion_recommended"],
            }
        else:
            result = evaluation_status(args.plan, report_path=args.report)
    except (
        ExtremeScoreEvaluatorError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            canonical_json(
                {"error": {"type": type(exc).__name__, "message": str(exc)}}
            ),
            file=sys.stderr,
        )
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
