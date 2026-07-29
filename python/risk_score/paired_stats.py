#!/usr/bin/env python3
"""Pair- and position-aware statistics for checkpoint promotion.

The statistical unit is a deterministic two-game color pair. Repeated pairs
for the same ``positionId`` are averaged before inference, so repetitions do
not silently receive extra weight. Confidence bounds use the intercept-only
CR1 cluster variance (the Bessel ``G/(G-1)`` correction) and a Student-t
critical value with ``G-1`` degrees of freedom. A deterministic Rademacher
wild-position-cluster bootstrap is reported as a sensitivity analysis.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


SCHEMA_VERSION = 1
V1_POLICY_PATH = Path(__file__).with_name("promotion_policy_v1.json")
V2_POLICY_PATH = Path(__file__).with_name("promotion_policy_v2.json")
DEFAULT_POLICY_PATH = V2_POLICY_PATH
RISK_METRICS = (
    "final_20",
    "final_50",
    "lead_40_loss",
    "lead_80_loss",
    "high_confidence_loss",
    "targeted_lead_40_suite_loss",
    "targeted_lead_80_suite_loss",
)


class MatchValidationError(ValueError):
    """Raised when rows cannot form promotion-valid deterministic pairs."""

    def __init__(self, report: Dict[str, Any]):
        self.report = report
        codes = ", ".join(report.get("error_codes", [])) or "UNKNOWN"
        super().__init__("invalid deterministic match rows: " + codes)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def exact_zero_event_upper_bound(alpha: float, independent_clusters: int) -> float:
    """Return the exact one-sided no-event bound without cancellation."""

    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or not 0.0 < float(alpha) < 1.0
    ):
        raise ValueError("alpha must be finite and strictly between zero and one")
    if (
        isinstance(independent_clusters, bool)
        or not isinstance(independent_clusters, int)
        or independent_clusters <= 0
    ):
        raise ValueError("independent_clusters must be a positive integer")
    return -math.expm1(math.log(float(alpha)) / independent_clusters)


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        policy = json.load(handle)
    if not isinstance(policy, dict):
        raise ValueError("promotion policy must be a JSON object")
    return policy


def load_jsonl(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: row must be a JSON object")
                rows.append(value)
    if not rows:
        raise ValueError("no JSONL rows were loaded")
    return rows


def realized_utility(
    outcome: float,
    candidate_margin: Optional[float],
    *,
    win_weight: float,
    score_scale: float,
    score_power: float,
) -> float:
    """Return candidate-perspective terminal utility.

    A missing numeric terminal score contributes a score term of exactly zero;
    its resolved win/draw/loss outcome is retained.
    """

    if outcome not in (-1.0, 0.0, 1.0):
        raise ValueError("outcome must be -1, 0, or 1")
    score_term = 0.0
    if candidate_margin is not None and candidate_margin != 0.0:
        score_term = math.copysign(
            (1.0 + abs(candidate_margin) / score_scale) ** score_power - 1.0,
            candidate_margin,
        )
    value = win_weight * outcome + score_term
    if not math.isfinite(value):
        raise ValueError("realized utility is not finite")
    return value


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nonempty_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _winner_from_final_result(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    if value.startswith("B+"):
        return "B"
    if value.startswith("W+"):
        return "W"
    if value in ("0", "draw", "Draw"):
        return "draw"
    return None


def _row_winner(row: Mapping[str, Any]) -> Tuple[Optional[str], bool]:
    raw_winner = row.get("winner")
    explicit = raw_winner if raw_winner in ("B", "W", "draw") else None
    from_result = _winner_from_final_result(row.get("finalResult"))
    conflict = explicit is not None and from_result is not None and explicit != from_result
    return explicit or from_result, conflict


def _flag(row: Mapping[str, Any], name: str) -> Tuple[bool, bool]:
    if name not in row:
        return False, True
    value = row[name]
    return bool(value) if isinstance(value, bool) else False, isinstance(value, bool)


def _issue(
    issues: List[Dict[str, Any]],
    code: str,
    *,
    index: Optional[int] = None,
    game_id: Optional[str] = None,
    pair_id: Optional[str] = None,
    detail: str,
) -> None:
    issue: Dict[str, Any] = {"code": code, "detail": detail}
    if index is not None:
        issue["row_index"] = index
    if game_id is not None:
        issue["game_id"] = game_id
    if pair_id is not None:
        issue["pair_id"] = pair_id
    issues.append(issue)


def _validate_and_context(
    records: Iterable[Dict[str, Any]],
    candidate_bot: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    if candidate_bot is None or (isinstance(candidate_bot, str) and not candidate_bot.strip()):
        raise ValueError("candidate_bot must be nonempty")

    rows = list(records)
    issues: List[Dict[str, Any]] = []
    if not rows:
        _issue(issues, "NO_MATCH_ROWS", detail="at least one complete color pair is required")
    contexts: List[Dict[str, Any]] = []
    seen_game_ids: Dict[str, int] = {}
    pair_contexts: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    counts: Dict[str, int] = {
        "rows": len(rows),
        "pairs": 0,
        "position_clusters": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "true_no_results": 0,
        "resignations": 0,
        "turn_limits": 0,
        "unresolved_rows": 0,
        "resolved_missing_numeric_scores": 0,
        "duplicate_game_ids": 0,
        "missing_pair_members": 0,
        "duplicate_pair_members": 0,
        "incomplete_pairs": 0,
    }

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            _issue(issues, "ROW_NOT_OBJECT", index=index, detail="row must be an object")
            continue

        schema_version = row.get("schemaVersion")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != SCHEMA_VERSION
        ):
            _issue(
                issues,
                "UNSUPPORTED_SCHEMA_VERSION",
                index=index,
                detail=f"schemaVersion must equal {SCHEMA_VERSION}",
            )
        game_id = _nonempty_string(row.get("gameId"))
        pair_id = _nonempty_string(row.get("pairId"))
        position_id = _nonempty_string(row.get("positionId"))
        schedule_id = _nonempty_string(row.get("scheduleId"))
        seed = _nonempty_string(row.get("seed"))
        if game_id is None:
            _issue(
                issues,
                "MISSING_GAME_ID",
                index=index,
                detail="gameId must be a nonempty string",
            )
        elif game_id in seen_game_ids:
            counts["duplicate_game_ids"] += 1
            _issue(
                issues,
                "DUPLICATE_GAME_ID",
                index=index,
                game_id=game_id,
                detail=f"also present at row {seen_game_ids[game_id]}",
            )
        else:
            seen_game_ids[game_id] = index
        if pair_id is None:
            _issue(
                issues,
                "MISSING_PAIR_ID",
                index=index,
                game_id=game_id,
                detail="pairId must be a nonempty string",
            )
        if position_id is None:
            _issue(
                issues,
                "MISSING_POSITION_ID",
                index=index,
                game_id=game_id,
                pair_id=pair_id,
                detail="positionId must be a nonempty string",
            )
        if schedule_id is None:
            _issue(
                issues,
                "MISSING_SCHEDULE_ID",
                index=index,
                game_id=game_id,
                pair_id=pair_id,
                detail="scheduleId must be a nonempty string",
            )
        if seed is None:
            _issue(
                issues,
                "MISSING_SEED",
                index=index,
                game_id=game_id,
                pair_id=pair_id,
                detail="seed must be a nonempty string",
            )

        black_bot = row.get("blackBot")
        white_bot = row.get("whiteBot")
        if (
            black_bot is None
            or white_bot is None
            or (isinstance(black_bot, str) and not black_bot.strip())
            or (isinstance(white_bot, str) and not white_bot.strip())
        ):
            _issue(
                issues,
                "MISSING_BOT_IDENTITY",
                index=index,
                game_id=game_id,
                pair_id=pair_id,
                detail="blackBot and whiteBot must both be nonempty",
            )
        candidate_is_black = black_bot == candidate_bot
        candidate_is_white = white_bot == candidate_bot
        if candidate_is_black == candidate_is_white:
            _issue(
                issues,
                "TARGET_BOT_NOT_EXACTLY_ONCE",
                index=index,
                game_id=game_id,
                pair_id=pair_id,
                detail="candidate bot must appear exactly once",
            )
        if black_bot == white_bot:
            _issue(
                issues,
                "IDENTICAL_BOTS",
                index=index,
                game_id=game_id,
                pair_id=pair_id,
                detail="blackBot and whiteBot must differ",
            )

        resignation, resignation_valid = _flag(row, "resignation")
        turn_limit, turn_limit_valid = _flag(row, "hitTurnLimit")
        no_result, no_result_valid = _flag(row, "noResult")
        for name, valid in (
            ("resignation", resignation_valid),
            ("hitTurnLimit", turn_limit_valid),
            ("noResult", no_result_valid),
        ):
            if name not in row:
                _issue(
                    issues,
                    "MISSING_TERMINAL_FLAG",
                    index=index,
                    game_id=game_id,
                    pair_id=pair_id,
                    detail=f"{name} must be explicitly present",
                )
            if not valid:
                _issue(
                    issues,
                    "INVALID_BOOLEAN_FLAG",
                    index=index,
                    game_id=game_id,
                    pair_id=pair_id,
                    detail=f"{name} must be boolean when present",
                )

        if resignation:
            counts["resignations"] += 1
            _issue(
                issues,
                "RESIGNATION",
                index=index,
                game_id=game_id,
                pair_id=pair_id,
                detail="resignations are invalid for promotion",
            )
        if turn_limit:
            counts["turn_limits"] += 1
            _issue(
                issues,
                "TURN_LIMIT",
                index=index,
                game_id=game_id,
                pair_id=pair_id,
                detail="turn-limit games are invalid for promotion",
            )

        raw_winner = row.get("winner")
        if raw_winner not in ("B", "W", "draw", None):
            _issue(
                issues,
                "INVALID_WINNER",
                index=index,
                game_id=game_id,
                pair_id=pair_id,
                detail="winner must be B, W, draw, or null",
            )
        winner, winner_conflict = _row_winner(row)
        if winner_conflict:
            _issue(
                issues,
                "WINNER_RESULT_CONFLICT",
                index=index,
                game_id=game_id,
                pair_id=pair_id,
                detail="winner conflicts with finalResult",
            )

        outcome: Optional[float] = None
        if no_result:
            counts["true_no_results"] += 1
            outcome = 0.0
            if winner is not None:
                _issue(
                    issues,
                    "NO_RESULT_WITH_WINNER",
                    index=index,
                    game_id=game_id,
                    pair_id=pair_id,
                    detail="a true no-result must not name a winner",
                )
        elif not turn_limit:
            if winner == "draw":
                outcome = 0.0
                counts["draws"] += 1
            elif winner in ("B", "W") and candidate_is_black != candidate_is_white:
                candidate_color = "B" if candidate_is_black else "W"
                outcome = 1.0 if winner == candidate_color else -1.0
                counts["wins" if outcome > 0.0 else "losses"] += 1
            else:
                counts["unresolved_rows"] += 1
                _issue(
                    issues,
                    "UNRESOLVED_ROW",
                    index=index,
                    game_id=game_id,
                    pair_id=pair_id,
                    detail="row is neither a resolved result nor an explicit no-result",
                )

        raw_score = row.get("finalWhiteMinusBlackScore")
        numeric_score = _finite_number(raw_score)
        if raw_score is not None and numeric_score is None:
            _issue(
                issues,
                "NONFINITE_OR_NONNUMERIC_SCORE",
                index=index,
                game_id=game_id,
                pair_id=pair_id,
                detail="finalWhiteMinusBlackScore must be finite numeric or null",
            )
        if no_result and numeric_score is not None:
            _issue(
                issues,
                "NO_RESULT_WITH_NUMERIC_SCORE",
                index=index,
                game_id=game_id,
                pair_id=pair_id,
                detail="true no-result must not have a numeric terminal score",
            )
        if outcome is not None and not no_result and numeric_score is None and not resignation:
            counts["resolved_missing_numeric_scores"] += 1

        candidate_margin: Optional[float] = None
        if numeric_score is not None and candidate_is_black != candidate_is_white:
            candidate_margin = numeric_score if candidate_is_white else -numeric_score

        context = {
            "index": index,
            "row": row,
            "game_id": game_id,
            "pair_id": pair_id,
            "position_id": position_id,
            "schedule_id": schedule_id,
            "seed": seed,
            "candidate_is_black": candidate_is_black,
            "candidate_is_white": candidate_is_white,
            "candidate_color": "B" if candidate_is_black else "W" if candidate_is_white else None,
            "candidate_bot": candidate_bot,
            "reference_bot": white_bot if candidate_is_black else black_bot if candidate_is_white else None,
            "black_bot": black_bot,
            "white_bot": white_bot,
            "outcome": outcome,
            "candidate_margin": candidate_margin,
            "no_result": no_result,
            "resignation": resignation,
            "turn_limit": turn_limit,
        }
        contexts.append(context)
        if pair_id is not None:
            pair_contexts[pair_id].append(context)

    valid_pair_ids: Set[str] = set()
    for pair_id in sorted(pair_contexts):
        members = pair_contexts[pair_id]
        if len(members) != 2:
            counts["incomplete_pairs"] += 1
            if len(members) < 2:
                missing = 2 - len(members)
                counts["missing_pair_members"] += missing
                _issue(
                    issues,
                    "MISSING_PAIR_MEMBER",
                    pair_id=pair_id,
                    detail=f"pair has {len(members)} row(s), expected 2",
                )
            else:
                duplicate = len(members) - 2
                counts["duplicate_pair_members"] += duplicate
                _issue(
                    issues,
                    "DUPLICATE_PAIR_MEMBER",
                    pair_id=pair_id,
                    detail=f"pair has {len(members)} rows, expected 2",
                )
            continue

        first, second = members
        if first["schedule_id"] != second["schedule_id"]:
            _issue(
                issues,
                "PAIR_SCHEDULE_MISMATCH",
                pair_id=pair_id,
                detail="pair members have different scheduleId values",
            )
        if first["position_id"] != second["position_id"]:
            _issue(
                issues,
                "PAIR_POSITION_MISMATCH",
                pair_id=pair_id,
                detail="pair members have different positionId values",
            )
        orientations = sorted(
            member["candidate_color"] for member in members if member["candidate_color"] is not None
        )
        if orientations != ["B", "W"]:
            counts["duplicate_pair_members"] += 1
            _issue(
                issues,
                "DUPLICATE_PAIR_MEMBER",
                pair_id=pair_id,
                detail="pair must contain one candidate-black and one candidate-white game",
            )
        if not (
            first["black_bot"] == second["white_bot"]
            and first["white_bot"] == second["black_bot"]
        ):
            _issue(
                issues,
                "PAIR_NOT_COLOR_REVERSED",
                pair_id=pair_id,
                detail="blackBot/whiteBot assignments are not exact reversals",
            )
        if first["reference_bot"] != second["reference_bot"]:
            _issue(
                issues,
                "PAIR_REFERENCE_MISMATCH",
                pair_id=pair_id,
                detail="pair members use different reference bots",
            )
        if first["seed"] is not None and first["seed"] == second["seed"]:
            _issue(
                issues,
                "PAIR_SEED_COLLISION",
                pair_id=pair_id,
                detail="color-reversed pair members must use distinct seeds",
            )
        valid_pair_ids.add(pair_id)

    counts["pairs"] = len(pair_contexts)
    counts["position_clusters"] = len(
        {
            context["position_id"]
            for context in contexts
            if context["position_id"] is not None
        }
    )
    counts["missing_games"] = counts["missing_pair_members"]
    counts["structural_errors"] = sum(
        1
        for issue in issues
        if issue["code"]
        in {
            "DUPLICATE_GAME_ID",
            "DUPLICATE_PAIR_MEMBER",
            "IDENTICAL_BOTS",
            "INVALID_WINNER",
            "MISSING_BOT_IDENTITY",
            "MISSING_GAME_ID",
            "MISSING_PAIR_ID",
            "MISSING_PAIR_MEMBER",
            "MISSING_POSITION_ID",
            "MISSING_SCHEDULE_ID",
            "MISSING_SEED",
            "MISSING_TERMINAL_FLAG",
            "NO_MATCH_ROWS",
            "PAIR_NOT_COLOR_REVERSED",
            "PAIR_POSITION_MISMATCH",
            "PAIR_REFERENCE_MISMATCH",
            "PAIR_SCHEDULE_MISMATCH",
            "PAIR_SEED_COLLISION",
            "ROW_NOT_OBJECT",
            "TARGET_BOT_NOT_EXACTLY_ONCE",
            "UNSUPPORTED_SCHEMA_VERSION",
        }
    )
    issues.sort(
        key=lambda issue: (
            issue["code"],
            str(issue.get("pair_id", "")),
            str(issue.get("game_id", "")),
            issue.get("row_index", -1),
            issue["detail"],
        )
    )
    error_codes = sorted({issue["code"] for issue in issues})
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "promotion_valid": not issues,
        "error_codes": error_codes,
        "errors": issues,
        "counts": dict(sorted(counts.items())),
    }
    report.update(counts)
    return rows, report, contexts


def validate_match_rows(
    records: Iterable[Dict[str, Any]],
    *,
    candidate_bot: Any = None,
    target_bot: Any = None,
) -> Dict[str, Any]:
    """Return a deterministic validation report without raising for bad rows."""

    if candidate_bot is None:
        candidate_bot = target_bot
    elif target_bot is not None and target_bot != candidate_bot:
        raise ValueError("candidate_bot and target_bot aliases disagree")
    _, report, _ = _validate_and_context(records, candidate_bot)
    return report


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 300
    epsilon = 3.0e-14
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        twice = 2 * iteration
        numerator = iteration * (b - iteration) * x / ((qam + twice) * (a + twice))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        numerator = -(a + iteration) * (qab + iteration) * x / (
            (a + twice) * (qap + twice)
        )
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(
        degrees_of_freedom / 2.0,
        0.5,
        x,
    )
    return 1.0 - tail if value > 0.0 else tail


@lru_cache(maxsize=128)
def _student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    if not 0.5 < probability < 1.0:
        raise ValueError("only upper-half Student-t quantiles are supported")
    low = 0.0
    high = 1.0
    while _student_t_cdf(high, degrees_of_freedom) < probability:
        high *= 2.0
        if high > 1.0e8:
            raise ArithmeticError("could not bracket Student-t quantile")
    for _ in range(90):
        middle = (low + high) / 2.0
        if _student_t_cdf(middle, degrees_of_freedom) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _quantile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = probability * (len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _normalise_label(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def _frozen_schedule_metadata(row: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    """Return suite identity only from runner-copied immutable schedule metadata."""

    suite = _normalise_label(row.get("suite"))
    suite_bank = _normalise_label(row.get("suiteBank"))
    schedule_id = row.get("scheduleId")
    suite_hash = row.get("suiteBankSha256")
    content_hash = row.get("positionContentSha256")
    semantic_hash = row.get("positionSemanticSha256")
    if not (
        suite is not None
        and suite == suite_bank
        and isinstance(schedule_id, str)
        and schedule_id == row.get("scheduleId")
        and isinstance(suite_hash, str)
        and len(suite_hash) == 64
        and all(character in "0123456789abcdef" for character in suite_hash)
        and isinstance(content_hash, str)
        and len(content_hash) == 64
        and all(character in "0123456789abcdef" for character in content_hash)
        and isinstance(semantic_hash, str)
        and len(semantic_hash) == 64
        and all(character in "0123456789abcdef" for character in semantic_hash)
    ):
        return None
    return {
        "suite": suite,
        "schedule_id": schedule_id,
        "suite_hash": suite_hash,
    }


def _suite_labels(row: Mapping[str, Any]) -> Set[str]:
    metadata = _frozen_schedule_metadata(row)
    if metadata is None:
        return set()
    return {metadata["suite"]}


def _risk_aliases(metric: str) -> Tuple[str, ...]:
    camel = {
        "final_20": "final20",
        "final_50": "final50",
        "lead_40_loss": "lead40Loss",
        "lead_80_loss": "lead80Loss",
        "high_confidence_loss": "highConfidenceLoss",
        "targeted_lead_40_suite_loss": "targetedLead40SuiteLoss",
        "targeted_lead_80_suite_loss": "targetedLead80SuiteLoss",
    }[metric]
    return metric, metric.replace("_", "-"), camel


def _mapping_flag(mapping: Any, aliases: Sequence[str]) -> Optional[bool]:
    if not isinstance(mapping, dict):
        return None
    for alias in aliases:
        if alias in mapping:
            value = mapping[alias]
            if not isinstance(value, bool):
                raise ValueError(f"risk flag {alias!r} must be boolean")
            return value
    return None


def _explicit_risk_flag(
    row: Mapping[str, Any],
    metric: str,
    role: str,
) -> Optional[bool]:
    aliases = _risk_aliases(metric)
    role_names = {
        "candidate": ("candidate", "target"),
        "reference": ("reference", "opponent"),
    }[role]
    sources: List[Any] = [row, row.get("metadata"), row.get("riskFlags")]
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        sources.append(metadata.get("riskFlags"))

    for source in sources:
        if not isinstance(source, dict):
            continue
        for role_name in role_names:
            value = _mapping_flag(source.get(role_name), aliases)
            if value is not None:
                return value
        titled = "candidateRiskFlags" if role == "candidate" else "referenceRiskFlags"
        value = _mapping_flag(source.get(titled), aliases)
        if value is not None:
            return value
        prefix = "candidate" if role == "candidate" else "reference"
        for alias in aliases:
            combined = prefix + alias[:1].upper() + alias[1:]
            if combined in source:
                value = source[combined]
                if not isinstance(value, bool):
                    raise ValueError(f"risk flag {combined!r} must be boolean")
                return value
    return None


def _aggregate_values(
    contexts: Sequence[Dict[str, Any]],
    values_by_game: Mapping[str, float],
    *,
    allowed_pair_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    pairs: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for context in contexts:
        pair_id = context["pair_id"]
        if pair_id is not None:
            pairs[pair_id].append(context)

    pair_rows: List[Dict[str, Any]] = []
    missing_games = 0
    for pair_id in sorted(pairs):
        if allowed_pair_ids is not None and pair_id not in allowed_pair_ids:
            continue
        members = pairs[pair_id]
        values: List[float] = []
        for member in members:
            game_id = member["game_id"]
            if game_id is None or game_id not in values_by_game:
                missing_games += 1
                continue
            values.append(float(values_by_game[game_id]))
        if len(values) != 2:
            continue
        labels: Set[str] = set()
        for member in members:
            labels.update(_suite_labels(member["row"]))
        if len(labels) > 1:
            raise ValueError(f"pair {pair_id!r} has conflicting frozen suite metadata")
        pair_rows.append(
            {
                "pair_id": pair_id,
                "position_id": members[0]["position_id"],
                "schedule_id": members[0]["schedule_id"],
                "suite": ",".join(sorted(labels)) or "ordinary",
                "value": statistics.fmean(values),
            }
        )

    by_position: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_position[row["position_id"]].append(row)
    position_rows: List[Dict[str, Any]] = []
    for position_id in sorted(by_position):
        rows = by_position[position_id]
        schedule_ids = sorted({row["schedule_id"] for row in rows})
        suites = sorted({row["suite"] for row in rows})
        position_rows.append(
            {
                "position_id": position_id,
                "pair_count": len(rows),
                "schedule_ids": schedule_ids,
                "stratum": {
                    "schedule_id": schedule_ids[0]
                    if len(schedule_ids) == 1
                    else schedule_ids,
                    "suite": suites[0] if len(suites) == 1 else suites,
                },
                "value": statistics.fmean(row["value"] for row in rows),
            }
        )
    return {
        "pair_values": pair_rows,
        "position_values": position_rows,
        "missing_game_values": missing_games,
    }


def _bootstrap_bounds(
    position_rows: Sequence[Dict[str, Any]],
    *,
    alpha: float,
    replications: int,
    seed: int,
) -> Dict[str, Any]:
    values = [float(row["value"]) for row in position_rows]
    estimate = statistics.fmean(values)
    rng = random.Random(seed)
    replicates: List[float] = []
    grouped_rows: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in position_rows:
        grouped_rows[canonical_json(row["stratum"])].append(row)
    strata: List[Dict[str, Any]] = []
    centered_groups: List[List[Tuple[str, float]]] = []
    for stratum_key in sorted(grouped_rows):
        rows = sorted(grouped_rows[stratum_key], key=lambda row: row["position_id"])
        stratum_mean = statistics.fmean(float(row["value"]) for row in rows)
        centered_groups.append(
            [
                (row["position_id"], float(row["value"]) - stratum_mean)
                for row in rows
            ]
        )
        strata.append(
            {
                "dimensions": json.loads(stratum_key),
                "position_clusters": len(rows),
            }
        )
    for _ in range(replications):
        perturbation = sum(
            residual * (1.0 if rng.getrandbits(1) else -1.0)
            for group in centered_groups
            for _, residual in group
        ) / len(position_rows)
        replicates.append(estimate + perturbation)
    return {
        "method": (
            "stratified_wild_position_cluster_rademacher_"
            "centered_within_declared_strata"
        ),
        "replications": replications,
        "seed": seed,
        "stratum_dimensions": ["schedule_id", "suite"],
        "strata": strata,
        "stratum_cluster_counts": [
            {
                **entry["dimensions"],
                "position_clusters": entry["position_clusters"],
            }
            for entry in strata
        ],
        "lower_bound": _quantile(replicates, alpha),
        "upper_bound": _quantile(replicates, 1.0 - alpha),
    }


def _metric_report(
    aggregation: Dict[str, Any],
    *,
    metric_name: str,
    alpha: float,
    nominal_confidence: float,
    bootstrap_replications: int,
    bootstrap_seed: int,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
    complete: bool = True,
    candidate_events: Optional[int] = None,
    reference_events: Optional[int] = None,
) -> Dict[str, Any]:
    pair_rows = aggregation["pair_values"]
    position_rows = aggregation["position_values"]
    values = [float(row["value"]) for row in position_rows]
    clusters = len(values)
    estimate = statistics.fmean(values) if values else None
    standard_error: Optional[float] = None
    critical_value: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    degrees_of_freedom: Optional[int] = None
    small_cluster_factor: Optional[float] = None
    bootstrap: Optional[Dict[str, Any]] = None

    if clusters >= 2 and estimate is not None:
        degrees_of_freedom = clusters - 1
        small_cluster_factor = clusters / (clusters - 1.0)
        variance = sum((value - estimate) ** 2 for value in values) / degrees_of_freedom
        standard_error = math.sqrt(max(0.0, variance) / clusters)
        critical_value = _student_t_quantile(1.0 - alpha, degrees_of_freedom)
        lower_bound = estimate - critical_value * standard_error
        upper_bound = estimate + critical_value * standard_error
        derived_seed = int.from_bytes(
            hashlib.sha256(f"{bootstrap_seed}:{metric_name}".encode("utf-8")).digest()[:8],
            "big",
        )
        bootstrap = _bootstrap_bounds(
            position_rows,
            alpha=alpha,
            replications=bootstrap_replications,
            seed=derived_seed,
        )

    zero_event_bound: Optional[float] = None
    if (
        candidate_events == 0
        and reference_events == 0
        and clusters > 0
        and estimate is not None
    ):
        zero_event_bound = exact_zero_event_upper_bound(alpha, clusters)
        if upper_bound is None or upper_bound < zero_event_bound:
            upper_bound = zero_event_bound
        if bootstrap is not None and (
            bootstrap["upper_bound"] is None
            or bootstrap["upper_bound"] < zero_event_bound
        ):
            bootstrap["upper_bound"] = zero_event_bound
            bootstrap["zero_event_upper_bound_applied"] = True

    def clamp(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if lower_limit is not None:
            value = max(lower_limit, value)
        if upper_limit is not None:
            value = min(upper_limit, value)
        return value

    lower_bound = clamp(lower_bound)
    upper_bound = clamp(upper_bound)
    if bootstrap is not None:
        bootstrap["lower_bound"] = clamp(bootstrap["lower_bound"])
        bootstrap["upper_bound"] = clamp(bootstrap["upper_bound"])

    result: Dict[str, Any] = {
        "metric": metric_name,
        "available": estimate is not None and clusters >= 2,
        "complete": complete and aggregation["missing_game_values"] == 0,
        "estimate": estimate,
        "standard_error": standard_error,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "one_sided_alpha": alpha,
        "allocated_one_sided_confidence": 1.0 - alpha,
        "nominal_one_sided_confidence": nominal_confidence,
        "position_clusters": clusters,
        "color_pairs": len(pair_rows),
        "degrees_of_freedom": degrees_of_freedom,
        "small_cluster_correction": {
            "method": "CR1_BESSEL_WITH_STUDENT_T",
            "variance_multiplier": small_cluster_factor,
            "critical_value": critical_value,
        },
        "bootstrap": bootstrap,
        "pair_values": pair_rows,
        "position_values": position_rows,
        "missing_game_values": aggregation["missing_game_values"],
    }
    if candidate_events is not None:
        result["candidate_events"] = candidate_events
    if reference_events is not None:
        result["reference_events"] = reference_events
    if candidate_events is not None and reference_events is not None:
        result["direction"] = "candidate_minus_reference"
        result["matched_within_game"] = True
    if zero_event_bound is not None:
        result["zero_event_uncertainty_upper_bound"] = zero_event_bound
        result["zero_event_independent_position_clusters"] = clusters
        result["zero_event_uncertainty_method"] = (
            "one_sided_exact_no_event_bound_using_independent_position_clusters"
        )
    return result


def _look_settings(policy: Mapping[str, Any], look_number: int) -> Dict[str, Any]:
    try:
        confidence = policy["confidence"]
        sequential = confidence["sequential_testing"]
        looks = sequential["looks"]
    except (KeyError, TypeError) as exc:
        raise ValueError("policy is missing confidence sequential-testing settings") from exc
    if not isinstance(looks, list):
        raise ValueError("policy confidence sequential looks must be a list")
    for look in looks:
        if isinstance(look, dict) and look.get("look_number") == look_number:
            return {
                "look_count": sequential.get("look_count"),
                "routine_alpha": float(look["routine_one_sided_alpha"]),
                "catastrophe_alpha": float(look["catastrophe_one_sided_alpha"]),
                "routine_nominal_confidence": float(
                    confidence["routine"]["nominal_one_sided_confidence"]
                ),
                "catastrophe_nominal_confidence": float(
                    confidence["catastrophe"]["nominal_one_sided_confidence"]
                ),
                "allocation_method": sequential.get("allocation_method"),
                "data_dependent_thresholds_allowed": sequential.get(
                    "data_dependent_thresholds_allowed"
                ),
            }
    raise ValueError(f"look_number {look_number!r} is not defined by policy")


def _policy_objective(
    policy: Mapping[str, Any],
) -> Tuple[float, float, float, Dict[str, float]]:
    try:
        objective = policy["objective"]
        win_weight = float(objective["win_weight"])
        score_scale = float(objective["score_scale"])
        score_power = float(objective["score_power"])
        win_scores = {
            name: float(objective["ordinary_win_score"][name])
            for name in ("win", "draw", "loss", "true_no_result")
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("policy objective is incomplete") from exc
    if not all(
        math.isfinite(value)
        for value in (win_weight, score_scale, score_power, *win_scores.values())
    ):
        raise ValueError("policy objective values must be finite")
    if score_scale <= 0.0 or score_power <= 0.0:
        raise ValueError("score scale and power must be positive")
    if any(not 0.0 <= value <= 1.0 for value in win_scores.values()):
        raise ValueError("ordinary win scores must be between zero and one")
    return win_weight, score_scale, score_power, win_scores


def _trace_state(
    move_records: Optional[Iterable[Dict[str, Any]]],
    contexts: Sequence[Dict[str, Any]],
    *,
    high_confidence_threshold: float,
    lead_40_threshold: float,
    lead_80_threshold: float,
) -> Tuple[Dict[str, Dict[Any, Dict[str, bool]]], Dict[str, Any]]:
    known_games = {context["game_id"] for context in contexts}
    traces: Dict[str, Dict[Any, Dict[str, bool]]] = defaultdict(dict)
    supplied = move_records is not None
    rows = list(move_records) if move_records is not None else []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"move trace row {index} must be an object")
        game_id = _nonempty_string(row.get("gameId"))
        if game_id is None:
            raise ValueError(f"move trace row {index} has no nonempty gameId")
        if game_id not in known_games:
            raise ValueError(f"move trace row {index} references unknown gameId {game_id!r}")
        bot = _nonempty_string(row.get("bot"))
        if bot is None:
            raise ValueError(f"move trace row {index} has no nonempty bot")
        state = traces.setdefault(game_id, {}).setdefault(
            bot,
            {"lead_40": False, "lead_80": False, "high_confidence": False},
        )
        score_lead = _finite_number(row.get("scoreLead"))
        win_probability = _finite_number(row.get("winProbability"))
        if row.get("scoreLead") is not None and score_lead is None:
            raise ValueError(f"move trace row {index} has invalid scoreLead")
        if row.get("winProbability") is not None and win_probability is None:
            raise ValueError(f"move trace row {index} has invalid winProbability")
        if win_probability is not None and not 0.0 <= win_probability <= 1.0:
            raise ValueError(f"move trace row {index} has out-of-range winProbability")
        if score_lead is not None:
            state["lead_40"] = state["lead_40"] or score_lead >= lead_40_threshold
            state["lead_80"] = state["lead_80"] or score_lead >= lead_80_threshold
        if win_probability is not None:
            state["high_confidence"] = (
                state["high_confidence"] or win_probability >= high_confidence_threshold
            )

    both_bot_coverage = 0
    for context in contexts:
        game_id = context["game_id"]
        game_traces = traces.get(game_id, {})
        if (
            context["candidate_bot"] in game_traces
            and context["reference_bot"] in game_traces
        ):
            both_bot_coverage += 1
    return traces, {
        "supplied": supplied,
        "rows": len(rows),
        "games_with_both_bot_traces": both_bot_coverage,
        "total_games": len(contexts),
        "complete": supplied and both_bot_coverage == len(contexts),
    }


def compute_paired_statistics(
    records: Iterable[Dict[str, Any]],
    *,
    candidate_bot: Any = None,
    target_bot: Any = None,
    move_records: Optional[Iterable[Dict[str, Any]]] = None,
    policy: Optional[Dict[str, Any]] = None,
    look_number: int = 1,
    bootstrap_replications: Optional[int] = None,
    bootstrap_seed: Optional[int] = None,
    data_binding: Optional[Dict[str, Any]] = None,
    finalized: bool = False,
) -> Dict[str, Any]:
    """Validate rows and compute pair/position-clustered promotion statistics."""

    if candidate_bot is None:
        candidate_bot = target_bot
    elif target_bot is not None and target_bot != candidate_bot:
        raise ValueError("candidate_bot and target_bot aliases disagree")
    active_policy = load_policy() if policy is None else policy
    if not isinstance(active_policy, dict):
        raise ValueError("policy must be a dictionary")
    rows, validation, contexts = _validate_and_context(records, candidate_bot)
    if not validation["promotion_valid"]:
        raise MatchValidationError(validation)

    settings = _look_settings(active_policy, look_number)
    win_weight, score_scale, score_power, win_scores = _policy_objective(active_policy)
    bootstrap_config = active_policy.get("bootstrap", {})
    expected_bootstrap = {
        "method": (
            "stratified_wild_position_cluster_rademacher_"
            "centered_within_declared_strata"
        ),
        "weights": "rademacher",
        "strata": ["schedule_id", "suite"],
        "centering": "within_each_declared_stratum",
        "report_as_sensitivity": True,
        "zero_event_upper_bound": (
            "one_sided_exact_no_event_bound_using_independent_position_clusters"
        ),
    }
    if any(
        bootstrap_config.get(name) != expected
        for name, expected in expected_bootstrap.items()
    ):
        raise ValueError("policy bootstrap contract is unsupported or incomplete")
    replications = (
        int(bootstrap_config.get("replications", 9999))
        if bootstrap_replications is None
        else int(bootstrap_replications)
    )
    seed = (
        int(bootstrap_config.get("seed", 20260728))
        if bootstrap_seed is None
        else int(bootstrap_seed)
    )
    if replications <= 0:
        raise ValueError("bootstrap_replications must be positive")
    if not isinstance(finalized, bool):
        raise ValueError("finalized must be boolean")
    if data_binding is not None and not isinstance(data_binding, dict):
        raise ValueError("data_binding must be an object when supplied")
    if finalized and data_binding is None:
        raise ValueError("finalized statistics require immutable data_binding")
    if finalized and data_binding is not None:
        for name in (
            "candidate_hash",
            "reference_hash",
            "suite_hash",
            "schedule_hash",
            "config_hash",
            "runner_manifest_hash",
            "execution_hash",
            "katago_binary_hash",
        ):
            value = data_binding.get(name)
            if not (
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"finalized data_binding {name} must be SHA-256")
        for name in ("comparison", "suite", "schedule_id"):
            if not _nonempty_string(data_binding.get(name)):
                raise ValueError(f"finalized data_binding {name} must be nonempty")

    utility_by_game: Dict[str, float] = {}
    win_score_by_game: Dict[str, float] = {}
    for context in contexts:
        outcome = context["outcome"]
        if outcome is None:
            raise AssertionError("validated context unexpectedly lacks an outcome")
        utility_by_game[context["game_id"]] = realized_utility(
            outcome,
            context["candidate_margin"],
            win_weight=win_weight,
            score_scale=score_scale,
            score_power=score_power,
        )
        win_score_key = (
            "true_no_result"
            if context["no_result"]
            else "win"
            if outcome > 0.0
            else "loss"
            if outcome < 0.0
            else "draw"
        )
        win_score_by_game[context["game_id"]] = win_scores[win_score_key]

    utility_metric = _metric_report(
        _aggregate_values(contexts, utility_by_game),
        metric_name="realized_utility",
        alpha=settings["routine_alpha"],
        nominal_confidence=settings["routine_nominal_confidence"],
        bootstrap_replications=replications,
        bootstrap_seed=seed,
    )
    win_rate_metric = _metric_report(
        _aggregate_values(contexts, win_score_by_game),
        metric_name="win_rate",
        alpha=settings["routine_alpha"],
        nominal_confidence=settings["routine_nominal_confidence"],
        bootstrap_replications=replications,
        bootstrap_seed=seed,
        lower_limit=0.0,
        upper_limit=1.0,
    )

    threshold_config = active_policy.get("promotion_thresholds", {})
    high_confidence_threshold = float(
        threshold_config.get("high_confidence_win_probability", 0.95)
    )
    final_margin_thresholds = threshold_config.get(
        "final_margin_threshold_points",
        [20.0, 50.0],
    )
    lead_thresholds = threshold_config.get("lead_threshold_points", [40.0, 80.0])
    if (
        not isinstance(final_margin_thresholds, list)
        or len(final_margin_thresholds) != 2
        or not isinstance(lead_thresholds, list)
        or len(lead_thresholds) != 2
    ):
        raise ValueError("policy must define two final-margin and two lead thresholds")
    final_20_threshold, final_50_threshold = (
        float(final_margin_thresholds[0]),
        float(final_margin_thresholds[1]),
    )
    lead_40_threshold, lead_80_threshold = (
        float(lead_thresholds[0]),
        float(lead_thresholds[1]),
    )
    if not (
        all(
            math.isfinite(value)
            for value in (
                final_20_threshold,
                final_50_threshold,
                lead_40_threshold,
                lead_80_threshold,
                high_confidence_threshold,
            )
        )
        and
        0.0 < final_20_threshold < final_50_threshold
        and 0.0 < lead_40_threshold < lead_80_threshold
        and 0.0 < high_confidence_threshold <= 1.0
    ):
        raise ValueError("policy risk thresholds must be ordered and positive")
    traces, trace_coverage = _trace_state(
        move_records,
        contexts,
        high_confidence_threshold=high_confidence_threshold,
        lead_40_threshold=lead_40_threshold,
        lead_80_threshold=lead_80_threshold,
    )

    risk_values: Dict[str, Dict[str, float]] = {
        name: {} for name in RISK_METRICS
    }
    candidate_event_counts = {name: 0 for name in RISK_METRICS}
    reference_event_counts = {name: 0 for name in RISK_METRICS}
    classification_missing = {name: 0 for name in RISK_METRICS}
    targeted_pairs: Dict[str, Set[str]] = {
        "targeted_lead_40_suite_loss": set(),
        "targeted_lead_80_suite_loss": set(),
    }

    for context in contexts:
        row = context["row"]
        game_id = context["game_id"]
        outcome = context["outcome"]
        margin = context["candidate_margin"]

        for metric, threshold in (
            ("final_20", final_20_threshold),
            ("final_50", final_50_threshold),
        ):
            explicit_candidate = _explicit_risk_flag(row, metric, "candidate")
            explicit_reference = _explicit_risk_flag(row, metric, "reference")
            derived_candidate: Optional[bool] = None
            derived_reference: Optional[bool] = None
            if margin is not None:
                derived_candidate = margin <= -threshold
                derived_reference = margin >= threshold
            elif context["no_result"]:
                derived_candidate = False
                derived_reference = False
            if (
                explicit_candidate is not None
                and derived_candidate is not None
                and explicit_candidate != derived_candidate
            ):
                raise ValueError(f"{game_id}: explicit {metric} candidate flag conflicts with score")
            if (
                explicit_reference is not None
                and derived_reference is not None
                and explicit_reference != derived_reference
            ):
                raise ValueError(f"{game_id}: explicit {metric} reference flag conflicts with score")
            candidate_event = (
                explicit_candidate if explicit_candidate is not None else derived_candidate
            )
            reference_event = (
                explicit_reference if explicit_reference is not None else derived_reference
            )
            if candidate_event is None or reference_event is None:
                classification_missing[metric] += 1
                candidate_event = False if candidate_event is None else candidate_event
                reference_event = False if reference_event is None else reference_event
            candidate_event_counts[metric] += int(candidate_event)
            reference_event_counts[metric] += int(reference_event)
            risk_values[metric][game_id] = float(candidate_event) - float(reference_event)

        trace_state = traces.get(game_id, {})
        candidate_trace = trace_state.get(candidate_bot)
        reference_trace = trace_state.get(context["reference_bot"])
        candidate_lost = outcome == -1.0
        reference_lost = outcome == 1.0
        trace_metrics = (
            ("lead_40_loss", "lead_40"),
            ("lead_80_loss", "lead_80"),
            ("high_confidence_loss", "high_confidence"),
        )
        for metric, trace_key in trace_metrics:
            explicit_candidate = _explicit_risk_flag(row, metric, "candidate")
            explicit_reference = _explicit_risk_flag(row, metric, "reference")
            derived_candidate = bool(
                candidate_lost and candidate_trace and candidate_trace[trace_key]
            )
            derived_reference = bool(
                reference_lost and reference_trace and reference_trace[trace_key]
            )
            if (
                explicit_candidate is not None
                and candidate_trace is not None
                and explicit_candidate != derived_candidate
            ):
                raise ValueError(
                    f"{game_id}: explicit {metric} candidate flag conflicts with trace"
                )
            if (
                explicit_reference is not None
                and reference_trace is not None
                and explicit_reference != derived_reference
            ):
                raise ValueError(
                    f"{game_id}: explicit {metric} reference flag conflicts with trace"
                )
            candidate_event = (
                explicit_candidate
                if explicit_candidate is not None
                else derived_candidate
            )
            reference_event = (
                explicit_reference
                if explicit_reference is not None
                else derived_reference
            )
            if explicit_candidate is None and candidate_trace is None:
                classification_missing[metric] += 1
            if explicit_reference is None and reference_trace is None:
                classification_missing[metric] += 1
            candidate_event_counts[metric] += int(candidate_event)
            reference_event_counts[metric] += int(reference_event)
            risk_values[metric][game_id] = float(candidate_event) - float(reference_event)

        labels = _suite_labels(row)
        for metric, expected_labels in (
            ("targeted_lead_40_suite_loss", {"lead-40", "lead40"}),
            ("targeted_lead_80_suite_loss", {"lead-80", "lead80"}),
        ):
            explicit_candidate = _explicit_risk_flag(row, metric, "candidate")
            explicit_reference = _explicit_risk_flag(row, metric, "reference")
            is_targeted = bool(labels.intersection(expected_labels))
            if not is_targeted:
                if explicit_candidate is not None or explicit_reference is not None:
                    raise ValueError(
                        f"{game_id}: explicit {metric} flag is not backed by "
                        "frozen schedule suite metadata"
                    )
                continue
            targeted_pairs[metric].add(context["pair_id"])
            derived_candidate = outcome == -1.0
            derived_reference = outcome == 1.0
            if (
                explicit_candidate is not None
                and explicit_candidate != derived_candidate
            ):
                raise ValueError(
                    f"{game_id}: explicit {metric} candidate flag conflicts with outcome"
                )
            if (
                explicit_reference is not None
                and explicit_reference != derived_reference
            ):
                raise ValueError(
                    f"{game_id}: explicit {metric} reference flag conflicts with outcome"
                )
            candidate_event = (
                explicit_candidate if explicit_candidate is not None else derived_candidate
            )
            reference_event = (
                explicit_reference if explicit_reference is not None else derived_reference
            )
            if (explicit_candidate is None) != (explicit_reference is None):
                classification_missing[metric] += 1
            candidate_event_counts[metric] += int(candidate_event)
            reference_event_counts[metric] += int(reference_event)
            risk_values[metric][game_id] = float(candidate_event) - float(reference_event)

    risks: Dict[str, Dict[str, Any]] = {}
    for metric in RISK_METRICS:
        allowed_pairs = targeted_pairs.get(metric)
        aggregation = _aggregate_values(
            contexts,
            risk_values[metric],
            allowed_pair_ids=allowed_pairs if allowed_pairs is not None else None,
        )
        complete = classification_missing[metric] == 0
        if allowed_pairs is not None and not allowed_pairs:
            complete = False
        risks[metric] = _metric_report(
            aggregation,
            metric_name=metric,
            alpha=settings["catastrophe_alpha"],
            nominal_confidence=settings["catastrophe_nominal_confidence"],
            bootstrap_replications=replications,
            bootstrap_seed=seed,
            lower_limit=-1.0,
            upper_limit=1.0,
            complete=complete,
            candidate_events=candidate_event_counts[metric],
            reference_events=reference_event_counts[metric],
        )
        risks[metric]["classification_missing"] = classification_missing[metric]

    pair_ids_by_suite: Dict[str, Set[str]] = {"lead_40": set(), "lead_80": set()}
    for context in contexts:
        labels = _suite_labels(context["row"])
        if labels.intersection({"lead-40", "lead40"}):
            pair_ids_by_suite["lead_40"].add(context["pair_id"])
        if labels.intersection({"lead-80", "lead80"}):
            pair_ids_by_suite["lead_80"].add(context["pair_id"])

    suite_metrics: Dict[str, Dict[str, Any]] = {}
    combined_pair_ids = pair_ids_by_suite["lead_40"].union(pair_ids_by_suite["lead_80"])
    for suite_name, pair_ids in (
        ("lead_40", pair_ids_by_suite["lead_40"]),
        ("lead_80", pair_ids_by_suite["lead_80"]),
        ("combined_lead", combined_pair_ids),
    ):
        aggregation = _aggregate_values(
            contexts,
            utility_by_game,
            allowed_pair_ids=pair_ids,
        )
        suite_metrics[suite_name] = _metric_report(
            aggregation,
            metric_name=f"{suite_name}_realized_utility",
            alpha=settings["routine_alpha"],
            nominal_confidence=settings["routine_nominal_confidence"],
            bootstrap_replications=replications,
            bootstrap_seed=seed,
            complete=bool(pair_ids),
        )

    no_result_rate = (
        validation["true_no_results"] / len(rows) if rows else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "finalized": finalized,
        "data_binding": data_binding,
        "policy_version": active_policy.get("policy_version"),
        "policy_hash": canonical_sha256(active_policy),
        "candidate_bot": candidate_bot,
        "look": {
            "number": look_number,
            "total_prespecified_looks": settings["look_count"],
            "alpha_allocation_method": settings["allocation_method"],
            "routine_one_sided_alpha": settings["routine_alpha"],
            "catastrophe_one_sided_alpha": settings["catastrophe_alpha"],
            "routine_allocated_one_sided_confidence": 1.0
            - settings["routine_alpha"],
            "catastrophe_allocated_one_sided_confidence": 1.0
            - settings["catastrophe_alpha"],
            "routine_nominal_one_sided_confidence": settings[
                "routine_nominal_confidence"
            ],
            "catastrophe_nominal_one_sided_confidence": settings[
                "catastrophe_nominal_confidence"
            ],
            "data_dependent_thresholds_allowed": settings[
                "data_dependent_thresholds_allowed"
            ],
        },
        "objective": {
            "formula": (
                "win_weight*y + sign(m)*((1+abs(m)/score_scale)**score_power-1)"
            ),
            "candidate_perspective_outcome": "win=1, draw/no-result=0, loss=-1",
            "win_weight": win_weight,
            "score_scale": score_scale,
            "score_power": score_power,
            "missing_numeric_score_term": 0.0,
            "ordinary_win_score": win_scores,
        },
        "risk_definitions": {
            "direction": "candidate_minus_reference",
            "final_margin_threshold_points": [
                final_20_threshold,
                final_50_threshold,
            ],
            "lead_threshold_points": [
                lead_40_threshold,
                lead_80_threshold,
            ],
            "high_confidence_win_probability": high_confidence_threshold,
            "lead_and_high_confidence_events_use_own_turn_traces": True,
            "targeted_suite_events_use_frozen_schedule_binding": True,
            "explicit_flags_cannot_define_targeted_membership": True,
        },
        "validation": validation,
        "counts": {
            "games": len(rows),
            "color_pairs": validation["pairs"],
            "position_clusters": validation["position_clusters"],
            "resolved_missing_numeric_scores": validation[
                "resolved_missing_numeric_scores"
            ],
            "true_no_results": validation["true_no_results"],
            "true_no_result_rate": no_result_rate,
        },
        "aggregation_order": [
            "game_to_color_pair_arithmetic_mean",
            "repeated_pairs_to_position_id_arithmetic_mean",
            "position_cluster_inference",
        ],
        "metrics": {
            "realized_utility": utility_metric,
            "win_rate": win_rate_metric,
        },
        "suite_metrics": suite_metrics,
        "risk_differences": risks,
        "move_trace_coverage": trace_coverage,
        "bootstrap": {
            "method": bootstrap_config.get("method"),
            "weights": bootstrap_config.get("weights"),
            "replications": replications,
            "base_seed": seed,
        },
    }


def finalize_statistics_artifact(
    report: Mapping[str, Any],
    *,
    cell_name: str,
    metric_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Hash a finalized report and emit its exact gate-binding manifest."""

    if not isinstance(report, dict) or report.get("finalized") is not True:
        raise ValueError("statistics report must be finalized")
    if not _nonempty_string(cell_name):
        raise ValueError("cell_name must be nonempty")
    binding = report.get("data_binding")
    if not isinstance(binding, dict):
        raise ValueError("finalized statistics report lacks data_binding")
    artifact = json.loads(canonical_json(report))
    artifact_hash = canonical_sha256(artifact)
    metrics = artifact.get("metrics")
    primary_metric = (
        metrics.get("realized_utility") or metrics.get("win_rate")
        if isinstance(metrics, dict)
        else None
    )
    if not isinstance(primary_metric, dict):
        raise ValueError("finalized statistics report has no primary metric")
    position_values = primary_metric.get("position_values")
    if not isinstance(position_values, list):
        raise ValueError("primary metric has no position values")
    position_ids = sorted(
        row.get("position_id")
        for row in position_values
        if isinstance(row, dict) and _nonempty_string(row.get("position_id"))
    )
    if len(position_ids) != len(position_values) or len(position_ids) != len(
        set(position_ids)
    ):
        raise ValueError("primary metric position IDs are incomplete or duplicated")
    if metric_names is None:
        names: Set[str] = set()
        for container_name in ("metrics", "risk_differences"):
            container = artifact.get(container_name)
            if isinstance(container, dict):
                names.update(
                    value.get("metric")
                    for value in container.values()
                    if isinstance(value, dict)
                    and _nonempty_string(value.get("metric"))
                )
        selected_metric_names = sorted(names)
    else:
        selected_metric_names = sorted(set(metric_names))
        if not selected_metric_names or not all(
            _nonempty_string(name) for name in selected_metric_names
        ):
            raise ValueError("metric_names must contain nonempty names")
    manifest = {
        "schema_version": 1,
        "finalized": True,
        "cell_name": cell_name,
        **binding,
        "statistics_artifact_hash": artifact_hash,
        "color_pairs": artifact["counts"]["color_pairs"],
        "position_ids": position_ids,
        "metric_names": selected_metric_names,
    }
    return {
        "statistics_artifact": artifact,
        "statistics_artifact_hash": artifact_hash,
        "statistics_manifest": manifest,
        "statistics_manifest_hash": canonical_sha256(manifest),
    }


# Descriptive aliases kept intentionally small for callers and tests.
analyze_paired_matches = compute_paired_statistics
paired_statistics = compute_paired_statistics
summarize_paired_matches = compute_paired_statistics
validate_deterministic_match_rows = validate_match_rows
candidate_realized_utility = realized_utility


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute promotion statistics from deterministic paired match JSONL."
    )
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--candidate-bot", required=True)
    parser.add_argument("--moves", nargs="+", type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--look", type=int, default=1)
    parser.add_argument("--bootstrap-replications", type=int)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("-o", "--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = compute_paired_statistics(
            load_jsonl(args.results),
            candidate_bot=args.candidate_bot,
            move_records=load_jsonl(args.moves) if args.moves else None,
            policy=load_policy(args.policy),
            look_number=args.look,
            bootstrap_replications=args.bootstrap_replications,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
