#!/usr/bin/env python3
"""Pair-safe, crash-recoverable execution of deterministic KataGo matches."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from risk_score.build_evaluation_suites import semantic_position_sha256
from risk_score.generate_schedule import validate_position

SCHEMA_VERSION = 1
RUNNER_CONTRACT = "risk-score-pair-safe-evaluation-runner-v2"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_SCHEDULE_FIELDS = (
    "schemaVersion",
    "scheduleId",
    "gameId",
    "pairId",
    "positionId",
    "seed",
    "blackBot",
    "whiteBot",
    "startPosition",
)
_REQUIRED_RESULT_FIELDS = (
    "schemaVersion",
    "scheduleId",
    "gameId",
    "pairId",
    "positionId",
    "seed",
    "blackBot",
    "whiteBot",
    "blackBotIndex",
    "whiteBotIndex",
    "board",
    "rules",
    "komi",
    "finalResult",
    "finalWhiteMinusBlackScore",
    "winner",
    "moveCount",
    "blackMoveCount",
    "whiteMoveCount",
    "startTurnNumber",
    "hitTurnLimit",
    "resignation",
    "noResult",
    "scored",
    "gameHash",
)
_SCHEDULE_BINDING_FIELDS = (
    "suite",
    "suiteBank",
    "suiteBankSha256",
    "positionContentSha256",
    "positionSemanticSha256",
)
_OPTIONAL_SCHEDULE_PROVENANCE_FIELDS = (
    "suiteQualifiedName",
    "suiteHoldout",
    "independentClusterId",
)


class EvaluationError(RuntimeError):
    """Base error for evaluation planning and execution."""


class EvaluationValidationError(EvaluationError):
    """An input or output does not satisfy the deterministic contract."""


class EvaluationConflictError(EvaluationError):
    """An existing artifact contradicts the requested evaluation."""


@dataclass(frozen=True)
class EvaluationSpec:
    """Immutable content identities and policy coordinates for an evaluation."""

    candidate_model_sha: str
    reference_model_sha: str
    original_model_sha: str
    config_sha: str
    schedule_sha: str
    policy_sha: str
    comparison: str
    suite: str
    stage: str
    look: str
    topology: str
    suite_manifest_sha: Optional[str] = None
    suite_bank_sha: Optional[str] = None
    schedule_id: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in (
            "candidate_model_sha",
            "reference_model_sha",
            "original_model_sha",
            "config_sha",
            "schedule_sha",
            "policy_sha",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(
                value.lower()
            ):
                raise ValueError(f"{field_name} must be a 64-character SHA-256")
            object.__setattr__(self, field_name, value.lower())
        for field_name in ("suite_manifest_sha", "suite_bank_sha"):
            value = getattr(self, field_name)
            if value is not None:
                if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(
                    value.lower()
                ):
                    raise ValueError(f"{field_name} must be a 64-character SHA-256")
                object.__setattr__(self, field_name, value.lower())
        for field_name in ("comparison", "suite", "stage", "look", "topology"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or not value
                or "\n" in value
                or "\r" in value
            ):
                raise ValueError(f"{field_name} must be a nonempty single-line string")
        if self.schedule_id is not None:
            _nonempty_string(self.schedule_id, "schedule_id")

    def to_dict(self) -> Dict[str, Any]:
        return dict(asdict(self))

    @property
    def evaluation_key(self) -> str:
        return compute_evaluation_key(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationSpec":
        expected = set(cls.__dataclass_fields__)
        extra = sorted(set(value).difference(expected))
        optional = {"suite_manifest_sha", "suite_bank_sha", "schedule_id"}
        missing = sorted(expected.difference(value).difference(optional))
        if extra or missing:
            raise ValueError(
                f"EvaluationSpec keys differ; missing={missing}, extra={extra}"
            )
        return cls(**{key: value[key] for key in expected if key in value})


@dataclass(frozen=True)
class RunnerExecutionIdentity:
    katago_binary_sha: str
    move_traces: bool
    extra_argv: Tuple[str, ...]
    effective_shard_count: int
    effective_max_parallelism: int
    max_attempts: int
    cwd: str
    timeout: Optional[float]
    replace_env: bool
    environment_sha: str

    def __post_init__(self) -> None:
        for field_name in ("katago_binary_sha", "environment_sha"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"{field_name} must be a 64-character SHA-256")
        if (
            self.effective_shard_count <= 0
            or self.effective_max_parallelism <= 0
            or self.max_attempts <= 0
        ):
            raise ValueError("effective execution counts must be positive")
        _nonempty_string(self.cwd, "execution cwd")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "katagoBinarySha256": self.katago_binary_sha,
            "moveTraces": self.move_traces,
            "extraArgv": list(self.extra_argv),
            "effectiveShardCount": self.effective_shard_count,
            "effectiveMaxParallelism": self.effective_max_parallelism,
            "maxAttempts": self.max_attempts,
            "cwd": self.cwd,
            "timeout": self.timeout,
            "replaceEnv": self.replace_env,
            "effectiveEnvironmentSha256": self.environment_sha,
        }


@dataclass(frozen=True)
class ScheduleShard:
    index: int
    rows: Tuple[Dict[str, Any], ...]
    pair_ids: Tuple[str, ...]
    game_ids: Tuple[str, ...]


@dataclass(frozen=True)
class CommandPlan:
    shard_index: int
    attempt: int
    argv: Tuple[str, ...]
    schedule_path: Path
    result_path: Path
    move_path: Optional[Path]
    cwd: Optional[Path]
    env_overrides: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class EvaluationPlan:
    spec: EvaluationSpec
    evaluation_key: str
    schedule_path: Path
    schedule_id: str
    execution: RunnerExecutionIdentity
    schedule_rows: Tuple[Dict[str, Any], ...]
    manifest_cell: Optional[Mapping[str, Any]]
    shards: Tuple[ScheduleShard, ...]
    commands: Tuple[CommandPlan, ...]
    partial_dir: Path
    final_dir: Path


@dataclass(frozen=True)
class ShardResult:
    shard_index: int
    attempts: int
    schedule_path: Path
    result_path: Path
    move_path: Optional[Path]
    game_ids: Tuple[str, ...]
    pair_ids: Tuple[str, ...]
    result_sha256: str
    move_sha256: Optional[str]
    reused: bool = False


@dataclass(frozen=True)
class EvaluationResult:
    spec: EvaluationSpec
    evaluation_key: str
    final_dir: Path
    result_path: Path
    move_path: Optional[Path]
    manifest_path: Path
    manifest_sha256: str
    shards: Tuple[ShardResult, ...]
    reused: bool = False


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


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


def compute_evaluation_key(spec: EvaluationSpec) -> str:
    """Return the content-addressed input-spec key (without execution settings)."""

    return "eval-" + canonical_sha256(
        {
            "runnerContract": RUNNER_CONTRACT,
            "candidateModelSha256": spec.candidate_model_sha,
            "referenceModelSha256": spec.reference_model_sha,
            "originalModelSha256": spec.original_model_sha,
            "configSha256": spec.config_sha,
            "scheduleSha256": spec.schedule_sha,
            "policySha256": spec.policy_sha,
            "comparison": spec.comparison,
            "suite": spec.suite,
            "stage": spec.stage,
            "look": spec.look,
            "topology": spec.topology,
            "suiteManifestSha256": spec.suite_manifest_sha,
            "suiteBankSha256": spec.suite_bank_sha,
            "scheduleId": spec.schedule_id,
        }
    )


def compute_execution_key(
    spec: EvaluationSpec, execution: RunnerExecutionIdentity
) -> str:
    """Bind an EvaluationSpec to the exact verified runner execution."""

    return "eval-" + canonical_sha256(
        {
            "runnerContract": RUNNER_CONTRACT,
            "evaluationSpec": spec.to_dict(),
            "execution": execution.to_dict(),
        }
    )


def _nonempty_string(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        raise EvaluationValidationError(
            f"{source} must be a nonempty single-line string"
        )
    return value


def _game_start_player(start_position: Mapping[str, Any], source: str) -> str:
    player = start_position["nextPla"]
    if player not in ("B", "W"):
        raise EvaluationValidationError(f"{source} nextPla must be B or W")
    for index, move_player in enumerate(start_position["movePlas"]):
        if move_player != player:
            raise EvaluationValidationError(
                f"{source} movePlas[{index}] does not alternate from nextPla"
            )
        player = "W" if player == "B" else "B"
    return player


def _load_jsonl(
    path: Path, kind: str, *, allow_empty: bool = False
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationValidationError(
                    f"{path}:{line_number}: malformed {kind} JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise EvaluationValidationError(
                    f"{path}:{line_number}: {kind} row must be a JSON object"
                )
            rows.append(row)
    if not rows and not allow_empty:
        raise EvaluationValidationError(f"{path}: no {kind} rows")
    return rows


def validate_schedule_rows(
    rows: Iterable[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    """Validate the full color-pair schedule contract."""

    checked: List[Dict[str, Any]] = []
    game_ids: Set[str] = set()
    schedule_id: Optional[str] = None
    pair_rows: Dict[str, List[Dict[str, Any]]] = {}
    common_bindings: Dict[str, Any] = {}

    for index, input_row in enumerate(rows):
        source = f"schedule row {index}"
        if not isinstance(input_row, Mapping):
            raise EvaluationValidationError(f"{source} must be a JSON object")
        row = dict(input_row)
        missing = [field for field in _REQUIRED_SCHEDULE_FIELDS if field not in row]
        if missing:
            raise EvaluationValidationError(
                f"{source} is missing required fields: {', '.join(missing)}"
            )
        if type(row["schemaVersion"]) is not int or row["schemaVersion"] != 1:
            raise EvaluationValidationError(f"{source} has unsupported schemaVersion")
        current_schedule_id = _nonempty_string(
            row["scheduleId"], f"{source} scheduleId"
        )
        game_id = _nonempty_string(row["gameId"], f"{source} gameId")
        pair_id = _nonempty_string(row["pairId"], f"{source} pairId")
        _nonempty_string(row["positionId"], f"{source} positionId")
        _nonempty_string(row["seed"], f"{source} seed")
        if schedule_id is None:
            schedule_id = current_schedule_id
        elif schedule_id != current_schedule_id:
            raise EvaluationValidationError(
                "all schedule rows must share one scheduleId"
            )
        if game_id in game_ids:
            raise EvaluationValidationError(f"duplicate schedule gameId {game_id!r}")
        game_ids.add(game_id)
        if type(row["blackBot"]) is not int or row["blackBot"] < 0:
            raise EvaluationValidationError(
                f"{source} blackBot must be a nonnegative integer"
            )
        if type(row["whiteBot"]) is not int or row["whiteBot"] < 0:
            raise EvaluationValidationError(
                f"{source} whiteBot must be a nonnegative integer"
            )
        if row["blackBot"] == row["whiteBot"]:
            raise EvaluationValidationError(f"{source} must use two different bots")
        try:
            validate_position(row["startPosition"], f"{source} startPosition")
        except ValueError as exc:
            raise EvaluationValidationError(str(exc)) from exc
        start_position = row["startPosition"]
        _game_start_player(start_position, f"{source} startPosition")
        if (
            type(start_position["initialTurnNumber"]) is not int
            or start_position["initialTurnNumber"] < 0
        ):
            raise EvaluationValidationError(
                f"{source} initialTurnNumber must be a nonnegative integer"
            )

        present_bindings = {field for field in _SCHEDULE_BINDING_FIELDS if field in row}
        if present_bindings and present_bindings != set(_SCHEDULE_BINDING_FIELDS):
            raise EvaluationValidationError(
                f"{source} has incomplete suite binding fields"
            )
        if present_bindings:
            suite = _nonempty_string(row["suite"], f"{source} suite")
            suite_bank = _nonempty_string(row["suiteBank"], f"{source} suiteBank")
            if suite != suite_bank:
                raise EvaluationValidationError(
                    f"{source} suite and suiteBank must match"
                )
            for field in (
                "suiteBankSha256",
                "positionContentSha256",
                "positionSemanticSha256",
            ):
                value = row[field]
                if (
                    not isinstance(value, str)
                    or _SHA256_PATTERN.fullmatch(value) is None
                ):
                    raise EvaluationValidationError(
                        f"{source} {field} must be a SHA-256"
                    )
            if row["positionContentSha256"] != canonical_sha256(start_position):
                raise EvaluationValidationError(
                    f"{source} positionContentSha256 contradicts startPosition"
                )
            if row["positionSemanticSha256"] != semantic_position_sha256(
                start_position
            ):
                raise EvaluationValidationError(
                    f"{source} positionSemanticSha256 contradicts startPosition"
                )
            for field in ("suite", "suiteBank", "suiteBankSha256"):
                if field not in common_bindings:
                    common_bindings[field] = row[field]
                elif common_bindings[field] != row[field]:
                    raise EvaluationValidationError(
                        f"schedule rows use inconsistent {field}"
                    )
            if "suiteQualifiedName" in row:
                _nonempty_string(
                    row["suiteQualifiedName"], f"{source} suiteQualifiedName"
                )
            if "suiteHoldout" in row and row["suiteHoldout"] is not None:
                _nonempty_string(row["suiteHoldout"], f"{source} suiteHoldout")
            if "independentClusterId" in row:
                cluster_id = row["independentClusterId"]
                if (
                    not isinstance(cluster_id, str)
                    or _SHA256_PATTERN.fullmatch(cluster_id) is None
                ):
                    raise EvaluationValidationError(
                        f"{source} independentClusterId must be a SHA-256"
                    )
                if cluster_id != row["positionSemanticSha256"]:
                    raise EvaluationValidationError(
                        f"{source} independentClusterId contradicts "
                        "positionSemanticSha256"
                    )
        checked.append(row)
        pair_rows.setdefault(pair_id, []).append(row)

    if not checked:
        raise EvaluationValidationError("schedule must contain at least one row")

    for pair_id, group in pair_rows.items():
        if len(group) != 2:
            raise EvaluationValidationError(
                f"pairId {pair_id!r} must contain exactly two color-reversed games"
            )
        first, second = group
        if (
            first["blackBot"] != second["whiteBot"]
            or first["whiteBot"] != second["blackBot"]
        ):
            raise EvaluationValidationError(f"pairId {pair_id!r} is not color reversed")
        if first["positionId"] != second["positionId"]:
            raise EvaluationValidationError(
                f"pairId {pair_id!r} contains multiple positionIds"
            )
        if canonical_json(first["startPosition"]) != canonical_json(
            second["startPosition"]
        ):
            raise EvaluationValidationError(
                f"pairId {pair_id!r} contains different start positions"
            )
        if first["seed"] == second["seed"]:
            raise EvaluationValidationError(
                f"pairId {pair_id!r} must use distinct seeds"
            )
        for field in _SCHEDULE_BINDING_FIELDS:
            if first.get(field) != second.get(field):
                raise EvaluationValidationError(
                    f"pairId {pair_id!r} has inconsistent {field}"
                )
        for field in _OPTIONAL_SCHEDULE_PROVENANCE_FIELDS:
            if first.get(field) != second.get(field):
                raise EvaluationValidationError(
                    f"pairId {pair_id!r} has inconsistent {field}"
                )
    return tuple(checked)


def load_schedule(path: Path) -> Tuple[Dict[str, Any], ...]:
    return validate_schedule_rows(_load_jsonl(Path(path), "schedule"))


def shard_schedule(
    rows: Iterable[Mapping[str, Any]], shard_count: int
) -> Tuple[ScheduleShard, ...]:
    """Balance complete pair groups across at most ``shard_count`` shards."""

    if type(shard_count) is not int or shard_count <= 0:
        raise ValueError("shard_count must be a positive integer")
    checked = validate_schedule_rows(rows)
    groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    order: List[str] = []
    for row_index, row in enumerate(checked):
        pair_id = row["pairId"]
        if pair_id not in groups:
            groups[pair_id] = []
            order.append(pair_id)
        groups[pair_id].append((row_index, row))

    actual_count = min(shard_count, len(groups))
    assigned: List[List[Tuple[int, Dict[str, Any]]]] = [[] for _ in range(actual_count)]
    loads = [0] * actual_count
    order_index = {pair_id: index for index, pair_id in enumerate(order)}
    for pair_id in sorted(
        order, key=lambda value: (-len(groups[value]), order_index[value])
    ):
        target = min(range(actual_count), key=lambda index: (loads[index], index))
        assigned[target].extend(groups[pair_id])
        loads[target] += len(groups[pair_id])

    shards: List[ScheduleShard] = []
    for shard_index, indexed_rows in enumerate(assigned):
        ordered_rows = tuple(
            row for _, row in sorted(indexed_rows, key=lambda item: item[0])
        )
        pair_ids = tuple(dict.fromkeys(row["pairId"] for row in ordered_rows))
        shards.append(
            ScheduleShard(
                index=shard_index,
                rows=ordered_rows,
                pair_ids=pair_ids,
                game_ids=tuple(row["gameId"] for row in ordered_rows),
            )
        )
    return tuple(shards)


def _expected_by_game(
    schedule_rows: Iterable[Mapping[str, Any]],
) -> Tuple[Tuple[Dict[str, Any], ...], Dict[str, Dict[str, Any]]]:
    checked = validate_schedule_rows(schedule_rows)
    return checked, {row["gameId"]: row for row in checked}


def _validate_identity_fields(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    source: str,
) -> None:
    for field in ("scheduleId", "gameId", "pairId", "positionId"):
        if row.get(field) != expected[field]:
            raise EvaluationValidationError(
                f"{source} has wrong {field}: {row.get(field)!r}, "
                f"expected {expected[field]!r}"
            )


def _finite_number(value: Any, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationValidationError(f"{source} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise EvaluationValidationError(f"{source} must be a finite number")
    return number


def _nonnegative_integer(value: Any, source: str) -> int:
    if type(value) is not int or value < 0:
        raise EvaluationValidationError(f"{source} must be a nonnegative integer")
    return value


def _boolean(value: Any, source: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluationValidationError(f"{source} must be boolean")
    return value


def _result_winner(final_result: str) -> Optional[str]:
    if final_result.startswith("B+"):
        return "B"
    if final_result.startswith("W+"):
        return "W"
    if final_result in ("0", "draw", "Draw"):
        return "draw"
    return None


def _copy_schedule_bindings(
    row: Dict[str, Any], expected: Mapping[str, Any], source: str
) -> None:
    start_position = expected["startPosition"]
    if "metadata" in start_position:
        metadata = json.loads(canonical_json(start_position["metadata"]))
        if "metadata" in row and row["metadata"] != metadata:
            raise EvaluationValidationError(
                f"{source} metadata contradicts immutable schedule metadata"
            )
        row["metadata"] = metadata
    elif "metadata" in row:
        raise EvaluationValidationError(f"{source} has unscheduled metadata")

    for field in _SCHEDULE_BINDING_FIELDS:
        if field in expected:
            if field in row and row[field] != expected[field]:
                raise EvaluationValidationError(
                    f"{source} {field} contradicts immutable schedule binding"
                )
            row[field] = expected[field]
        elif field in row:
            raise EvaluationValidationError(
                f"{source} has unscheduled binding field {field}"
            )
    for field in _OPTIONAL_SCHEDULE_PROVENANCE_FIELDS:
        if field in expected:
            if field in row and row[field] != expected[field]:
                raise EvaluationValidationError(
                    f"{source} {field} contradicts immutable schedule provenance"
                )
            row[field] = expected[field]
        elif field in row:
            raise EvaluationValidationError(
                f"{source} has unscheduled provenance field {field}"
            )


def validate_result_rows(
    rows: Iterable[Mapping[str, Any]],
    schedule_rows: Iterable[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    """Require exactly one correctly identified result for every scheduled game."""

    schedule, expected_by_game = _expected_by_game(schedule_rows)
    by_game: Dict[str, Dict[str, Any]] = {}
    bot_names_by_index: Dict[int, str] = {}
    common_rules: Optional[str] = None
    common_komi: Optional[float] = None
    for index, input_row in enumerate(rows):
        source = f"result row {index}"
        if not isinstance(input_row, Mapping):
            raise EvaluationValidationError(f"{source} must be a JSON object")
        row = dict(input_row)
        missing_fields = [
            field for field in _REQUIRED_RESULT_FIELDS if field not in row
        ]
        if missing_fields:
            raise EvaluationValidationError(
                f"{source} is missing required C++ v1 fields: "
                f"{', '.join(missing_fields)}"
            )
        if type(row.get("schemaVersion")) is not int or row["schemaVersion"] != 1:
            raise EvaluationValidationError(f"{source} has unsupported schemaVersion")
        game_id = _nonempty_string(row.get("gameId"), f"{source} gameId")
        if game_id in by_game:
            raise EvaluationValidationError(f"duplicate result gameId {game_id!r}")
        expected = expected_by_game.get(game_id)
        if expected is None:
            raise EvaluationValidationError(f"unexpected result gameId {game_id!r}")
        _validate_identity_fields(row, expected, source)
        if row["seed"] != expected["seed"]:
            raise EvaluationValidationError(f"{source} has wrong schedule seed")

        black_index = row["blackBotIndex"]
        white_index = row["whiteBotIndex"]
        if type(black_index) is not int or black_index != expected["blackBot"]:
            raise EvaluationValidationError(f"{source} has wrong blackBotIndex")
        if type(white_index) is not int or white_index != expected["whiteBot"]:
            raise EvaluationValidationError(f"{source} has wrong whiteBotIndex")
        black_name = _nonempty_string(row["blackBot"], f"{source} blackBot")
        white_name = _nonempty_string(row["whiteBot"], f"{source} whiteBot")
        if black_name == white_name:
            raise EvaluationValidationError(f"{source} bot names must be distinct")
        for bot_index, bot_name in (
            (black_index, black_name),
            (white_index, white_name),
        ):
            commanded_name = {0: "candidate", 1: "reference"}.get(bot_index)
            if commanded_name is not None and bot_name != commanded_name:
                raise EvaluationValidationError(
                    f"{source} bot name for index {bot_index} must be "
                    f"{commanded_name!r}"
                )
            previous_name = bot_names_by_index.setdefault(bot_index, bot_name)
            if previous_name != bot_name:
                raise EvaluationValidationError(
                    f"{source} changes bot name for index {bot_index}"
                )

        board = row["board"]
        if (
            not isinstance(board, dict)
            or set(board) != {"xSize", "ySize"}
            or type(board["xSize"]) is not int
            or type(board["ySize"]) is not int
            or board["xSize"] != expected["startPosition"]["xSize"]
            or board["ySize"] != expected["startPosition"]["ySize"]
        ):
            raise EvaluationValidationError(
                f"{source} board dimensions contradict startPosition"
            )
        rules = row["rules"]
        if not isinstance(rules, dict) or not rules:
            raise EvaluationValidationError(f"{source} rules must be a nonempty object")
        rules_json = canonical_json(rules)
        if common_rules is None:
            common_rules = rules_json
        elif common_rules != rules_json:
            raise EvaluationValidationError("result rows use inconsistent rules")
        komi = _finite_number(row["komi"], f"{source} komi")
        if common_komi is None:
            common_komi = komi
        elif common_komi != komi:
            raise EvaluationValidationError("result rows use inconsistent komi")

        start_position = expected["startPosition"]
        expected_start_turn = start_position["initialTurnNumber"] + len(
            start_position["moveLocs"]
        )
        start_turn = _nonnegative_integer(
            row["startTurnNumber"], f"{source} startTurnNumber"
        )
        if start_turn != expected_start_turn:
            raise EvaluationValidationError(
                f"{source} startTurnNumber contradicts startPosition"
            )
        move_count = _nonnegative_integer(row["moveCount"], f"{source} moveCount")
        black_moves = _nonnegative_integer(
            row["blackMoveCount"], f"{source} blackMoveCount"
        )
        white_moves = _nonnegative_integer(
            row["whiteMoveCount"], f"{source} whiteMoveCount"
        )
        if black_moves + white_moves != move_count:
            raise EvaluationValidationError(
                f"{source} color move counts do not sum to moveCount"
            )
        next_player = _game_start_player(start_position, f"{source} startPosition")
        expected_first_moves = (move_count + 1) // 2
        expected_second_moves = move_count // 2
        expected_black_moves = (
            expected_first_moves if next_player == "B" else expected_second_moves
        )
        expected_white_moves = move_count - expected_black_moves
        if (black_moves, white_moves) != (
            expected_black_moves,
            expected_white_moves,
        ):
            raise EvaluationValidationError(
                f"{source} color move counts contradict starting player"
            )

        hit_turn_limit = _boolean(row["hitTurnLimit"], f"{source} hitTurnLimit")
        resignation = _boolean(row["resignation"], f"{source} resignation")
        no_result = _boolean(row["noResult"], f"{source} noResult")
        scored = _boolean(row["scored"], f"{source} scored")
        if resignation:
            raise EvaluationValidationError(
                f"{source} resignation is invalid for promotion"
            )
        if hit_turn_limit:
            raise EvaluationValidationError(
                f"{source} turn limit is invalid for promotion"
            )
        if resignation and no_result:
            raise EvaluationValidationError(
                f"{source} cannot be both resignation and no-result"
            )

        final_result = _nonempty_string(row["finalResult"], f"{source} finalResult")
        winner = row["winner"]
        if winner not in ("B", "W", "draw", None):
            raise EvaluationValidationError(
                f"{source} winner must be B, W, draw, or null"
            )
        score_value = row["finalWhiteMinusBlackScore"]
        if no_result:
            if scored or winner is not None or score_value is not None:
                raise EvaluationValidationError(
                    f"{source} no-result has inconsistent scored/winner/score fields"
                )
            if final_result != "Void":
                raise EvaluationValidationError(
                    f"{source} no-result finalResult must be 'Void'"
                )
        else:
            if not scored:
                raise EvaluationValidationError(
                    f"{source} resolved promotion game must be scored"
                )
            score = _finite_number(score_value, f"{source} finalWhiteMinusBlackScore")
            score_winner = "W" if score > 0.0 else "B" if score < 0.0 else "draw"
            if winner != score_winner:
                raise EvaluationValidationError(
                    f"{source} winner contradicts final score"
                )
            if _result_winner(final_result) != winner:
                raise EvaluationValidationError(
                    f"{source} finalResult contradicts winner"
                )
            if winner in ("B", "W"):
                try:
                    result_margin = float(final_result.split("+", 1)[1])
                except (IndexError, ValueError) as exc:
                    raise EvaluationValidationError(
                        f"{source} scored finalResult must contain a numeric margin"
                    ) from exc
                if not math.isfinite(result_margin) or not math.isclose(
                    result_margin,
                    abs(score),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ):
                    raise EvaluationValidationError(
                        f"{source} finalResult margin contradicts final score"
                    )
        _nonempty_string(row["gameHash"], f"{source} gameHash")
        _copy_schedule_bindings(row, expected, source)
        by_game[game_id] = row
    missing = [row["gameId"] for row in schedule if row["gameId"] not in by_game]
    if missing:
        raise EvaluationValidationError(f"missing scheduled result gameIds: {missing}")
    return tuple(by_game[row["gameId"]] for row in schedule)


def validate_result_jsonl(
    path: Path, schedule_rows: Iterable[Mapping[str, Any]]
) -> Tuple[Dict[str, Any], ...]:
    return validate_result_rows(_load_jsonl(Path(path), "result"), schedule_rows)


def validate_move_rows(
    rows: Iterable[Mapping[str, Any]],
    schedule_rows: Iterable[Mapping[str, Any]],
    result_rows: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Tuple[Dict[str, Any], ...]:
    """Validate an exact contiguous move trace against finalized game results."""

    schedule, expected_by_game = _expected_by_game(schedule_rows)
    if result_rows is None:
        raise EvaluationValidationError(
            "validated result rows are required for move trace validation"
        )
    results = validate_result_rows(result_rows, schedule)
    results_by_game = {row["gameId"]: row for row in results}
    game_order = {row["gameId"]: index for index, row in enumerate(schedule)}
    seen_turns: Set[Tuple[str, int]] = set()
    turns_by_game: Dict[str, List[int]] = {row["gameId"]: [] for row in schedule}
    checked: List[Dict[str, Any]] = []
    for index, input_row in enumerate(rows):
        source = f"move row {index}"
        if not isinstance(input_row, Mapping):
            raise EvaluationValidationError(f"{source} must be a JSON object")
        row = dict(input_row)
        if type(row.get("schemaVersion")) is not int or row["schemaVersion"] != 1:
            raise EvaluationValidationError(f"{source} has unsupported schemaVersion")
        game_id = _nonempty_string(row.get("gameId"), f"{source} gameId")
        expected = expected_by_game.get(game_id)
        if expected is None:
            raise EvaluationValidationError(f"unexpected move gameId {game_id!r}")
        _validate_identity_fields(row, expected, source)
        if row.get("seed") != expected["seed"]:
            raise EvaluationValidationError(f"{source} has wrong schedule seed")
        turn_number = row.get("turnNumber")
        if type(turn_number) is not int or turn_number < 0:
            raise EvaluationValidationError(
                f"{source} turnNumber must be a nonnegative integer"
            )
        identity = (game_id, turn_number)
        if identity in seen_turns:
            raise EvaluationValidationError(
                f"duplicate move trace identity gameId={game_id!r}, "
                f"turnNumber={turn_number}"
            )
        seen_turns.add(identity)
        result = results_by_game[game_id]
        start_turn = result["startTurnNumber"]
        offset = turn_number - start_turn
        if offset < 0 or offset >= result["moveCount"]:
            raise EvaluationValidationError(
                f"{source} turnNumber is outside the result move range"
            )
        first_player = _game_start_player(
            expected["startPosition"], f"{source} startPosition"
        )
        expected_player = (
            first_player if offset % 2 == 0 else "W" if first_player == "B" else "B"
        )
        if row.get("player") != expected_player:
            raise EvaluationValidationError(
                f"{source} player does not alternate from startPosition.nextPla"
            )
        expected_bot = (
            result["blackBot"] if expected_player == "B" else result["whiteBot"]
        )
        if row.get("bot") != expected_bot:
            raise EvaluationValidationError(
                f"{source} bot does not match the result color assignment"
            )
        score_lead = _finite_number(row.get("scoreLead"), f"{source} scoreLead")
        win_probability = _finite_number(
            row.get("winProbability"), f"{source} winProbability"
        )
        if not 0.0 <= win_probability <= 1.0:
            raise EvaluationValidationError(
                f"{source} winProbability must be between zero and one"
            )
        row["scoreLead"] = score_lead
        row["winProbability"] = win_probability
        turns_by_game[game_id].append(turn_number)
        checked.append(row)
    for result in results:
        game_id = result["gameId"]
        expected_turns = list(
            range(
                result["startTurnNumber"],
                result["startTurnNumber"] + result["moveCount"],
            )
        )
        actual_turns = sorted(turns_by_game[game_id])
        if actual_turns != expected_turns:
            raise EvaluationValidationError(
                f"move traces for gameId {game_id!r} are missing or have extra turns; "
                f"expected {expected_turns}, got {actual_turns}"
            )
    return tuple(
        sorted(
            checked,
            key=lambda row: (game_order[row["gameId"]], row["turnNumber"]),
        )
    )


def validate_move_jsonl(
    path: Path,
    schedule_rows: Iterable[Mapping[str, Any]],
    result_rows: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Tuple[Dict[str, Any], ...]:
    return validate_move_rows(
        _load_jsonl(Path(path), "move trace", allow_empty=True),
        schedule_rows,
        result_rows,
    )


def _override_value(value: Union[str, Path], name: str) -> str:
    text = str(value)
    if not text or "," in text or "\n" in text or "\r" in text:
        raise ValueError(
            f"{name} cannot be empty or contain comma/newline in KataGo overrides"
        )
    return text


def build_match_command(
    katago_binary: Path,
    config_path: Path,
    candidate_model_path: Path,
    reference_model_path: Path,
    schedule_path: Path,
    result_path: Path,
    *,
    game_count: int,
    move_path: Optional[Path] = None,
    candidate_name: str = "candidate",
    reference_name: str = "reference",
    extra_args: Sequence[str] = (),
) -> Tuple[str, ...]:
    """Construct a shell-free KataGo argv tuple for one shard."""

    if type(game_count) is not int or game_count <= 0:
        raise ValueError("game_count must be a positive integer")
    overrides = [
        "numBots=2",
        f"botName0={_override_value(candidate_name, 'candidate_name')}",
        f"botName1={_override_value(reference_name, 'reference_name')}",
        f"nnModelFile0={_override_value(candidate_model_path, 'candidate_model_path')}",
        f"nnModelFile1={_override_value(reference_model_path, 'reference_model_path')}",
        f"deterministicScheduleFile={_override_value(schedule_path, 'schedule_path')}",
        f"matchResultJsonlFile={_override_value(result_path, 'result_path')}",
        f"numGamesTotal={game_count}",
    ]
    if move_path is None:
        overrides.append("matchMoveJsonlFile=")
    else:
        overrides.append(
            f"matchMoveJsonlFile={_override_value(move_path, 'move_path')}"
        )
    argv = [
        _override_value(katago_binary, "katago_binary"),
        "match",
        "-config",
        _override_value(config_path, "config_path"),
        "-override-config",
        ",".join(overrides),
    ]
    for argument in extra_args:
        if not isinstance(argument, str) or "\x00" in argument:
            raise ValueError("extra_args must contain strings without NUL bytes")
        argv.append(argument)
    return tuple(argv)


def _canonical_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_fsynced(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_publish_bytes(
    path: Path,
    data: bytes,
    *,
    replace_conflicting: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == data:
            return
        if not replace_conflicting:
            raise EvaluationConflictError(f"existing artifact contradicts {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _failure_text(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:4000]


def _verify_file_hash(path: Path, expected_sha: str, role: str) -> None:
    try:
        actual = file_sha256(path)
    except OSError as exc:
        raise EvaluationValidationError(f"cannot hash {role} {path}: {exc}") from exc
    if actual != expected_sha:
        raise EvaluationValidationError(
            f"{role} SHA-256 mismatch for {path}: {actual}, expected {expected_sha}"
        )


def _verify_canonical_json_hash(path: Path, expected_sha: str, role: str) -> None:
    def unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise EvaluationValidationError(f"cannot load {role} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationValidationError(f"{role} {path} must be a JSON object")
    actual = canonical_sha256(value)
    if actual != expected_sha:
        raise EvaluationValidationError(
            f"{role} canonical SHA-256 mismatch for {path}: "
            f"{actual}, expected {expected_sha}"
        )


def load_suite_manifest(path: Path) -> Dict[str, Any]:
    """Load a canonical suite manifest and verify its self-hashed payload."""

    manifest_path = Path(path)
    try:
        data = manifest_path.read_bytes()
        value = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError(
            f"cannot load evaluation suite manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise EvaluationValidationError(
            "evaluation suite manifest must be a JSON object"
        )
    expected_data = (canonical_json(value) + "\n").encode("utf-8")
    if data != expected_data:
        raise EvaluationValidationError(
            "evaluation suite manifest must be canonical newline-terminated JSON"
        )
    payload = dict(value)
    payload_hash = payload.pop("manifestPayloadSha256", None)
    if payload_hash != canonical_sha256(payload):
        raise EvaluationValidationError(
            "evaluation suite manifest payload SHA-256 is invalid"
        )
    return value


def _iter_manifest_cells(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(value, dict):
        return
    has_coordinate = (
        ("comparison" in value)
        and ("stage" in value)
        and ("look" in value)
        and ("schedule_hash" in value or isinstance(value.get("schedule"), dict))
    )
    if has_coordinate:
        yield value
        return
    for child in value.values():
        yield from _iter_manifest_cells(child)


def _relative_manifest_path(value: Any, source: str) -> str:
    text = _nonempty_string(value, source)
    path = Path(text)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise EvaluationValidationError(f"{source} must be a normalized relative path")
    return path.as_posix()


def _manifest_cell_field(
    cell: Mapping[str, Any],
    name: str,
    *,
    nested: Optional[Tuple[str, str]] = None,
) -> Any:
    if name in cell:
        return cell[name]
    if nested is not None:
        container = cell.get(nested[0])
        if isinstance(container, dict):
            return container.get(nested[1])
    return None


def resolve_manifest_cell(
    manifest: Union[Path, Mapping[str, Any]],
    *,
    stage: str,
    look: str,
    comparison: str,
    suite: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve and validate one exact stage/look/comparison manifest cell."""

    if isinstance(manifest, (str, os.PathLike, Path)):
        loaded = load_suite_manifest(Path(manifest))
    elif isinstance(manifest, Mapping):
        loaded = json.loads(canonical_json(manifest))
        payload = dict(loaded)
        payload_hash = payload.pop("manifestPayloadSha256", None)
        if payload_hash != canonical_sha256(payload):
            raise EvaluationValidationError(
                "evaluation suite manifest payload SHA-256 is invalid"
            )
    else:
        raise EvaluationValidationError(
            "evaluation suite manifest must be a path or object"
        )

    cells = list(_iter_manifest_cells(loaded.get("cells")))
    matches = [
        dict(cell)
        for cell in cells
        if cell.get("stage") == stage
        and cell.get("look") == look
        and cell.get("comparison") == comparison
        and (suite is None or cell.get("suite") == suite)
    ]
    if len(matches) != 1:
        raise EvaluationValidationError(
            "suite manifest coordinates must resolve exactly one cell; "
            f"stage={stage!r}, look={look!r}, comparison={comparison!r}, "
            f"suite={suite!r}, matches={len(matches)}"
        )
    cell = matches[0]
    cell_name = _nonempty_string(cell.get("cell_name"), "manifest cell_name")
    cell_id = _nonempty_string(cell.get("cell_id"), "manifest cell_id")
    cell_payload = dict(cell)
    cell_payload.pop("cell_id", None)
    expected_cell_id = "suite-cell-" + canonical_sha256(cell_payload)
    if cell_id != expected_cell_id:
        raise EvaluationValidationError("suite manifest cell_id is invalid")

    manifest_policy_hash = loaded.get("policy_hash")
    manifest_policy_version = loaded.get("policy_version")
    manifest_source_revision = loaded.get("source_revision")
    if (
        cell.get("policy_hash") != manifest_policy_hash
        or cell.get("policy_version") != manifest_policy_version
        or cell.get("source_revision") != manifest_source_revision
    ):
        raise EvaluationValidationError(
            "suite manifest cell policy/source binding is inconsistent"
        )
    for field_name in ("policy_hash", "bank_hash", "schedule_hash"):
        value = cell.get(field_name)
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise EvaluationValidationError(
                f"suite manifest cell {field_name} must be a SHA-256"
            )
    _relative_manifest_path(cell.get("bank_path"), "manifest cell bank_path")
    _relative_manifest_path(cell.get("schedule_path"), "manifest cell schedule_path")
    _nonempty_string(cell.get("schedule_id"), "manifest cell schedule_id")

    pair_count = cell.get("color_pairs")
    row_count = cell.get("schedule_row_count")
    minimum_clusters = cell.get("minimum_independent_position_clusters")
    if (
        type(pair_count) is not int
        or pair_count <= 0
        or type(row_count) is not int
        or row_count != pair_count * 2
        or type(minimum_clusters) is not int
        or minimum_clusters <= 0
        or minimum_clusters > pair_count
    ):
        raise EvaluationValidationError(
            "suite manifest cell has invalid pair/row/cluster quotas"
        )
    cluster_ids = cell.get("independent_cluster_ids")
    if not (
        isinstance(cluster_ids, list)
        and len(cluster_ids) == pair_count
        and len(cluster_ids) == len(set(cluster_ids))
        and all(
            isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None
            for value in cluster_ids
        )
    ):
        raise EvaluationValidationError(
            "suite manifest cell must use one unique SHA-256 cluster per pair"
        )
    if cell.get("independent_cluster_ids_hash") != canonical_sha256(cluster_ids):
        raise EvaluationValidationError(
            "suite manifest cell independent_cluster_ids_hash is invalid"
        )
    position_ids = cell.get("position_ids")
    if not (
        isinstance(position_ids, list)
        and len(position_ids) == pair_count
        and len(position_ids) == len(set(position_ids))
        and all(isinstance(value, str) and value for value in position_ids)
    ):
        raise EvaluationValidationError(
            "suite manifest cell position_ids are incomplete or duplicated"
        )
    if cell.get("position_ids_hash") != canonical_sha256(position_ids):
        raise EvaluationValidationError(
            "suite manifest cell position_ids_hash is invalid"
        )
    cell["cell_name"] = cell_name
    return cell


def build_evaluation_matrix(
    *,
    candidate_model_sha: str,
    champion_model_sha: str,
    original_model_sha: str,
    powered_config_sha: str,
    standard_config_sha: str,
    powered_schedule_sha: str,
    standard_schedule_sha: str,
    policy_sha: str,
    suite: str,
    stage: str,
    look: str,
    topology: str,
    include_standard_champion: bool = False,
    lead_40_schedule_sha: Optional[str] = None,
    lead_80_schedule_sha: Optional[str] = None,
    suite_manifest_sha: Optional[str] = None,
    ordinary_suite_bank_sha: Optional[str] = None,
    lead_40_suite_bank_sha: Optional[str] = None,
    lead_80_suite_bank_sha: Optional[str] = None,
    powered_schedule_id: Optional[str] = None,
    standard_schedule_id: Optional[str] = None,
    lead_40_schedule_id: Optional[str] = None,
    lead_80_schedule_id: Optional[str] = None,
) -> Tuple[EvaluationSpec, ...]:
    """Construct the policy-independent powered/standard comparison matrix."""

    coordinates = [
        (
            "candidate-vs-champion-powered",
            suite,
            champion_model_sha,
            powered_config_sha,
            powered_schedule_sha,
            ordinary_suite_bank_sha,
            powered_schedule_id,
        ),
        (
            "candidate-vs-original-powered",
            suite,
            original_model_sha,
            powered_config_sha,
            powered_schedule_sha,
            ordinary_suite_bank_sha,
            powered_schedule_id,
        ),
    ]
    if include_standard_champion:
        coordinates.append(
            (
                "candidate-vs-champion-standard",
                suite,
                champion_model_sha,
                standard_config_sha,
                standard_schedule_sha,
                ordinary_suite_bank_sha,
                standard_schedule_id,
            )
        )
    coordinates.append(
        (
            "candidate-vs-original-standard",
            suite,
            original_model_sha,
            standard_config_sha,
            standard_schedule_sha,
            ordinary_suite_bank_sha,
            standard_schedule_id,
        )
    )
    if (lead_40_schedule_sha is None) != (lead_80_schedule_sha is None):
        raise ValueError(
            "confirmation requires both Lead-40 and Lead-80 schedule hashes"
        )
    if lead_40_schedule_sha is not None and lead_80_schedule_sha is not None:
        normalized_stage = stage.lower().replace("_", "-")
        if "confirmation" not in normalized_stage and normalized_stage not in (
            "stage-3",
            "3",
        ):
            raise ValueError("Lead confirmation cells require a confirmation stage")
        coordinates.extend(
            (
                (
                    "candidate-vs-champion-powered-lead-40",
                    "lead-40",
                    champion_model_sha,
                    powered_config_sha,
                    lead_40_schedule_sha,
                    lead_40_suite_bank_sha,
                    lead_40_schedule_id,
                ),
                (
                    "candidate-vs-champion-powered-lead-80",
                    "lead-80",
                    champion_model_sha,
                    powered_config_sha,
                    lead_80_schedule_sha,
                    lead_80_suite_bank_sha,
                    lead_80_schedule_id,
                ),
            )
        )
    return tuple(
        EvaluationSpec(
            candidate_model_sha=candidate_model_sha,
            reference_model_sha=reference_sha,
            original_model_sha=original_model_sha,
            config_sha=config_sha,
            schedule_sha=schedule_sha,
            policy_sha=policy_sha,
            comparison=comparison,
            suite=cell_suite,
            stage=stage,
            look=look,
            topology=topology,
            suite_manifest_sha=suite_manifest_sha,
            suite_bank_sha=suite_bank_sha,
            schedule_id=schedule_id,
        )
        for (
            comparison,
            cell_suite,
            reference_sha,
            config_sha,
            schedule_sha,
            suite_bank_sha,
            schedule_id,
        ) in coordinates
    )


class EvaluationRunner:
    """Execute independent schedule shards and publish one complete result bundle."""

    def __init__(
        self,
        *,
        katago_binary: Path,
        config_path: Path,
        output_root: Path,
        shard_count: int = 1,
        max_parallel: int = 1,
        max_attempts: int = 2,
        include_move_traces: bool = False,
        env: Optional[Mapping[str, str]] = None,
        replace_env: bool = False,
        cwd: Optional[Path] = None,
        timeout: Optional[float] = None,
        extra_args: Sequence[str] = (),
        subprocess_runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        if type(shard_count) is not int or shard_count <= 0:
            raise ValueError("shard_count must be a positive integer")
        if type(max_parallel) is not int or max_parallel <= 0:
            raise ValueError("max_parallel must be a positive integer")
        if type(max_attempts) is not int or max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be positive")
        if not callable(subprocess_runner):
            raise ValueError("subprocess_runner must be callable")

        overrides = dict(env or {})
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in overrides.items()
        ):
            raise ValueError("environment keys and values must be strings")
        environment = {} if replace_env else dict(os.environ)
        environment.update(overrides)

        self.katago_binary = Path(katago_binary)
        self.config_path = Path(config_path)
        self.output_root = Path(output_root)
        self.shard_count = shard_count
        self.max_parallel = max_parallel
        self.max_attempts = max_attempts
        self.include_move_traces = include_move_traces
        self.replace_env = replace_env
        self.env_overrides = tuple(sorted(overrides.items()))
        self.environment = environment
        self.cwd = (Path(cwd) if cwd is not None else Path.cwd()).resolve()
        self.timeout = float(timeout) if timeout is not None else None
        self.extra_args = tuple(extra_args)
        self.subprocess_runner = subprocess_runner

    def _paths_for_shard(
        self,
        partial_dir: Path,
        shard: ScheduleShard,
        attempt: Optional[int] = None,
    ) -> Tuple[Path, Path, Optional[Path]]:
        schedule_path = partial_dir / f"shard-{shard.index:03d}.schedule.jsonl"
        if attempt is None:
            result_path = partial_dir / f"shard-{shard.index:03d}.results.jsonl"
            move_path = (
                partial_dir / f"shard-{shard.index:03d}.moves.jsonl"
                if self.include_move_traces
                else None
            )
        else:
            result_path = (
                partial_dir
                / f"shard-{shard.index:03d}.attempt-{attempt:03d}.results.jsonl"
            )
            move_path = (
                partial_dir
                / f"shard-{shard.index:03d}.attempt-{attempt:03d}.moves.jsonl"
                if self.include_move_traces
                else None
            )
        return schedule_path, result_path, move_path

    def _command_plan(
        self,
        partial_dir: Path,
        shard: ScheduleShard,
        candidate_model_path: Path,
        reference_model_path: Path,
        attempt: int,
    ) -> CommandPlan:
        schedule_path, result_path, move_path = self._paths_for_shard(
            partial_dir, shard, attempt
        )
        return CommandPlan(
            shard_index=shard.index,
            attempt=attempt,
            argv=build_match_command(
                self.katago_binary,
                self.config_path,
                candidate_model_path,
                reference_model_path,
                schedule_path,
                result_path,
                game_count=len(shard.rows),
                move_path=move_path,
                extra_args=self.extra_args,
            ),
            schedule_path=schedule_path,
            result_path=result_path,
            move_path=move_path,
            cwd=self.cwd,
            env_overrides=self.env_overrides,
        )

    def _verify_inputs(
        self,
        spec: EvaluationSpec,
        schedule_path: Path,
        candidate_model_path: Path,
        reference_model_path: Path,
        original_model_path: Optional[Path],
        policy_path: Optional[Path],
        suite_manifest_path: Optional[Path],
    ) -> str:
        if original_model_path is None:
            raise EvaluationValidationError(
                "original_model_path is required for every promotion evaluation"
            )
        if policy_path is None:
            raise EvaluationValidationError(
                "policy_path is required for every promotion evaluation"
            )
        try:
            binary_sha = file_sha256(self.katago_binary)
        except OSError as exc:
            raise EvaluationValidationError(
                f"cannot hash KataGo binary {self.katago_binary}: {exc}"
            ) from exc
        _verify_file_hash(
            candidate_model_path, spec.candidate_model_sha, "candidate model"
        )
        _verify_file_hash(
            reference_model_path, spec.reference_model_sha, "reference model"
        )
        _verify_file_hash(self.config_path, spec.config_sha, "match config")
        _verify_file_hash(schedule_path, spec.schedule_sha, "schedule")
        _verify_file_hash(
            original_model_path, spec.original_model_sha, "original model"
        )
        _verify_canonical_json_hash(policy_path, spec.policy_sha, "promotion policy")
        if spec.suite_manifest_sha is not None:
            if suite_manifest_path is None:
                raise EvaluationValidationError(
                    "suite_manifest_path is required by EvaluationSpec"
                )
            _verify_file_hash(
                suite_manifest_path,
                spec.suite_manifest_sha,
                "evaluation suite manifest",
            )
        elif suite_manifest_path is not None:
            raise EvaluationValidationError(
                "suite_manifest_path requires suite_manifest_sha in EvaluationSpec"
            )
        return binary_sha

    def _execution_identity(
        self, binary_sha: str, shards: Sequence[ScheduleShard]
    ) -> RunnerExecutionIdentity:
        effective_shards = len(shards)
        return RunnerExecutionIdentity(
            katago_binary_sha=binary_sha,
            move_traces=self.include_move_traces,
            extra_argv=self.extra_args,
            effective_shard_count=effective_shards,
            effective_max_parallelism=min(self.max_parallel, effective_shards),
            max_attempts=self.max_attempts,
            cwd=str(self.cwd),
            timeout=self.timeout,
            replace_env=self.replace_env,
            environment_sha=canonical_sha256(dict(sorted(self.environment.items()))),
        )

    def _verify_suite_binding(
        self,
        spec: EvaluationSpec,
        suite_manifest_path: Optional[Path],
        schedule_path: Path,
        schedule_rows: Sequence[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if spec.suite_manifest_sha is None:
            return None
        assert suite_manifest_path is not None
        manifest = load_suite_manifest(suite_manifest_path)
        exact_cells = list(_iter_manifest_cells(manifest.get("cells")))
        if exact_cells:
            matching_exact = [
                cell
                for cell in exact_cells
                if cell.get("stage") == spec.stage
                and cell.get("look") == spec.look
                and cell.get("comparison") == spec.comparison
                and cell.get("suite") == spec.suite
            ]
            if len(matching_exact) > 1:
                raise EvaluationValidationError(
                    "suite manifest has duplicate exact evaluation coordinates"
                )
            if not matching_exact and spec.stage == "stage-3":
                raise EvaluationValidationError(
                    "suite manifest has no exact Stage-3 evaluation cell"
                )
        else:
            matching_exact = []
        if matching_exact:
            cell = resolve_manifest_cell(
                manifest,
                stage=spec.stage,
                look=spec.look,
                comparison=spec.comparison,
                suite=spec.suite,
            )
            if manifest.get("policy_hash") != spec.policy_sha:
                raise EvaluationValidationError(
                    "suite manifest policy hash contradicts EvaluationSpec"
                )
            expected_schedule_path = (
                suite_manifest_path.parent / cell["schedule_path"]
            ).resolve()
            if schedule_path.resolve() != expected_schedule_path:
                raise EvaluationValidationError(
                    "schedule path does not match exact suite manifest cell"
                )
            expected_values = {
                "schedule_hash": spec.schedule_sha,
                "schedule_id": schedule_rows[0]["scheduleId"],
                "schedule_row_count": len(schedule_rows),
                "color_pairs": len({row["pairId"] for row in schedule_rows}),
                "bank_hash": spec.suite_bank_sha,
            }
            actual_values = {key: cell.get(key) for key in expected_values}
            if actual_values != expected_values:
                raise EvaluationValidationError(
                    "suite manifest contradicts exact schedule cell: "
                    f"expected {expected_values}, got {actual_values}"
                )
            if spec.schedule_id is None:
                raise EvaluationValidationError(
                    "exact suite manifest cells require schedule_id in EvaluationSpec"
                )
            if spec.schedule_id != cell["schedule_id"]:
                raise EvaluationValidationError(
                    "EvaluationSpec schedule_id contradicts manifest cell"
                )

            pairs: Dict[str, Mapping[str, Any]] = {}
            for row in schedule_rows:
                pairs.setdefault(row["pairId"], row)
            cluster_ids = [
                row.get(
                    "independentClusterId",
                    row.get("positionSemanticSha256"),
                )
                for row in pairs.values()
            ]
            position_ids = sorted(row["positionId"] for row in pairs.values())
            if cluster_ids != cell["independent_cluster_ids"]:
                raise EvaluationValidationError(
                    "schedule independent clusters contradict manifest cell"
                )
            if canonical_sha256(cluster_ids) != cell["independent_cluster_ids_hash"]:
                raise EvaluationValidationError(
                    "schedule independent cluster hash contradicts manifest cell"
                )
            if position_ids != cell["position_ids"]:
                raise EvaluationValidationError(
                    "schedule position IDs contradict manifest cell"
                )
            if canonical_sha256(position_ids) != cell["position_ids_hash"]:
                raise EvaluationValidationError(
                    "schedule position ID hash contradicts manifest cell"
                )
            pair_counts_by_cluster: Dict[str, int] = {}
            for cluster_id in cluster_ids:
                pair_counts_by_cluster[cluster_id] = (
                    pair_counts_by_cluster.get(cluster_id, 0) + 1
                )
            if len(cluster_ids) < cell["minimum_independent_position_clusters"] or any(
                count != 1 for count in pair_counts_by_cluster.values()
            ):
                raise EvaluationValidationError(
                    "exact manifest cell must use one pair per independent cluster"
                )
            qualified_names = {row.get("suiteQualifiedName") for row in schedule_rows}
            if qualified_names != {cell["bank_name"]}:
                raise EvaluationValidationError(
                    "schedule qualified bank name contradicts manifest cell"
                )
            return cell

        matching_banks = [
            bank
            for bank in manifest.get("banks", [])
            if isinstance(bank, dict)
            and isinstance(bank.get("positions"), dict)
            and bank["positions"].get("sha256") == spec.suite_bank_sha
        ]
        if len(matching_banks) != 1:
            raise EvaluationValidationError(
                "suite_bank_sha must identify exactly one manifest bank"
            )
        bank = matching_banks[0]
        schedule = bank.get("schedule")
        if not isinstance(schedule, dict):
            raise EvaluationValidationError("suite bank has no schedule manifest")
        expected = {
            "sha256": spec.schedule_sha,
            "scheduleId": schedule_rows[0]["scheduleId"],
            "rowCount": len(schedule_rows),
            "pairCount": len({row["pairId"] for row in schedule_rows}),
        }
        actual = {key: schedule.get(key) for key in expected}
        if actual != expected:
            raise EvaluationValidationError(
                f"suite artifact manifest contradicts exact schedule cell: "
                f"expected {expected}, got {actual}"
            )
        if bank.get("name") != spec.suite:
            raise EvaluationValidationError(
                "EvaluationSpec suite does not match suite manifest bank"
            )
        return None

    def plan(
        self,
        spec: EvaluationSpec,
        schedule_path: Path,
        candidate_model_path: Path,
        reference_model_path: Path,
        *,
        original_model_path: Optional[Path] = None,
        policy_path: Optional[Path] = None,
        suite_manifest_path: Optional[Path] = None,
        verify_hashes: bool = True,
    ) -> EvaluationPlan:
        """Validate inputs and return a mutation-free, shell-free command plan."""

        schedule_path = Path(schedule_path)
        candidate_model_path = Path(candidate_model_path)
        reference_model_path = Path(reference_model_path)
        if not verify_hashes:
            raise EvaluationValidationError(
                "verify_hashes=False is unsafe and unsupported for promotion"
            )
        binary_sha = self._verify_inputs(
            spec,
            schedule_path,
            candidate_model_path,
            reference_model_path,
            Path(original_model_path) if original_model_path is not None else None,
            Path(policy_path) if policy_path is not None else None,
            Path(suite_manifest_path) if suite_manifest_path is not None else None,
        )
        schedule_rows = load_schedule(schedule_path)
        scheduled_bot_indices = {
            row[field] for row in schedule_rows for field in ("blackBot", "whiteBot")
        }
        if scheduled_bot_indices != {0, 1}:
            raise EvaluationValidationError(
                "evaluation runner schedules must use bot indices 0 and 1"
            )
        manifest_cell = self._verify_suite_binding(
            spec,
            Path(suite_manifest_path) if suite_manifest_path is not None else None,
            schedule_path,
            schedule_rows,
        )
        if (
            spec.schedule_id is not None
            and spec.schedule_id != schedule_rows[0]["scheduleId"]
        ):
            raise EvaluationValidationError(
                "EvaluationSpec schedule_id contradicts schedule"
            )
        schedule_bank_hashes = {
            row.get("suiteBankSha256")
            for row in schedule_rows
            if row.get("suiteBankSha256") is not None
        }
        if spec.suite_bank_sha is not None:
            if schedule_bank_hashes != {spec.suite_bank_sha}:
                raise EvaluationValidationError(
                    "EvaluationSpec suite_bank_sha contradicts schedule"
                )
        elif schedule_bank_hashes:
            raise EvaluationValidationError(
                "bound suite schedule requires suite_bank_sha in EvaluationSpec"
            )
        if schedule_bank_hashes and spec.suite_manifest_sha is None:
            raise EvaluationValidationError(
                "bound suite schedule requires suite_manifest_sha in EvaluationSpec"
            )
        shards = shard_schedule(schedule_rows, self.shard_count)
        execution = self._execution_identity(binary_sha, shards)
        key = compute_execution_key(spec, execution)
        partial_dir = self.output_root / "partial" / key
        final_dir = self.output_root / "final" / key
        commands = tuple(
            self._command_plan(
                partial_dir,
                shard,
                candidate_model_path,
                reference_model_path,
                1,
            )
            for shard in shards
        )
        return EvaluationPlan(
            spec=spec,
            evaluation_key=key,
            schedule_path=schedule_path,
            schedule_id=schedule_rows[0]["scheduleId"],
            execution=execution,
            schedule_rows=schedule_rows,
            manifest_cell=manifest_cell,
            shards=shards,
            commands=commands,
            partial_dir=partial_dir,
            final_dir=final_dir,
        )

    def _partial_identity(self, plan: EvaluationPlan) -> Dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "runnerContract": RUNNER_CONTRACT,
            "evaluationKey": plan.evaluation_key,
            "evaluationSpec": plan.spec.to_dict(),
            "execution": plan.execution.to_dict(),
            "schedule": {
                "sha256": plan.spec.schedule_sha,
                "scheduleId": plan.schedule_id,
                "rowCount": len(plan.schedule_rows),
                "pairCount": len({row["pairId"] for row in plan.schedule_rows}),
                "suiteManifestSha256": plan.spec.suite_manifest_sha,
                "suiteBankSha256": plan.spec.suite_bank_sha,
                "manifestCell": plan.manifest_cell,
                "manifestCellSha256": (
                    canonical_sha256(plan.manifest_cell)
                    if plan.manifest_cell is not None
                    else None
                ),
            },
            "shards": [
                {
                    "index": shard.index,
                    "gameIds": list(shard.game_ids),
                    "pairIds": list(shard.pair_ids),
                }
                for shard in plan.shards
            ],
        }

    def _publish_shard_result(
        self,
        plan: EvaluationPlan,
        shard: ScheduleShard,
        attempt: int,
        attempt_result_path: Path,
        attempt_move_path: Optional[Path],
    ) -> ShardResult:
        result_rows = validate_result_jsonl(attempt_result_path, shard.rows)
        move_rows = (
            validate_move_jsonl(attempt_move_path, shard.rows, result_rows)
            if attempt_move_path is not None
            else None
        )
        schedule_path, result_path, move_path = self._paths_for_shard(
            plan.partial_dir, shard
        )
        _atomic_publish_bytes(
            result_path,
            _canonical_jsonl(result_rows),
            replace_conflicting=True,
        )
        if move_rows is not None and move_path is not None:
            _atomic_publish_bytes(
                move_path,
                _canonical_jsonl(move_rows),
                replace_conflicting=True,
            )
        result_hash = file_sha256(result_path)
        move_hash = file_sha256(move_path) if move_path is not None else None
        complete = {
            "schemaVersion": SCHEMA_VERSION,
            "runnerContract": RUNNER_CONTRACT,
            "evaluationKey": plan.evaluation_key,
            "shardIndex": shard.index,
            "attempts": attempt,
            "scheduleSha256": file_sha256(schedule_path),
            "resultSha256": result_hash,
            "moveSha256": move_hash,
            "gameIds": list(shard.game_ids),
            "pairIds": list(shard.pair_ids),
        }
        _atomic_publish_bytes(
            plan.partial_dir / f"shard-{shard.index:03d}.complete.json",
            (canonical_json(complete) + "\n").encode("utf-8"),
            replace_conflicting=True,
        )
        return ShardResult(
            shard_index=shard.index,
            attempts=attempt,
            schedule_path=schedule_path,
            result_path=result_path,
            move_path=move_path,
            game_ids=shard.game_ids,
            pair_ids=shard.pair_ids,
            result_sha256=result_hash,
            move_sha256=move_hash,
            reused=False,
        )

    def _recover_complete_shard(
        self, plan: EvaluationPlan, shard: ScheduleShard
    ) -> Optional[ShardResult]:
        schedule_path, result_path, move_path = self._paths_for_shard(
            plan.partial_dir, shard
        )
        if not result_path.is_file() or (
            move_path is not None and not move_path.is_file()
        ):
            return None
        try:
            result_rows = validate_result_jsonl(result_path, shard.rows)
            canonical_result_data = _canonical_jsonl(result_rows)
            if result_path.read_bytes() != canonical_result_data:
                _atomic_publish_bytes(
                    result_path,
                    canonical_result_data,
                    replace_conflicting=True,
                )
            if move_path is not None:
                move_rows = validate_move_jsonl(move_path, shard.rows, result_rows)
                canonical_move_data = _canonical_jsonl(move_rows)
                if move_path.read_bytes() != canonical_move_data:
                    _atomic_publish_bytes(
                        move_path,
                        canonical_move_data,
                        replace_conflicting=True,
                    )
        except (OSError, EvaluationValidationError):
            return None

        attempts = 0
        complete_path = plan.partial_dir / f"shard-{shard.index:03d}.complete.json"
        if complete_path.is_file():
            try:
                complete = json.loads(complete_path.read_text(encoding="utf-8"))
                if (
                    complete.get("evaluationKey") == plan.evaluation_key
                    and complete.get("gameIds") == list(shard.game_ids)
                    and complete.get("resultSha256") == file_sha256(result_path)
                    and complete.get("moveSha256")
                    == (file_sha256(move_path) if move_path is not None else None)
                ):
                    attempts = int(complete.get("attempts", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                attempts = 0
        if attempts < 0:
            attempts = 0
        result_hash = file_sha256(result_path)
        move_hash = file_sha256(move_path) if move_path is not None else None
        complete = {
            "schemaVersion": SCHEMA_VERSION,
            "runnerContract": RUNNER_CONTRACT,
            "evaluationKey": plan.evaluation_key,
            "shardIndex": shard.index,
            "attempts": attempts,
            "scheduleSha256": file_sha256(schedule_path),
            "resultSha256": result_hash,
            "moveSha256": move_hash,
            "gameIds": list(shard.game_ids),
            "pairIds": list(shard.pair_ids),
        }
        _atomic_publish_bytes(
            complete_path,
            (canonical_json(complete) + "\n").encode("utf-8"),
            replace_conflicting=True,
        )
        return ShardResult(
            shard_index=shard.index,
            attempts=attempts,
            schedule_path=schedule_path,
            result_path=result_path,
            move_path=move_path,
            game_ids=shard.game_ids,
            pair_ids=shard.pair_ids,
            result_sha256=result_hash,
            move_sha256=move_hash,
            reused=True,
        )

    def _record_attempt_failure(
        self,
        plan: EvaluationPlan,
        shard: ScheduleShard,
        attempt: int,
        error: str,
        returncode: Optional[int],
    ) -> None:
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "evaluationKey": plan.evaluation_key,
            "shardIndex": shard.index,
            "attempt": attempt,
            "returncode": returncode,
            "error": error[:4000],
        }
        _atomic_publish_bytes(
            plan.partial_dir
            / f"shard-{shard.index:03d}.attempt-{attempt:03d}.failure.json",
            (canonical_json(payload) + "\n").encode("utf-8"),
            replace_conflicting=True,
        )

    def _run_shard(
        self,
        plan: EvaluationPlan,
        shard: ScheduleShard,
        candidate_model_path: Path,
        reference_model_path: Path,
    ) -> ShardResult:
        schedule_path, _, _ = self._paths_for_shard(plan.partial_dir, shard)
        _atomic_publish_bytes(schedule_path, _canonical_jsonl(shard.rows))
        recovered = self._recover_complete_shard(plan, shard)
        if recovered is not None:
            return recovered

        errors: List[str] = []
        for attempt in range(1, self.max_attempts + 1):
            command = self._command_plan(
                plan.partial_dir,
                shard,
                candidate_model_path,
                reference_model_path,
                attempt,
            )
            failure_path = (
                plan.partial_dir
                / f"shard-{shard.index:03d}.attempt-{attempt:03d}.failure.json"
            )
            if failure_path.exists():
                errors.append(f"attempt {attempt} was previously recorded as failed")
                continue

            result_exists = command.result_path.exists()
            move_exists = command.move_path is not None and command.move_path.exists()
            complete_attempt_exists = result_exists and (
                command.move_path is None or move_exists
            )
            any_attempt_artifact_exists = result_exists or move_exists
            if complete_attempt_exists:
                try:
                    result = self._publish_shard_result(
                        plan,
                        shard,
                        attempt,
                        command.result_path,
                        command.move_path,
                    )
                    return replace(result, reused=True)
                except (OSError, EvaluationValidationError) as exc:
                    error = _failure_text(exc)
                    errors.append(error)
                    self._record_attempt_failure(
                        plan, shard, attempt, error, returncode=None
                    )
                    continue
            if any_attempt_artifact_exists:
                error = "incomplete pre-existing attempt artifacts"
                errors.append(error)
                self._record_attempt_failure(
                    plan, shard, attempt, error, returncode=None
                )
                continue

            try:
                completed = self.subprocess_runner(
                    list(command.argv),
                    env=dict(self.environment),
                    cwd=str(command.cwd) if command.cwd is not None else None,
                    timeout=self.timeout,
                    capture_output=True,
                    text=True,
                    shell=False,
                )
            except Exception as exc:
                error = _failure_text(exc)
                errors.append(error)
                self._record_attempt_failure(
                    plan, shard, attempt, error, returncode=None
                )
                continue

            returncode = getattr(completed, "returncode", None)
            if type(returncode) is not int or returncode != 0:
                error = f"katago match returned {returncode!r}"
                stderr = getattr(completed, "stderr", "")
                if isinstance(stderr, str) and stderr:
                    error += f": {stderr[:1000]}"
                errors.append(error)
                self._record_attempt_failure(
                    plan, shard, attempt, error, returncode=returncode
                )
                continue
            try:
                return self._publish_shard_result(
                    plan,
                    shard,
                    attempt,
                    command.result_path,
                    command.move_path,
                )
            except (OSError, EvaluationValidationError) as exc:
                error = _failure_text(exc)
                errors.append(error)
                self._record_attempt_failure(
                    plan, shard, attempt, error, returncode=returncode
                )

        raise EvaluationError(
            f"shard {shard.index} failed after {self.max_attempts} attempts: "
            + "; ".join(errors)
        )

    def _final_manifest(
        self,
        plan: EvaluationPlan,
        shard_results: Sequence[ShardResult],
        result_data: bytes,
        move_data: Optional[bytes],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "runnerContract": RUNNER_CONTRACT,
            "evaluationKey": plan.evaluation_key,
            "evaluationSpec": plan.spec.to_dict(),
            "execution": plan.execution.to_dict(),
            "cell": {
                "comparison": plan.spec.comparison,
                "suite": plan.spec.suite,
                "stage": plan.spec.stage,
                "look": plan.spec.look,
                "gameCount": len(plan.schedule_rows),
                "colorPairCount": len({row["pairId"] for row in plan.schedule_rows}),
            },
            "schedule": {
                "sha256": plan.spec.schedule_sha,
                "scheduleId": plan.schedule_id,
                "rowCount": len(plan.schedule_rows),
                "pairCount": len({row["pairId"] for row in plan.schedule_rows}),
                "suiteManifestSha256": plan.spec.suite_manifest_sha,
                "suiteBankSha256": plan.spec.suite_bank_sha,
                "manifestCell": plan.manifest_cell,
                "manifestCellSha256": (
                    canonical_sha256(plan.manifest_cell)
                    if plan.manifest_cell is not None
                    else None
                ),
            },
            "results": {
                "path": "results.jsonl",
                "sha256": hashlib.sha256(result_data).hexdigest(),
                "rowCount": len(plan.schedule_rows),
            },
            "moves": (
                {
                    "path": "moves.jsonl",
                    "sha256": hashlib.sha256(move_data).hexdigest(),
                    "rowCount": len(move_data.decode("utf-8").splitlines()),
                }
                if move_data is not None
                else None
            ),
            "shards": [
                {
                    "index": result.shard_index,
                    "gameIds": list(result.game_ids),
                    "pairIds": list(result.pair_ids),
                    "scheduleSha256": file_sha256(result.schedule_path),
                    "resultSha256": result.result_sha256,
                    "moveSha256": result.move_sha256,
                }
                for result in sorted(shard_results, key=lambda item: item.shard_index)
            ],
        }
        manifest = dict(payload)
        manifest["manifestPayloadSha256"] = canonical_sha256(payload)
        return manifest

    def _reconcile_final(self, plan: EvaluationPlan) -> Optional[EvaluationResult]:
        final_dir = plan.final_dir
        if not final_dir.exists():
            return None
        if not final_dir.is_dir():
            raise EvaluationConflictError(
                f"final output is not a directory: {final_dir}"
            )
        manifest_path = final_dir / "manifest.json"
        result_path = final_dir / "results.jsonl"
        move_path = final_dir / "moves.jsonl" if self.include_move_traces else None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationConflictError(
                f"cannot reconcile final evaluation {final_dir}: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise EvaluationConflictError("final manifest must be an object")
        payload = dict(manifest)
        manifest_payload_sha = payload.pop("manifestPayloadSha256", None)
        if manifest_payload_sha != canonical_sha256(payload):
            raise EvaluationConflictError("final manifest payload SHA-256 is invalid")
        if (
            manifest.get("runnerContract") != RUNNER_CONTRACT
            or manifest.get("evaluationKey") != plan.evaluation_key
            or manifest.get("evaluationSpec") != plan.spec.to_dict()
            or manifest.get("execution") != plan.execution.to_dict()
        ):
            raise EvaluationConflictError("final manifest contradicts EvaluationSpec")
        expected_cell = {
            "comparison": plan.spec.comparison,
            "suite": plan.spec.suite,
            "stage": plan.spec.stage,
            "look": plan.spec.look,
            "gameCount": len(plan.schedule_rows),
            "colorPairCount": len({row["pairId"] for row in plan.schedule_rows}),
        }
        if manifest.get("cell") != expected_cell:
            raise EvaluationConflictError(
                "final manifest contradicts evaluation cell counts"
            )
        schedule_manifest = manifest.get("schedule")
        if (
            not isinstance(schedule_manifest, dict)
            or schedule_manifest.get("sha256") != plan.spec.schedule_sha
            or schedule_manifest.get("scheduleId") != plan.schedule_id
            or schedule_manifest.get("rowCount") != len(plan.schedule_rows)
            or schedule_manifest.get("pairCount")
            != len({row["pairId"] for row in plan.schedule_rows})
            or schedule_manifest.get("suiteManifestSha256")
            != plan.spec.suite_manifest_sha
            or schedule_manifest.get("suiteBankSha256") != plan.spec.suite_bank_sha
            or schedule_manifest.get("manifestCell") != plan.manifest_cell
            or schedule_manifest.get("manifestCellSha256")
            != (
                canonical_sha256(plan.manifest_cell)
                if plan.manifest_cell is not None
                else None
            )
        ):
            raise EvaluationConflictError("final manifest contradicts schedule")
        results_manifest = manifest.get("results")
        if (
            not isinstance(results_manifest, dict)
            or results_manifest.get("path") != "results.jsonl"
            or not result_path.is_file()
            or results_manifest.get("sha256") != file_sha256(result_path)
            or results_manifest.get("rowCount") != len(plan.schedule_rows)
        ):
            raise EvaluationConflictError("final result artifact contradicts manifest")
        result_rows = validate_result_jsonl(result_path, plan.schedule_rows)
        if result_path.read_bytes() != _canonical_jsonl(result_rows):
            raise EvaluationConflictError(
                "final result artifact is not canonical finalized JSONL"
            )
        moves_manifest = manifest.get("moves")
        if self.include_move_traces:
            if (
                not isinstance(moves_manifest, dict)
                or moves_manifest.get("path") != "moves.jsonl"
                or move_path is None
                or not move_path.is_file()
                or moves_manifest.get("sha256") != file_sha256(move_path)
            ):
                raise EvaluationConflictError(
                    "final move artifact contradicts manifest"
                )
            move_rows = validate_move_jsonl(move_path, plan.schedule_rows, result_rows)
            if move_path.read_bytes() != _canonical_jsonl(move_rows):
                raise EvaluationConflictError(
                    "final move artifact is not canonical validated JSONL"
                )
            if moves_manifest.get("rowCount") != len(move_rows):
                raise EvaluationConflictError(
                    "final move row count contradicts manifest"
                )
        elif moves_manifest is not None:
            raise EvaluationConflictError("unexpected move artifact in final manifest")

        shard_results: List[ShardResult] = []
        manifest_shards = manifest.get("shards")
        if not isinstance(manifest_shards, list) or len(manifest_shards) != len(
            plan.shards
        ):
            raise EvaluationConflictError("final manifest has wrong shard count")
        by_index = {
            entry.get("index"): entry
            for entry in manifest_shards
            if isinstance(entry, dict)
        }
        for shard in plan.shards:
            entry = by_index.get(shard.index)
            if (
                entry is None
                or entry.get("gameIds") != list(shard.game_ids)
                or entry.get("pairIds") != list(shard.pair_ids)
            ):
                raise EvaluationConflictError(
                    "final manifest has contradictory shard IDs"
                )
            schedule_path, partial_result_path, partial_move_path = (
                self._paths_for_shard(plan.partial_dir, shard)
            )
            shard_results.append(
                ShardResult(
                    shard_index=shard.index,
                    attempts=0,
                    schedule_path=schedule_path,
                    result_path=partial_result_path,
                    move_path=partial_move_path,
                    game_ids=shard.game_ids,
                    pair_ids=shard.pair_ids,
                    result_sha256=entry.get("resultSha256"),
                    move_sha256=entry.get("moveSha256"),
                    reused=True,
                )
            )
        return EvaluationResult(
            spec=plan.spec,
            evaluation_key=plan.evaluation_key,
            final_dir=final_dir,
            result_path=result_path,
            move_path=move_path,
            manifest_path=manifest_path,
            manifest_sha256=file_sha256(manifest_path),
            shards=tuple(shard_results),
            reused=True,
        )

    def _merge_and_publish(
        self, plan: EvaluationPlan, shard_results: Sequence[ShardResult]
    ) -> EvaluationResult:
        result_by_game: Dict[str, Dict[str, Any]] = {}
        move_rows: List[Dict[str, Any]] = []
        ordered_shard_results = sorted(shard_results, key=lambda item: item.shard_index)
        for index, shard in enumerate(plan.shards):
            shard_result = ordered_shard_results[index]
            validated_results = validate_result_jsonl(
                shard_result.result_path, shard.rows
            )
            result_by_game.update({row["gameId"]: row for row in validated_results})
            if shard_result.move_path is not None:
                move_rows.extend(
                    validate_move_jsonl(
                        shard_result.move_path, shard.rows, validated_results
                    )
                )
        merged_results = tuple(
            result_by_game[row["gameId"]] for row in plan.schedule_rows
        )
        merged_results = validate_result_rows(merged_results, plan.schedule_rows)
        result_data = _canonical_jsonl(merged_results)
        move_data: Optional[bytes] = None
        if self.include_move_traces:
            merged_moves = validate_move_rows(
                move_rows, plan.schedule_rows, merged_results
            )
            move_data = _canonical_jsonl(merged_moves)

        manifest = self._final_manifest(plan, shard_results, result_data, move_data)
        manifest_data = (canonical_json(manifest) + "\n").encode("utf-8")
        final_root = plan.final_dir.parent
        final_root.mkdir(parents=True, exist_ok=True)
        temporary: Optional[Path] = Path(
            tempfile.mkdtemp(
                prefix=f".{plan.evaluation_key}.partial-", dir=str(final_root)
            )
        )
        try:
            assert temporary is not None
            _write_new_fsynced(temporary / "results.jsonl", result_data)
            if move_data is not None:
                _write_new_fsynced(temporary / "moves.jsonl", move_data)
            _write_new_fsynced(temporary / "manifest.json", manifest_data)
            _fsync_directory(temporary)
            try:
                os.rename(str(temporary), str(plan.final_dir))
                _fsync_directory(final_root)
                temporary = None
            except OSError as exc:
                if not plan.final_dir.exists():
                    raise
                existing = self._reconcile_final(plan)
                if existing is None:
                    raise EvaluationConflictError(
                        f"failed to publish or reconcile {plan.final_dir}"
                    ) from exc
                return existing
        finally:
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary)

        return EvaluationResult(
            spec=plan.spec,
            evaluation_key=plan.evaluation_key,
            final_dir=plan.final_dir,
            result_path=plan.final_dir / "results.jsonl",
            move_path=(
                plan.final_dir / "moves.jsonl" if self.include_move_traces else None
            ),
            manifest_path=plan.final_dir / "manifest.json",
            manifest_sha256=hashlib.sha256(manifest_data).hexdigest(),
            shards=tuple(sorted(shard_results, key=lambda item: item.shard_index)),
            reused=False,
        )

    def run(
        self,
        spec: EvaluationSpec,
        schedule_path: Path,
        candidate_model_path: Path,
        reference_model_path: Path,
        *,
        original_model_path: Optional[Path] = None,
        policy_path: Optional[Path] = None,
        suite_manifest_path: Optional[Path] = None,
        verify_hashes: bool = True,
        dry_run: bool = False,
    ) -> Union[EvaluationPlan, EvaluationResult]:
        """Run or dry-plan an evaluation, reconciling complete prior output."""

        candidate_model_path = Path(candidate_model_path)
        reference_model_path = Path(reference_model_path)
        plan = self.plan(
            spec,
            schedule_path,
            candidate_model_path,
            reference_model_path,
            original_model_path=original_model_path,
            policy_path=policy_path,
            suite_manifest_path=suite_manifest_path,
            verify_hashes=verify_hashes,
        )
        if dry_run:
            return plan

        already_final = self._reconcile_final(plan)
        if already_final is not None:
            return already_final

        plan.partial_dir.mkdir(parents=True, exist_ok=True)
        identity_data = (canonical_json(self._partial_identity(plan)) + "\n").encode(
            "utf-8"
        )
        _atomic_publish_bytes(plan.partial_dir / "evaluation.json", identity_data)
        for shard in plan.shards:
            schedule_file, _, _ = self._paths_for_shard(plan.partial_dir, shard)
            _atomic_publish_bytes(schedule_file, _canonical_jsonl(shard.rows))
        _fsync_directory(plan.partial_dir)

        shard_results: List[ShardResult] = []
        errors: List[Exception] = []
        workers = min(self.max_parallel, len(plan.shards))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    self._run_shard,
                    plan,
                    shard,
                    candidate_model_path,
                    reference_model_path,
                )
                for shard in plan.shards
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    shard_results.append(future.result())
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise EvaluationError(
                "evaluation did not finalize because shard execution failed: "
                + "; ".join(_failure_text(error) for error in errors)
            ) from errors[0]
        if len(shard_results) != len(plan.shards):
            raise EvaluationError("evaluation did not produce every shard result")
        return self._merge_and_publish(plan, shard_results)


def _load_spec(path: Path) -> EvaluationSpec:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid EvaluationSpec JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: EvaluationSpec must be a JSON object")
    return EvaluationSpec.from_dict(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pair-safe deterministic KataGo evaluation."
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--katago", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--reference-model", required=True, type=Path)
    parser.add_argument("--original-model", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--suite-manifest", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--moves", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        runner = EvaluationRunner(
            katago_binary=args.katago,
            config_path=args.config,
            output_root=args.output_root,
            shard_count=args.shards,
            max_parallel=args.max_parallel,
            max_attempts=args.max_attempts,
            include_move_traces=args.moves,
        )
        outcome = runner.run(
            _load_spec(args.spec),
            args.schedule,
            args.candidate_model,
            args.reference_model,
            original_model_path=args.original_model,
            policy_path=args.policy,
            suite_manifest_path=args.suite_manifest,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, EvaluationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if isinstance(outcome, EvaluationPlan):
        print(
            json.dumps(
                {
                    "evaluationKey": outcome.evaluation_key,
                    "commands": [
                        {"shard": command.shard_index, "argv": list(command.argv)}
                        for command in outcome.commands
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "evaluationKey": outcome.evaluation_key,
                    "finalDir": str(outcome.final_dir),
                    "manifestSha256": outcome.manifest_sha256,
                    "reused": outcome.reused,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
