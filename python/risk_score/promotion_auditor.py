#!/usr/bin/env python3
"""Derive rollout and deep-audit reports from closed, hash-bound evidence.

The promotion controller owns rollout transactions and deep-audit requests.
This module is the producer on the other side of those contracts.  It never
accepts a PASS flag: rollout health is reconstructed from immutable worker
acknowledgements and their closed output trees, while canary and deep-audit
decisions are reconstructed from schedule-bound EvaluationRunner outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from risk_score.evaluation_runner import (
    EvaluationResult,
    EvaluationRunner,
    EvaluationSpec,
    load_schedule,
    load_suite_manifest,
    validate_move_jsonl,
    validate_result_jsonl,
)
from risk_score.promotion_state import (
    ControllerLock,
    canonical_json_bytes,
    canonical_sha256,
    fsync_directory,
    load_champion,
    sha256_file,
)

SCHEMA_VERSION = 1
POLICY_VERSION = "risk-seeking-checkpoint-promotion-v3"
ROLLOUT_REPORT_CONTRACT = "risk-score-rollout-health-report-v1"
CANARY_RUNNER_CONTRACT = "risk-score-canary-fresh-audit-runner-v1"
CANARY_AUDIT_CONTRACT = "risk-score-canary-fresh-audit-v1"
CANARY_STATISTICS_CONTRACT = "risk-score-canary-fresh-audit-statistics-v1"
DEEP_REQUEST_CONTRACT = "risk-score-deep-audit-request-v2"
DEEP_REPORT_CONTRACT = "risk-score-deep-audit-report-v2"
DEEP_RUNNER_CONTRACT = "risk-score-deep-audit-runner-manifest-v1"
DEEP_STATISTICS_CONTRACT = "risk-score-deep-audit-statistics-v1"
MAX_JSON_BYTES = 64 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PARTIAL_NAMES = re.compile(r"(^\.|\.partial(?:[-.]|$)|\.tmp(?:[-.]|$))")
_SAFE_GENERATION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_VALID_SGF_RESULT_RE = re.compile(
    r"^(?:[BW]\+(?:R|Resign|T|Time|[0-9]+(?:\.[0-9]+)?)|0|Draw|draw)$"
)


class PromotionAuditorError(ValueError):
    """Auditor input or evidence is unsafe, incomplete, or contradictory."""


class AuditNotReady(PromotionAuditorError):
    """A controller-authored job does not yet have all required closed inputs."""


class AuditDecisionError(PromotionAuditorError):
    """Complete evidence derived a non-PASS rollout decision."""

    def __init__(self, message: str, *, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


@dataclass(frozen=True)
class AuditorRuntime:
    """The controller-owned paths and immutable identities used by the auditor."""

    promotion_root: Path
    rollout_quarantine: Path
    admitted_selfplay: Path
    rollout_report_inbox: Path
    accepted_models: Path
    original_model_path: Path
    policy_path: Path
    powered_config_path: Path
    selfplay_config_path: Path
    suite_manifest_path: Path
    audit_schedule_path: Path
    gpu_lease_config_path: Path
    policy: Mapping[str, Any]
    original_hash: str
    policy_hash: str
    powered_config_hash: str
    suite_manifest_hash: str
    audit_schedule_hash: str
    selfplay_config_hash: str
    gpu_lease_config_hash: str
    worker_count: int
    canary_worker_count: int
    intermediate_worker_count: int
    worker_threads: int

    @classmethod
    def from_controller_runtime(cls, runtime: Any) -> "AuditorRuntime":
        """Project a strict promotion_controller.RuntimeConfig."""

        return cls(
            promotion_root=Path(runtime.promotion_root),
            rollout_quarantine=Path(runtime.rollout_quarantine),
            admitted_selfplay=Path(runtime.admitted_selfplay),
            rollout_report_inbox=Path(runtime.rollout_report_inbox),
            accepted_models=Path(runtime.accepted_models),
            original_model_path=Path(runtime.original_model_path),
            policy_path=Path(runtime.policy_path),
            powered_config_path=Path(runtime.powered_config_path),
            selfplay_config_path=Path(runtime.selfplay_config_path),
            suite_manifest_path=Path(runtime.suites) / "manifest.json",
            audit_schedule_path=Path(runtime.audit_schedule_path),
            gpu_lease_config_path=Path(runtime.gpu_lease_config_path),
            policy=json.loads(canonical_json_bytes(runtime.frozen_policy)),
            original_hash=runtime.controller.original_hash,
            policy_hash=runtime.controller.policy_hash,
            powered_config_hash=runtime.controller.powered_config_hash,
            suite_manifest_hash=runtime.controller.suite_manifest_hash,
            audit_schedule_hash=runtime.controller.audit_schedule_hash,
            selfplay_config_hash=runtime.controller.selfplay_config_hash,
            gpu_lease_config_hash=runtime.controller.gpu_lease_config_hash,
            worker_count=runtime.controller.worker_count,
            canary_worker_count=runtime.controller.canary_worker_count,
            intermediate_worker_count=runtime.controller.intermediate_worker_count,
            worker_threads=runtime.controller.worker_threads,
        )

    @property
    def transactions(self) -> Path:
        return self.promotion_root / "transactions"

    @property
    def audit_queue(self) -> Path:
        return self.promotion_root / "audits" / "queue"

    @property
    def audit_outbox(self) -> Path:
        return self.promotion_root / "audits" / "outbox"

    @property
    def audit_artifacts(self) -> Path:
        return self.promotion_root / "audits" / "artifacts"

    @property
    def canary_games(self) -> int:
        return _positive_int(
            self.policy.get("rollout", {}).get("canary_games"),
            "policy rollout canary_games",
        )

    @property
    def canary_pairs(self) -> int:
        return _positive_int(
            self.policy.get("rollout", {}).get("canary_fresh_audit_color_pairs"),
            "policy rollout canary_fresh_audit_color_pairs",
        )

    @property
    def minimum_candidate_win_rate(self) -> float:
        return _finite_number(
            self.policy.get("promotion_thresholds", {}).get(
                "powered_win_rate_vs_champion_lower_bound_strictly_above"
            ),
            "policy powered win-rate threshold",
        )

    @property
    def maximum_no_result_rate(self) -> float:
        return _finite_number(
            self.policy.get("promotion_thresholds", {}).get(
                "true_no_result_rate_strictly_below"
            ),
            "policy true no-result threshold",
        )


@dataclass(frozen=True)
class EvaluationJob:
    """One immutable comparison whose result is derived, never asserted."""

    kind: str
    job_id: str
    generation_id: str
    candidate_hash: str
    reference_hash: str
    candidate_model_path: Path
    reference_model_path: Path
    original_model_path: Path
    config_path: Path
    schedule_path: Path
    schedule_hash: str
    schedule_id: str
    suite_name: str
    suite_bank_hash: str
    color_pairs: int
    max_visits: int
    output_root: Path
    spec: EvaluationSpec
    request_hash: Optional[str] = None
    cell: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Paths returned by an executor; no decision field exists by design."""

    results_path: Path
    moves_path: Path
    source_manifest_path: Optional[Path] = None


@dataclass(frozen=True)
class EvaluationObservation:
    results_path: Path
    moves_path: Path
    source_manifest_path: Optional[Path]
    results_sha256: str
    moves_sha256: str
    source_manifest_sha256: Optional[str]
    pair_ids: Tuple[str, ...]
    game_count: int
    true_no_results: int
    true_no_result_rate: float
    candidate_win_rate: float
    minimum_candidate_win_rate: float
    safety_failures: int
    decision: str


@dataclass(frozen=True)
class RolloutHealth:
    phase: str
    worker_count: int
    game_count: int
    minimum_game_count: int
    checks: Mapping[str, bool]
    workers: Tuple[Mapping[str, Any], ...]

    @property
    def decision(self) -> str:
        return "PASS" if all(self.checks.values()) else "FAIL"


@dataclass(frozen=True)
class ProductionResult:
    kind: str
    decision: str
    output_path: Path
    output_sha256: str
    reused: bool
    artifact_paths: Tuple[Path, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "decision": self.decision,
            "output": str(self.output_path),
            "sha256": self.output_sha256,
            "reused": self.reused,
            "artifacts": [str(path) for path in self.artifact_paths],
        }


EvaluationExecutor = Callable[[EvaluationJob], EvaluationArtifacts]
LeaseFactory = Callable[[], ContextManager[Mapping[str, Any]]]


def _reject_constant(value: str) -> None:
    raise PromotionAuditorError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PromotionAuditorError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _decode_json(data: bytes, source: Path) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, PromotionAuditorError) as exc:
        raise PromotionAuditorError(f"{source}: invalid JSON: {exc}") from exc


def _load_canonical_object(path: Path, role: str) -> Dict[str, Any]:
    source = Path(path)
    try:
        metadata = source.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PromotionAuditorError(f"{role} must be a regular non-symlink file")
    if metadata.st_size > MAX_JSON_BYTES:
        raise PromotionAuditorError(f"{role} exceeds the JSON size limit")
    data = source.read_bytes()
    value = _decode_json(data, source)
    if not isinstance(value, dict):
        raise PromotionAuditorError(f"{role} must have an object root")
    if data != canonical_json_bytes(value) + b"\n":
        raise PromotionAuditorError(f"{role} must be canonical newline-terminated JSON")
    return value


def _load_canonical_jsonl(path: Path, role: str) -> Tuple[Dict[str, Any], ...]:
    source = Path(path)
    try:
        metadata = source.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PromotionAuditorError(f"{role} must be a regular non-symlink file")
    if metadata.st_size > MAX_JSON_BYTES:
        raise PromotionAuditorError(f"{role} exceeds the JSONL size limit")
    data = source.read_bytes()
    if not data or not data.endswith(b"\n"):
        raise PromotionAuditorError(f"{role} must be nonempty and newline-terminated")
    rows: List[Dict[str, Any]] = []
    for line_number, raw in enumerate(data.splitlines(), start=1):
        if not raw:
            raise PromotionAuditorError(
                f"{role} contains a blank row at line {line_number}"
            )
        value = _decode_json(raw, source)
        if not isinstance(value, dict):
            raise PromotionAuditorError(f"{role} line {line_number} must be an object")
        rows.append(value)
    expected = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    if data != expected:
        raise PromotionAuditorError(f"{role} must be canonical JSONL")
    return tuple(rows)


def _sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PromotionAuditorError(f"{role} must be a lowercase 64-character SHA-256")
    return value


def _nonempty(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise PromotionAuditorError(f"{role} must be a nonempty single-line string")
    return value


def _positive_int(value: Any, role: str) -> int:
    if type(value) is not int or value <= 0:
        raise PromotionAuditorError(f"{role} must be a positive integer")
    return value


def _finite_number(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromotionAuditorError(f"{role} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise PromotionAuditorError(f"{role} must be finite")
    return result


def _safe_generation_id(value: Any) -> str:
    generation_id = _nonempty(value, "generation_id")
    if _SAFE_GENERATION_RE.fullmatch(generation_id) is None or generation_id in {
        ".",
        "..",
    }:
        raise PromotionAuditorError("generation_id is not a safe path component")
    return generation_id


def _require_absolute_under(path: Path, root: Path, role: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise PromotionAuditorError(f"{role} path must be absolute")
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(Path(root).resolve())
    except (OSError, ValueError) as exc:
        raise PromotionAuditorError(f"{role} path is outside its frozen root") from exc
    if resolved != candidate:
        raise PromotionAuditorError(f"{role} path must be normalized and symlink-free")
    return candidate


def _regular_file(path: Path, expected_hash: str, role: str) -> Path:
    source = Path(path)
    try:
        metadata = source.lstat()
    except FileNotFoundError as exc:
        raise PromotionAuditorError(f"{role} is missing: {source}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PromotionAuditorError(f"{role} must be a regular non-symlink file")
    if sha256_file(source) != expected_hash:
        raise PromotionAuditorError(f"{role} hash contradicts its frozen identity")
    return source


def _ensure_directory(path: Path, root: Path) -> Path:
    target = _require_absolute_under(Path(path), Path(root), "output directory")
    if target.exists():
        if target.is_symlink() or not target.is_dir():
            raise PromotionAuditorError(f"output path is not a directory: {target}")
        return target
    parent = target.parent
    _ensure_directory(parent, root) if parent != root else None
    target.mkdir()
    fsync_directory(parent)
    return target


def publish_canonical_json(path: Path, value: Mapping[str, Any]) -> bool:
    """Atomically create immutable canonical JSON; exact retries are success."""

    destination = Path(path)
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise PromotionAuditorError(
            f"output parent must be an existing non-symlink directory: "
            f"{destination.parent}"
        )
    data = canonical_json_bytes(value) + b"\n"
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != data
        ):
            raise PromotionAuditorError(
                f"immutable output conflicts with requested publication: {destination}"
            )
        return True
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.read_bytes() != data
            ):
                raise PromotionAuditorError(
                    f"concurrent immutable publication conflicts: {destination}"
                )
            return True
        fsync_directory(destination.parent)
        return False
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def tree_manifest(path: Path) -> Tuple[str, int, Tuple[Mapping[str, Any], ...]]:
    """Reproduce the controller tree manifest and reject partial files."""

    root_path = Path(path)
    if root_path.is_symlink() or not root_path.is_dir():
        raise PromotionAuditorError(f"worker output is not a directory: {root_path}")
    rows: List[Mapping[str, Any]] = []
    total = 0
    for root, directories, files in os.walk(root_path, followlinks=False):
        current = Path(root)
        for name in sorted(directories):
            child = current / name
            if child.is_symlink():
                raise PromotionAuditorError(f"symlink in worker output: {child}")
        for name in sorted(files):
            child = current / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise PromotionAuditorError(f"nonregular worker output: {child}")
            if _PARTIAL_NAMES.search(name):
                raise PromotionAuditorError(
                    f"partial worker output is forbidden: {child}"
                )
            row = {
                "path": child.relative_to(root_path).as_posix(),
                "size": metadata.st_size,
                "sha256": sha256_file(child),
            }
            rows.append(row)
            total += metadata.st_size
    digest = canonical_sha256({"schemaVersion": 1, "files": rows})
    return digest, total, tuple(rows)


def _sgf_properties(game: str, source: str) -> Mapping[str, Tuple[str, ...]]:
    if not game.startswith("(;") or not game.endswith(")"):
        raise PromotionAuditorError(f"{source}: SGF is not one complete game tree")
    depth = 0
    bracket_depth = 0
    escaped = False
    for character in game:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if bracket_depth:
            if character == "]":
                bracket_depth -= 1
            elif character == "[":
                bracket_depth += 1
            continue
        if character == "[":
            bracket_depth = 1
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise PromotionAuditorError(f"{source}: SGF parentheses are unbalanced")
    if depth != 0 or bracket_depth != 0 or escaped:
        raise PromotionAuditorError(f"{source}: SGF is truncated or unbalanced")

    result: Dict[str, List[str]] = {}
    index = 0
    while index < len(game):
        if not game[index].isupper():
            index += 1
            continue
        end = index
        while end < len(game) and (game[end].isupper() or game[end].isdigit()):
            end += 1
        name = game[index:end]
        if end >= len(game) or game[end] != "[":
            index = end
            continue
        values: List[str] = []
        while end < len(game) and game[end] == "[":
            end += 1
            value: List[str] = []
            escaped = False
            while end < len(game):
                character = game[end]
                end += 1
                if escaped:
                    value.append(character)
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == "]":
                    break
                else:
                    value.append(character)
            else:
                raise PromotionAuditorError(f"{source}: SGF property is truncated")
            values.append("".join(value))
        result.setdefault(name, []).extend(values)
        index = end
    return {key: tuple(value) for key, value in result.items()}


def _count_closed_sgfs(root: Path) -> int:
    count = 0
    seen = set()
    sgfs_paths = sorted(Path(root).rglob("*.sgfs"))
    if not sgfs_paths:
        raise PromotionAuditorError("closed worker output contains no SGFS games")
    for path in sgfs_paths:
        if path.is_symlink() or not path.is_file():
            raise PromotionAuditorError(f"invalid SGFS worker output: {path}")
        try:
            data = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise PromotionAuditorError(f"SGFS is not UTF-8: {path}") from exc
        if data and not data.endswith("\n"):
            raise PromotionAuditorError(f"SGFS is not newline-terminated: {path}")
        for line_number, game in enumerate(data.splitlines(), start=1):
            if not game:
                raise PromotionAuditorError(
                    f"{path}:{line_number}: blank SGFS row is forbidden"
                )
            properties = _sgf_properties(game, f"{path}:{line_number}")
            if properties.get("GM") != ("1",):
                raise PromotionAuditorError(
                    f"{path}:{line_number}: SGF must identify Go as GM[1]"
                )
            if properties.get("FF") != ("4",):
                raise PromotionAuditorError(f"{path}:{line_number}: SGF must use FF[4]")
            if len(properties.get("SZ", ())) != 1:
                raise PromotionAuditorError(
                    f"{path}:{line_number}: SGF board size is missing"
                )
            results = properties.get("RE", ())
            if len(results) != 1 or _VALID_SGF_RESULT_RE.fullmatch(results[0]) is None:
                raise PromotionAuditorError(
                    f"{path}:{line_number}: SGF has no valid terminal result"
                )
            identity = hashlib.sha256(game.encode("utf-8")).hexdigest()
            if identity in seen:
                raise PromotionAuditorError(
                    f"{path}:{line_number}: duplicate SGF game is forbidden"
                )
            seen.add(identity)
            count += 1
    return count


def _self_hashed(payload: Mapping[str, Any]) -> Dict[str, Any]:
    value = dict(payload)
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def _validate_self_hash(value: Mapping[str, Any], role: str) -> None:
    payload = dict(value)
    supplied = payload.pop("manifest_sha256", None)
    if supplied != canonical_sha256(payload):
        raise PromotionAuditorError(f"{role} self-hash is invalid")


class PromotionAuditor:
    """Crash-recoverable rollout and asynchronous deep-audit producer."""

    def __init__(
        self,
        runtime: AuditorRuntime,
        *,
        katago_binary: Optional[Path] = None,
        evaluation_executor: Optional[EvaluationExecutor] = None,
        lease_factory: Optional[LeaseFactory] = None,
        subprocess_runner: Callable[..., Any] = subprocess.run,
        shards: int = 1,
        max_parallel: int = 1,
        max_attempts: int = 2,
        gpu_index: int = 7,
        failure_hook: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.runtime = runtime
        self.katago_binary = None if katago_binary is None else Path(katago_binary)
        self.evaluation_executor = evaluation_executor
        self.lease_factory = lease_factory
        self.subprocess_runner = subprocess_runner
        self.shards = _positive_int(shards, "shards")
        self.max_parallel = _positive_int(max_parallel, "max_parallel")
        self.max_attempts = _positive_int(max_attempts, "max_attempts")
        if self.max_parallel > self.shards:
            raise PromotionAuditorError("max_parallel may not exceed shards")
        if type(gpu_index) is not int or gpu_index < 0:
            raise PromotionAuditorError("gpu_index must be nonnegative")
        self.gpu_index = gpu_index
        self.failure_hook = failure_hook or (lambda _step: None)

    def _validate_static_inputs(self) -> None:
        runtime = self.runtime
        for root, role in (
            (runtime.promotion_root, "promotion root"),
            (runtime.rollout_quarantine, "rollout root"),
            (runtime.admitted_selfplay, "admitted self-play root"),
            (runtime.rollout_report_inbox, "rollout report inbox"),
            (runtime.accepted_models, "accepted model root"),
        ):
            if not Path(root).is_absolute():
                raise PromotionAuditorError(f"{role} must be absolute")
            if root.exists() and (root.is_symlink() or not root.is_dir()):
                raise PromotionAuditorError(f"{role} must be a non-symlink directory")
        if runtime.policy.get("policy_version") != POLICY_VERSION:
            raise PromotionAuditorError(
                "auditor requires the promotion-ready v3 policy"
            )
        try:
            policy_value = json.loads(
                runtime.policy_path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PromotionAuditorError(f"cannot load frozen policy: {exc}") from exc
        if (
            not isinstance(policy_value, dict)
            or canonical_sha256(policy_value) != runtime.policy_hash
            or canonical_json_bytes(policy_value)
            != canonical_json_bytes(runtime.policy)
        ):
            raise PromotionAuditorError("frozen policy contradicts auditor runtime")
        _regular_file(
            runtime.powered_config_path,
            runtime.powered_config_hash,
            "powered audit config",
        )
        _regular_file(
            runtime.selfplay_config_path,
            runtime.selfplay_config_hash,
            "self-play config",
        )
        _regular_file(
            runtime.gpu_lease_config_path,
            runtime.gpu_lease_config_hash,
            "GPU lease config",
        )
        _regular_file(
            runtime.audit_schedule_path,
            runtime.audit_schedule_hash,
            "canary audit schedule",
        )
        _regular_file(
            runtime.suite_manifest_path,
            runtime.suite_manifest_hash,
            "suite manifest",
        )
        load_suite_manifest(runtime.suite_manifest_path)
        _regular_file(
            runtime.original_model_path,
            runtime.original_hash,
            "immutable original model",
        )
        rollout = runtime.policy.get("rollout")
        if not isinstance(rollout, Mapping):
            raise PromotionAuditorError("policy rollout contract is missing")
        expected_topology = {
            "worker_count": runtime.worker_count,
            "canary_workers": runtime.canary_worker_count,
            "intermediate_workers": runtime.intermediate_worker_count,
            "games_per_worker_initial_threads": runtime.worker_threads,
        }
        if any(rollout.get(key) != value for key, value in expected_topology.items()):
            raise PromotionAuditorError(
                "runtime rollout topology contradicts frozen policy"
            )
        if runtime.minimum_candidate_win_rate < 0.0 or (
            runtime.minimum_candidate_win_rate > 1.0
        ):
            raise PromotionAuditorError("candidate win-rate threshold is out of range")
        if not 0.0 <= runtime.maximum_no_result_rate <= 1.0:
            raise PromotionAuditorError("no-result threshold is out of range")

    def _prepare_output_layout(self) -> None:
        root = self.runtime.promotion_root
        for path in (
            self.runtime.rollout_report_inbox,
            root / "audits",
            self.runtime.audit_outbox,
            self.runtime.audit_artifacts,
            root / "audits" / "evaluations",
        ):
            _ensure_directory(path, root)

    def _transaction(self, generation_id: str) -> Path:
        generation_id = _safe_generation_id(generation_id)
        return self.runtime.transactions / generation_id

    def _load_intent(self, generation_id: str) -> Dict[str, Any]:
        transaction = self._transaction(generation_id)
        try:
            intent = _load_canonical_object(
                transaction / "intent.json", "promotion intent"
            )
        except FileNotFoundError as exc:
            raise AuditNotReady("promotion intent is not durable yet") from exc
        expected = {
            "schema_version": SCHEMA_VERSION,
            "generation_id": generation_id,
            "policy_hash": self.runtime.policy_hash,
            "selfplay_config_hash": self.runtime.selfplay_config_hash,
            "topology": "7-workers-100-threads",
        }
        if any(intent.get(key) != value for key, value in expected.items()):
            raise PromotionAuditorError(
                "promotion intent contradicts auditor runtime identity"
            )
        candidate_hash = _sha256(
            intent.get("candidate_hash"), "promotion intent candidate hash"
        )
        reference_hash = _sha256(
            intent.get("tested_champion_hash"),
            "promotion intent tested champion hash",
        )
        _sha256(intent.get("config_hash"), "promotion intent config hash")
        pass_path_value = _nonempty(
            intent.get("pass_report_path"), "promotion intent pass report path"
        )
        pass_hash = _sha256(
            intent.get("pass_report_hash"), "promotion intent pass report hash"
        )
        pass_path = _require_absolute_under(
            Path(pass_path_value), self.runtime.promotion_root, "PASS report"
        )
        _regular_file(pass_path, pass_hash, "promotion PASS report")
        leaf = (
            self.runtime.accepted_models
            / "generations"
            / candidate_hash
            / generation_id
            / "model.bin.gz"
        )
        _regular_file(leaf, candidate_hash, "generation candidate model")
        if reference_hash == candidate_hash:
            raise PromotionAuditorError("promotion intent candidate equals champion")
        return intent

    def _worker_output(self, generation_id: str, worker_id: int) -> Path:
        quarantined = (
            self.runtime.rollout_quarantine
            / generation_id
            / "data"
            / f"worker-{worker_id:03d}"
        )
        admitted = (
            self.runtime.admitted_selfplay / generation_id / f"worker-{worker_id:03d}"
        )
        candidates = [path for path in (quarantined, admitted) if path.exists()]
        if not candidates:
            raise AuditNotReady(f"worker {worker_id} has no closed output directory")
        if len(candidates) != 1:
            raise PromotionAuditorError(
                f"worker {worker_id} output exists in quarantine and admission"
            )
        return candidates[0]

    def _worker_evidence(
        self,
        intent: Mapping[str, Any],
        phase: str,
        worker_id: int,
    ) -> Mapping[str, Any]:
        generation_id = intent["generation_id"]
        candidate_hash = intent["candidate_hash"]
        rollout = self.runtime.rollout_quarantine / generation_id
        ack_path = rollout / "acknowledgements" / f"worker-{worker_id:03d}.json"
        try:
            acknowledgement = _load_canonical_object(
                ack_path, f"worker {worker_id} acknowledgement"
            )
        except FileNotFoundError as exc:
            raise AuditNotReady(
                f"worker {worker_id} acknowledgement is not available"
            ) from exc
        acknowledgement_hash = sha256_file(ack_path)
        expected_ack_fields = {
            "schema_version",
            "finalized",
            "generation_id",
            "worker_id",
            "model_hash",
            "selfplay_config_hash",
            "policy_hash",
            "threads",
            "output_manifest_hash",
            "closed_files",
            "process_identity",
            "report_hash",
        }
        if set(acknowledgement) != expected_ack_fields:
            raise PromotionAuditorError(
                f"worker {worker_id} acknowledgement fields are not exact"
            )
        expected = {
            "schema_version": SCHEMA_VERSION,
            "finalized": True,
            "generation_id": generation_id,
            "worker_id": worker_id,
            "model_hash": candidate_hash,
            "selfplay_config_hash": self.runtime.selfplay_config_hash,
            "policy_hash": self.runtime.policy_hash,
            "threads": self.runtime.worker_threads,
            "closed_files": True,
        }
        if any(acknowledgement.get(key) != value for key, value in expected.items()):
            raise PromotionAuditorError(
                f"worker {worker_id} acknowledgement identity is invalid"
            )
        source_hash = _sha256(
            acknowledgement["report_hash"],
            f"worker {worker_id} source report hash",
        )
        source_payload = dict(acknowledgement)
        source_payload.pop("report_hash")
        if (
            hashlib.sha256(canonical_json_bytes(source_payload) + b"\n").hexdigest()
            != source_hash
        ):
            raise PromotionAuditorError(
                f"worker {worker_id} acknowledgement source hash is invalid"
            )

        worker_intent = _load_canonical_object(
            rollout / f"worker-{worker_id:03d}" / "intent.json",
            f"worker {worker_id} intent",
        )
        expected_worker_intent = {
            "schema_version": SCHEMA_VERSION,
            "worker_id": worker_id,
            "generation_id": generation_id,
            "model_hash": candidate_hash,
            "selfplay_config_hash": self.runtime.selfplay_config_hash,
            "policy": str(self.runtime.policy_path),
            "policy_hash": self.runtime.policy_hash,
            "threads": self.runtime.worker_threads,
        }
        if worker_intent != expected_worker_intent:
            raise PromotionAuditorError(
                f"worker {worker_id} intent contradicts promotion"
            )

        launch_paths = [
            path
            for path in sorted(
                (rollout / f"worker-{worker_id:03d}").glob("launch-*.json")
            )
            if not path.name.endswith(".intent.json")
        ]
        if len(launch_paths) != 1:
            raise AuditNotReady(
                f"worker {worker_id} has no unique finalized launch marker"
            )
        launch = _load_canonical_object(
            launch_paths[0], f"worker {worker_id} launch marker"
        )
        expected_phase = (
            "canary"
            if worker_id < self.runtime.canary_worker_count
            else "intermediate"
            if worker_id < self.runtime.intermediate_worker_count
            else "full"
        )
        expected_launch = {
            "schema_version": SCHEMA_VERSION,
            "generation_id": generation_id,
            "model_hash": candidate_hash,
            "worker_id": worker_id,
            "phase": expected_phase,
            "selfplay_config_hash": self.runtime.selfplay_config_hash,
            "policy_hash": self.runtime.policy_hash,
            "supervisor_key": f"{generation_id}:worker-{worker_id:03d}",
            "process_identity": acknowledgement["process_identity"],
            "process_identity_verified": True,
        }
        if launch != expected_launch:
            raise PromotionAuditorError(
                f"worker {worker_id} launch provenance is invalid"
            )
        if phase == "canary" and expected_phase != "canary":
            raise PromotionAuditorError("canary report includes a non-canary worker")
        if phase == "intermediate" and worker_id >= (
            self.runtime.intermediate_worker_count
        ):
            raise PromotionAuditorError(
                "intermediate report includes a full-rollout worker"
            )
        identity = acknowledgement["process_identity"]
        if (
            not isinstance(identity, Mapping)
            or type(identity.get("pid")) is not int
            or identity["pid"] <= 0
            or type(identity.get("start_time_ticks")) is not int
            or identity["start_time_ticks"] < 0
            or not isinstance(identity.get("command_sha256"), str)
            or _SHA256_RE.fullmatch(identity["command_sha256"]) is None
        ):
            raise PromotionAuditorError(
                f"worker {worker_id} process identity is not verifiable"
            )

        output = self._worker_output(generation_id, worker_id)
        manifest_hash, output_size, files = tree_manifest(output)
        if output_size <= 0 or any(file["size"] <= 0 for file in files):
            raise PromotionAuditorError(f"worker {worker_id} closed output is empty")
        if acknowledgement["output_manifest_hash"] != manifest_hash:
            raise PromotionAuditorError(
                f"worker {worker_id} output changed after acknowledgement"
            )
        game_count = _count_closed_sgfs(output)
        final_manifest_hash, final_output_size, _ = tree_manifest(output)
        if (
            final_manifest_hash != manifest_hash
            or final_output_size != output_size
            or sha256_file(ack_path) != acknowledgement_hash
        ):
            raise PromotionAuditorError(
                f"worker {worker_id} evidence changed during audit"
            )
        return {
            "worker_id": worker_id,
            "phase": expected_phase,
            "acknowledgement_path": str(ack_path.resolve()),
            "acknowledgement_sha256": acknowledgement_hash,
            "source_report_sha256": source_hash,
            "output_path": str(output.resolve()),
            "output_manifest_hash": manifest_hash,
            "output_size_bytes": output_size,
            "file_count": len(files),
            "game_count": game_count,
            "process_identity": dict(identity),
        }

    def derive_rollout_health(self, generation_id: str, phase: str) -> RolloutHealth:
        """Derive phase health solely from controller state and closed workers."""

        if phase not in {"canary", "intermediate"}:
            raise PromotionAuditorError("phase must be canary or intermediate")
        self._validate_static_inputs()
        intent = self._load_intent(generation_id)
        required = (
            self.runtime.canary_worker_count
            if phase == "canary"
            else self.runtime.intermediate_worker_count
        )
        workers = tuple(
            self._worker_evidence(intent, phase, worker_id)
            for worker_id in range(required)
        )
        game_count = sum(worker["game_count"] for worker in workers)
        minimum = self.runtime.canary_games * required
        checks = {
            "model_purity_pass": all(
                worker["output_manifest_hash"] for worker in workers
            ),
            "output_schema_pass": all(worker["game_count"] > 0 for worker in workers),
            "throughput_pass": game_count >= minimum,
            "crash_error_pass": all(
                worker["source_report_sha256"] for worker in workers
            ),
            "behavior_pass": all(worker["game_count"] > 0 for worker in workers),
            "catastrophe_pass": all(worker["game_count"] > 0 for worker in workers),
        }
        return RolloutHealth(
            phase=phase,
            worker_count=required,
            game_count=game_count,
            minimum_game_count=minimum,
            checks=checks,
            workers=workers,
        )

    def _previous_champion_model(self, generation_id: str, expected_hash: str) -> Path:
        previous_path = self._transaction(generation_id) / "previous-champion.json"
        try:
            previous = load_champion(previous_path)
        except (OSError, ValueError) as exc:
            raise PromotionAuditorError(
                "previous champion snapshot is invalid"
            ) from exc
        if previous.champion_hash != expected_hash:
            raise PromotionAuditorError(
                "previous champion snapshot contradicts promotion intent"
            )
        if expected_hash == self.runtime.original_hash:
            return _regular_file(
                self.runtime.original_model_path,
                expected_hash,
                "original champion model",
            )
        path = (
            self.runtime.accepted_models
            / "generations"
            / expected_hash
            / previous.generation_id
            / "model.bin.gz"
        )
        return _regular_file(path, expected_hash, "previous champion model")

    def _candidate_model(self, generation_id: str, candidate_hash: str) -> Path:
        path = (
            self.runtime.accepted_models
            / "generations"
            / candidate_hash
            / generation_id
            / "model.bin.gz"
        )
        return _regular_file(path, candidate_hash, "generation candidate model")

    def _bank_for_schedule(
        self, schedule_hash: str, *, expected_path: Optional[Path] = None
    ) -> Mapping[str, Any]:
        manifest = load_suite_manifest(self.runtime.suite_manifest_path)
        matches = [
            bank
            for bank in manifest.get("banks", [])
            if isinstance(bank, Mapping)
            and isinstance(bank.get("schedule"), Mapping)
            and bank["schedule"].get("sha256") == schedule_hash
        ]
        if len(matches) != 1:
            raise PromotionAuditorError(
                "schedule hash must identify exactly one frozen suite bank"
            )
        bank = matches[0]
        positions = bank.get("positions")
        schedule = bank.get("schedule")
        if not isinstance(positions, Mapping) or not isinstance(schedule, Mapping):
            raise PromotionAuditorError("suite bank manifest is incomplete")
        relative = _nonempty(schedule.get("path"), "suite bank schedule path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise PromotionAuditorError("suite bank schedule path is unsafe")
        actual_path = (
            self.runtime.suite_manifest_path.parent / relative_path
        ).resolve()
        if expected_path is not None and actual_path != Path(expected_path):
            raise PromotionAuditorError(
                "queued schedule path contradicts frozen suite manifest"
            )
        _regular_file(actual_path, schedule_hash, "frozen suite schedule")
        return {
            "name": _nonempty(bank.get("name"), "suite bank name"),
            "qualified_name": bank.get("qualifiedName", bank.get("name")),
            "bank_hash": _sha256(positions.get("sha256"), "suite bank positions hash"),
            "schedule_path": actual_path,
            "schedule_hash": schedule_hash,
            "schedule_id": _nonempty(
                schedule.get("scheduleId"), "suite bank schedule ID"
            ),
            "row_count": _positive_int(
                schedule.get("rowCount"), "suite bank schedule row count"
            ),
            "pair_count": _positive_int(
                schedule.get("pairCount"), "suite bank schedule pair count"
            ),
        }

    def _build_job(
        self,
        *,
        kind: str,
        job_id: str,
        generation_id: str,
        candidate_hash: str,
        reference_hash: str,
        reference_model_path: Path,
        schedule_path: Path,
        schedule_hash: str,
        schedule_id: str,
        suite_name: str,
        suite_bank_hash: str,
        color_pairs: int,
        max_visits: int,
        request_hash: Optional[str] = None,
        cell: Optional[Mapping[str, Any]] = None,
    ) -> EvaluationJob:
        output_root = (
            self.runtime.promotion_root
            / "audits"
            / "evaluations"
            / kind
            / generation_id
            / job_id
        )
        spec = EvaluationSpec(
            candidate_model_sha=candidate_hash,
            reference_model_sha=reference_hash,
            original_model_sha=self.runtime.original_hash,
            config_sha=self.runtime.powered_config_hash,
            schedule_sha=schedule_hash,
            policy_sha=self.runtime.policy_hash,
            comparison=(
                "candidate-vs-champion-powered-canary-audit"
                if kind == "canary"
                else f"candidate-vs-{cell['control']}-powered-deep-audit"
            ),
            suite=suite_name,
            stage="canary-audit" if kind == "canary" else "deep-audit",
            look=("final" if kind == "canary" else f"visits-{max_visits}"),
            topology="single-gpu-manifest-bound-audit",
            max_visits=max_visits,
            suite_manifest_sha=self.runtime.suite_manifest_hash,
            suite_bank_sha=suite_bank_hash,
            schedule_id=schedule_id,
        )
        return EvaluationJob(
            kind=kind,
            job_id=job_id,
            generation_id=generation_id,
            candidate_hash=candidate_hash,
            reference_hash=reference_hash,
            candidate_model_path=self._candidate_model(generation_id, candidate_hash),
            reference_model_path=_regular_file(
                reference_model_path, reference_hash, "audit reference model"
            ),
            original_model_path=self.runtime.original_model_path,
            config_path=self.runtime.powered_config_path,
            schedule_path=schedule_path,
            schedule_hash=schedule_hash,
            schedule_id=schedule_id,
            suite_name=suite_name,
            suite_bank_hash=suite_bank_hash,
            color_pairs=color_pairs,
            max_visits=max_visits,
            output_root=output_root,
            spec=spec,
            request_hash=request_hash,
            cell=None if cell is None else json.loads(canonical_json_bytes(cell)),
        )

    def _canary_job(self, intent: Mapping[str, Any]) -> EvaluationJob:
        bank = self._bank_for_schedule(
            self.runtime.audit_schedule_hash,
            expected_path=self.runtime.audit_schedule_path,
        )
        if bank["pair_count"] != self.runtime.canary_pairs:
            raise PromotionAuditorError(
                "canary audit pair quota contradicts frozen schedule"
            )
        visits = (
            self.runtime.policy.get("evaluation_stages", {})
            .get("deep_audit", {})
            .get("visits")
        )
        if visits != [2000, 8000]:
            raise PromotionAuditorError("deep-audit visit tiers are not frozen")
        reference_hash = intent["tested_champion_hash"]
        return self._build_job(
            kind="canary",
            job_id="fresh-audit",
            generation_id=intent["generation_id"],
            candidate_hash=intent["candidate_hash"],
            reference_hash=reference_hash,
            reference_model_path=self._previous_champion_model(
                intent["generation_id"], reference_hash
            ),
            schedule_path=self.runtime.audit_schedule_path,
            schedule_hash=self.runtime.audit_schedule_hash,
            schedule_id=bank["schedule_id"],
            suite_name=bank["name"],
            suite_bank_hash=bank["bank_hash"],
            color_pairs=self.runtime.canary_pairs,
            max_visits=visits[0],
        )

    def _default_executor(self, job: EvaluationJob) -> EvaluationArtifacts:
        if self.katago_binary is None:
            raise PromotionAuditorError(
                "manifest-bound evaluation requires a KataGo binary"
            )
        binary = Path(self.katago_binary)
        if binary.is_symlink() or not binary.is_file():
            raise PromotionAuditorError(
                "KataGo binary must be a regular non-symlink file"
            )
        _ensure_directory(job.output_root, self.runtime.promotion_root)
        runner = EvaluationRunner(
            katago_binary=binary,
            config_path=job.config_path,
            output_root=job.output_root,
            shard_count=self.shards,
            max_parallel=self.max_parallel,
            max_attempts=self.max_attempts,
            include_move_traces=True,
            env={"CUDA_VISIBLE_DEVICES": str(self.gpu_index)},
            subprocess_runner=self.subprocess_runner,
        )
        result = runner.run(
            job.spec,
            job.schedule_path,
            job.candidate_model_path,
            job.reference_model_path,
            original_model_path=job.original_model_path,
            policy_path=self.runtime.policy_path,
            suite_manifest_path=self.runtime.suite_manifest_path,
        )
        if not isinstance(result, EvaluationResult) or result.move_path is None:
            raise PromotionAuditorError("EvaluationRunner did not finalize move traces")
        return EvaluationArtifacts(
            results_path=result.result_path,
            moves_path=result.move_path,
            source_manifest_path=result.manifest_path,
        )

    def _execute(self, job: EvaluationJob) -> EvaluationArtifacts:
        executor = self.evaluation_executor or self._default_executor
        artifacts = executor(job)
        if not isinstance(artifacts, EvaluationArtifacts):
            raise PromotionAuditorError(
                "evaluation executor must return paths, never a decision/PASS value"
            )
        return artifacts

    def _observe(
        self, job: EvaluationJob, artifacts: EvaluationArtifacts
    ) -> EvaluationObservation:
        _regular_file(job.candidate_model_path, job.candidate_hash, "candidate model")
        _regular_file(job.reference_model_path, job.reference_hash, "reference model")
        _regular_file(
            job.original_model_path, self.runtime.original_hash, "original model"
        )
        _regular_file(job.config_path, self.runtime.powered_config_hash, "audit config")
        _regular_file(job.schedule_path, job.schedule_hash, "audit schedule")
        schedule = load_schedule(job.schedule_path)
        pair_ids = tuple(sorted({row["pairId"] for row in schedule}))
        if (
            len(schedule) != 2 * job.color_pairs
            or len(pair_ids) != job.color_pairs
            or {row["scheduleId"] for row in schedule} != {job.schedule_id}
            or {row.get("suiteBankSha256") for row in schedule} != {job.suite_bank_hash}
        ):
            raise PromotionAuditorError(
                "audit schedule does not match the exact requested pair matrix"
            )
        results_path = _require_absolute_under(
            Path(artifacts.results_path),
            self.runtime.promotion_root,
            "evaluation results",
        )
        moves_path = _require_absolute_under(
            Path(artifacts.moves_path),
            self.runtime.promotion_root,
            "evaluation moves",
        )
        results_hash = sha256_file(results_path)
        moves_hash = sha256_file(moves_path)
        _load_canonical_jsonl(results_path, "evaluation results")
        _load_canonical_jsonl(moves_path, "evaluation moves")
        result_rows = validate_result_jsonl(results_path, schedule)
        validate_move_jsonl(moves_path, schedule, result_rows)
        source_path: Optional[Path] = None
        source_hash: Optional[str] = None
        if artifacts.source_manifest_path is not None:
            source_path = _require_absolute_under(
                Path(artifacts.source_manifest_path),
                self.runtime.promotion_root,
                "source runner manifest",
            )
            source = _load_canonical_object(source_path, "source runner manifest")
            payload = dict(source)
            identity = payload.pop("manifestPayloadSha256", None)
            if identity is not None and identity != canonical_sha256(payload):
                raise PromotionAuditorError(
                    "source runner manifest self-hash is invalid"
                )
            source_hash = sha256_file(source_path)

        scores = []
        true_no_results = 0
        for row in result_rows:
            if row.get("noResult") is True:
                true_no_results += 1
                scores.append(0.5)
                continue
            winner = row.get("winner")
            if winner in {None, "draw", "D", "Draw"}:
                scores.append(0.5)
                continue
            candidate_color = "B" if row.get("blackBot") == "candidate" else "W"
            scores.append(1.0 if winner == candidate_color else 0.0)
        if not scores:
            raise PromotionAuditorError("audit results contain no games")
        candidate_win_rate = sum(scores) / len(scores)
        no_result_rate = true_no_results / len(scores)
        failures = int(candidate_win_rate <= self.runtime.minimum_candidate_win_rate)
        if job.kind == "canary":
            failures += int(no_result_rate >= self.runtime.maximum_no_result_rate)
        decision = "PASS" if failures == 0 else "FAIL"
        _regular_file(job.candidate_model_path, job.candidate_hash, "candidate model")
        _regular_file(job.reference_model_path, job.reference_hash, "reference model")
        _regular_file(job.schedule_path, job.schedule_hash, "audit schedule")
        if (
            sha256_file(results_path) != results_hash
            or sha256_file(moves_path) != moves_hash
            or (source_path is not None and sha256_file(source_path) != source_hash)
        ):
            raise PromotionAuditorError(
                "evaluation evidence changed while its decision was derived"
            )
        return EvaluationObservation(
            results_path=results_path,
            moves_path=moves_path,
            source_manifest_path=source_path,
            results_sha256=results_hash,
            moves_sha256=moves_hash,
            source_manifest_sha256=source_hash,
            pair_ids=pair_ids,
            game_count=len(scores),
            true_no_results=true_no_results,
            true_no_result_rate=no_result_rate,
            candidate_win_rate=candidate_win_rate,
            minimum_candidate_win_rate=self.runtime.minimum_candidate_win_rate,
            safety_failures=failures,
            decision=decision,
        )

    def _lease(self) -> ContextManager[Mapping[str, Any]]:
        if self.lease_factory is None:
            raise PromotionAuditorError(
                "evaluation is forbidden without an exclusive GPU lease"
            )
        return self.lease_factory()

    def _run_pending_jobs(
        self, jobs: Sequence[EvaluationJob]
    ) -> Tuple[Mapping[str, EvaluationObservation], Mapping[str, Any]]:
        observations: Dict[str, EvaluationObservation] = {}
        proof: Mapping[str, Any]
        with self._lease() as lease_proof:
            if not isinstance(lease_proof, Mapping):
                raise PromotionAuditorError("GPU lease returned no provenance")
            proof = lease_proof
            for job in jobs:
                observations[job.job_id] = self._observe(job, self._execute(job))
        # Lease factories may update a mutable proof during trainer restoration.
        final_proof = json.loads(canonical_json_bytes(proof))
        return observations, final_proof

    def _canary_runner_value(
        self,
        job: EvaluationJob,
        observation: EvaluationObservation,
        lease_proof: Mapping[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract": CANARY_RUNNER_CONTRACT,
            "finalized": True,
            "generation_id": job.generation_id,
            "candidate_hash": job.candidate_hash,
            "reference_hash": job.reference_hash,
            "policy_hash": self.runtime.policy_hash,
            "suite_manifest_hash": self.runtime.suite_manifest_hash,
            "audit_schedule_hash": job.schedule_hash,
            "color_pairs": job.color_pairs,
            "pair_ids": list(observation.pair_ids),
            "pair_ids_sha256": canonical_sha256(list(observation.pair_ids)),
            "decision": observation.decision,
            "candidate_win_rate": observation.candidate_win_rate,
            "minimum_candidate_win_rate": observation.minimum_candidate_win_rate,
            "true_no_result_rate": observation.true_no_result_rate,
            "results_path": str(observation.results_path.resolve()),
            "results_sha256": observation.results_sha256,
            "moves_path": str(observation.moves_path.resolve()),
            "moves_sha256": observation.moves_sha256,
            "source_runner_manifest_path": (
                None
                if observation.source_manifest_path is None
                else str(observation.source_manifest_path.resolve())
            ),
            "source_runner_manifest_sha256": observation.source_manifest_sha256,
            "gpu_lease_proof": json.loads(canonical_json_bytes(lease_proof)),
            "gpu_lease_proof_sha256": canonical_sha256(lease_proof),
        }
        return _self_hashed(payload)

    def _validate_canary_runner(
        self, job: EvaluationJob, path: Path
    ) -> Tuple[Dict[str, Any], EvaluationObservation]:
        value = _load_canonical_object(path, "canary runner manifest")
        _validate_self_hash(value, "canary runner manifest")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "contract": CANARY_RUNNER_CONTRACT,
            "finalized": True,
            "generation_id": job.generation_id,
            "candidate_hash": job.candidate_hash,
            "reference_hash": job.reference_hash,
            "policy_hash": self.runtime.policy_hash,
            "suite_manifest_hash": self.runtime.suite_manifest_hash,
            "audit_schedule_hash": job.schedule_hash,
            "color_pairs": job.color_pairs,
        }
        if any(value.get(key) != item for key, item in expected.items()):
            raise PromotionAuditorError(
                "canary runner manifest contradicts promotion request"
            )
        source_value = value.get("source_runner_manifest_path")
        artifacts = EvaluationArtifacts(
            results_path=Path(_nonempty(value.get("results_path"), "results path")),
            moves_path=Path(_nonempty(value.get("moves_path"), "moves path")),
            source_manifest_path=(
                None
                if source_value is None
                else Path(_nonempty(source_value, "source runner manifest path"))
            ),
        )
        observation = self._observe(job, artifacts)
        comparisons = {
            "results_sha256": observation.results_sha256,
            "moves_sha256": observation.moves_sha256,
            "source_runner_manifest_sha256": observation.source_manifest_sha256,
            "pair_ids": list(observation.pair_ids),
            "pair_ids_sha256": canonical_sha256(list(observation.pair_ids)),
            "decision": observation.decision,
            "candidate_win_rate": observation.candidate_win_rate,
            "minimum_candidate_win_rate": observation.minimum_candidate_win_rate,
            "true_no_result_rate": observation.true_no_result_rate,
        }
        if any(value.get(key) != item for key, item in comparisons.items()):
            raise PromotionAuditorError(
                "canary runner decision is not derivable from its outputs"
            )
        proof = value.get("gpu_lease_proof")
        if not isinstance(proof, Mapping) or value.get(
            "gpu_lease_proof_sha256"
        ) != canonical_sha256(proof):
            raise PromotionAuditorError("canary runner GPU lease proof is invalid")
        return value, observation

    def _canary_artifacts(
        self, intent: Mapping[str, Any]
    ) -> Tuple[Mapping[str, Any], Path, Path, Path, bool]:
        transaction = self._transaction(intent["generation_id"])
        job = self._canary_job(intent)
        runner_path = transaction / "fresh-audit-runner.json"
        reused = runner_path.exists()
        if reused:
            runner_value, observation = self._validate_canary_runner(job, runner_path)
        else:
            observations, proof = self._run_pending_jobs((job,))
            self._validate_static_inputs()
            if self._load_intent(intent["generation_id"]) != intent:
                raise PromotionAuditorError(
                    "promotion intent changed during canary evaluation"
                )
            observation = observations[job.job_id]
            runner_value = self._canary_runner_value(job, observation, proof)
            publish_canonical_json(runner_path, runner_value)
            self.failure_hook("canary-runner-published")
            runner_value, observation = self._validate_canary_runner(job, runner_path)

        pair_ids = list(observation.pair_ids)
        pair_ids_hash = canonical_sha256(pair_ids)
        statistics_path = transaction / "fresh-audit-statistics.json"
        statistics = _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": CANARY_STATISTICS_CONTRACT,
                "finalized": True,
                "decision": observation.decision,
                "candidate_hash": intent["candidate_hash"],
                "reference_hash": intent["tested_champion_hash"],
                "policy_hash": self.runtime.policy_hash,
                "suite_manifest_hash": self.runtime.suite_manifest_hash,
                "audit_schedule_hash": self.runtime.audit_schedule_hash,
                "color_pairs": self.runtime.canary_pairs,
                "pair_ids": pair_ids,
                "pair_ids_sha256": pair_ids_hash,
                "safety_failures": observation.safety_failures,
                "candidate_win_rate": observation.candidate_win_rate,
                "minimum_candidate_win_rate": observation.minimum_candidate_win_rate,
                "true_no_results": observation.true_no_results,
                "true_no_result_rate": observation.true_no_result_rate,
                "runner_manifest_path": str(runner_path.resolve()),
                "runner_manifest_sha256": sha256_file(runner_path),
                "results_sha256": observation.results_sha256,
                "moves_sha256": observation.moves_sha256,
            }
        )
        reused = publish_canonical_json(statistics_path, statistics) and reused
        self.failure_hook("canary-statistics-published")

        audit_path = transaction / "fresh-audit.json"
        audit = _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": CANARY_AUDIT_CONTRACT,
                "finalized": True,
                "decision": observation.decision,
                "generation_id": intent["generation_id"],
                "candidate_hash": intent["candidate_hash"],
                "reference_hash": intent["tested_champion_hash"],
                "policy_hash": self.runtime.policy_hash,
                "suite_manifest_hash": self.runtime.suite_manifest_hash,
                "audit_schedule_hash": self.runtime.audit_schedule_hash,
                "color_pairs": self.runtime.canary_pairs,
                "pair_ids_sha256": pair_ids_hash,
                "statistics_artifact_path": str(statistics_path.resolve()),
                "statistics_artifact_sha256": sha256_file(statistics_path),
                "runner_manifest_path": str(runner_path.resolve()),
                "runner_manifest_sha256": sha256_file(runner_path),
                "safety_failures": observation.safety_failures,
            }
        )
        reused = publish_canonical_json(audit_path, audit) and reused
        self.failure_hook("canary-audit-published")
        return audit, runner_path, statistics_path, audit_path, reused

    def _validate_canary_artifacts(
        self, intent: Mapping[str, Any], report: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        job = self._canary_job(intent)
        audit_path_value = _nonempty(
            report.get("fresh_audit_manifest_path"),
            "canary fresh-audit manifest path",
        )
        audit_path = _require_absolute_under(
            Path(audit_path_value),
            self.runtime.promotion_root,
            "canary fresh-audit manifest",
        )
        expected_audit_hash = _sha256(
            report.get("fresh_audit_manifest_sha256"),
            "canary fresh-audit manifest hash",
        )
        _regular_file(audit_path, expected_audit_hash, "canary fresh-audit manifest")
        audit = _load_canonical_object(audit_path, "canary fresh-audit manifest")
        _validate_self_hash(audit, "canary fresh-audit manifest")
        runner_path = Path(
            _nonempty(audit.get("runner_manifest_path"), "canary runner path")
        )
        runner_hash = _sha256(audit.get("runner_manifest_sha256"), "canary runner hash")
        _regular_file(runner_path, runner_hash, "canary runner manifest")
        _, observation = self._validate_canary_runner(job, runner_path)
        statistics_path = Path(
            _nonempty(
                audit.get("statistics_artifact_path"),
                "canary statistics artifact path",
            )
        )
        statistics_hash = _sha256(
            audit.get("statistics_artifact_sha256"),
            "canary statistics artifact hash",
        )
        _regular_file(statistics_path, statistics_hash, "canary statistics")
        statistics = _load_canonical_object(
            statistics_path, "canary statistics artifact"
        )
        _validate_self_hash(statistics, "canary statistics artifact")
        pair_ids = list(observation.pair_ids)
        expected_statistics = {
            "schema_version": SCHEMA_VERSION,
            "contract": CANARY_STATISTICS_CONTRACT,
            "finalized": True,
            "decision": observation.decision,
            "candidate_hash": intent["candidate_hash"],
            "reference_hash": intent["tested_champion_hash"],
            "policy_hash": self.runtime.policy_hash,
            "suite_manifest_hash": self.runtime.suite_manifest_hash,
            "audit_schedule_hash": self.runtime.audit_schedule_hash,
            "color_pairs": self.runtime.canary_pairs,
            "pair_ids": pair_ids,
            "pair_ids_sha256": canonical_sha256(pair_ids),
            "safety_failures": observation.safety_failures,
            "runner_manifest_path": str(runner_path.resolve()),
            "runner_manifest_sha256": runner_hash,
            "results_sha256": observation.results_sha256,
            "moves_sha256": observation.moves_sha256,
            "candidate_win_rate": observation.candidate_win_rate,
            "minimum_candidate_win_rate": observation.minimum_candidate_win_rate,
            "true_no_results": observation.true_no_results,
            "true_no_result_rate": observation.true_no_result_rate,
        }
        if any(
            statistics.get(key) != value for key, value in expected_statistics.items()
        ):
            raise PromotionAuditorError("canary statistics are not output-derived")
        expected_audit = {
            "schema_version": SCHEMA_VERSION,
            "contract": CANARY_AUDIT_CONTRACT,
            "finalized": True,
            "decision": observation.decision,
            "generation_id": intent["generation_id"],
            "candidate_hash": intent["candidate_hash"],
            "reference_hash": intent["tested_champion_hash"],
            "policy_hash": self.runtime.policy_hash,
            "suite_manifest_hash": self.runtime.suite_manifest_hash,
            "audit_schedule_hash": self.runtime.audit_schedule_hash,
            "color_pairs": self.runtime.canary_pairs,
            "pair_ids_sha256": canonical_sha256(pair_ids),
            "statistics_artifact_path": str(statistics_path.resolve()),
            "statistics_artifact_sha256": statistics_hash,
            "runner_manifest_path": str(runner_path.resolve()),
            "runner_manifest_sha256": runner_hash,
            "safety_failures": observation.safety_failures,
        }
        if any(audit.get(key) != value for key, value in expected_audit.items()):
            raise PromotionAuditorError("canary audit manifest is not output-derived")
        return {
            "audit": audit,
            "statistics": statistics,
            "observation": observation,
        }

    def _load_canary_pass(self, intent: Mapping[str, Any]) -> Mapping[str, Any]:
        path = self._transaction(intent["generation_id"]) / "canary-pass.json"
        try:
            value = _load_canonical_object(path, "controller canary PASS")
        except FileNotFoundError as exc:
            raise AuditNotReady(
                "intermediate health waits for controller canary admission"
            ) from exc
        if (
            value.get("decision") != "PASS"
            or value.get("phase") != "canary"
            or value.get("generation_id") != intent["generation_id"]
            or value.get("candidate_hash") != intent["candidate_hash"]
        ):
            raise PromotionAuditorError("controller canary PASS is contradictory")
        report_hash = _sha256(
            value.get("report_hash"), "controller canary source report hash"
        )
        source = dict(value)
        source.pop("report_hash")
        actual_report_hash = hashlib.sha256(
            canonical_json_bytes(source) + b"\n"
        ).hexdigest()
        if actual_report_hash != report_hash:
            raise PromotionAuditorError("controller canary PASS source hash is invalid")
        _validate_self_hash(source, "canary rollout report")
        derived = self._validate_canary_artifacts(intent, source)
        if derived["observation"].decision != "PASS":
            raise PromotionAuditorError(
                "controller canary PASS contradicts fresh-audit outputs"
            )
        return value

    def _rollout_report(
        self,
        intent: Mapping[str, Any],
        health: RolloutHealth,
        *,
        canary_audit: Optional[Mapping[str, Any]],
        canary_pass: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        checks = dict(health.checks)
        fresh_pair_ids: List[str] = []
        fresh_manifest_path: Optional[str] = None
        fresh_manifest_hash: Optional[str] = None
        fresh_pairs = 0
        source_canary_hash: Optional[str] = None
        if health.phase == "canary":
            assert canary_audit is not None
            validated = self._validate_canary_artifacts(
                intent,
                {
                    "fresh_audit_manifest_path": canary_audit[
                        "fresh_audit_manifest_path"
                    ],
                    "fresh_audit_manifest_sha256": canary_audit[
                        "fresh_audit_manifest_sha256"
                    ],
                },
            )
            observation = validated["observation"]
            fresh_pair_ids = list(observation.pair_ids)
            fresh_pairs = len(fresh_pair_ids)
            fresh_manifest_path = canary_audit["fresh_audit_manifest_path"]
            fresh_manifest_hash = canary_audit["fresh_audit_manifest_sha256"]
            checks["behavior_pass"] = observation.decision == "PASS"
            checks["catastrophe_pass"] = observation.safety_failures == 0
        else:
            assert canary_pass is not None
            source_canary_hash = canary_pass["report_hash"]
            checks["behavior_pass"] = True
            checks["catastrophe_pass"] = True

        decision = "PASS" if all(checks.values()) else "FAIL"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract": ROLLOUT_REPORT_CONTRACT,
            "finalized": True,
            "decision": decision,
            "phase": health.phase,
            "generation_id": intent["generation_id"],
            "candidate_hash": intent["candidate_hash"],
            "policy_hash": self.runtime.policy_hash,
            "promotion_config_hash": intent["config_hash"],
            "selfplay_config_hash": self.runtime.selfplay_config_hash,
            "audit_schedule_hash": self.runtime.audit_schedule_hash,
            "topology": "7-workers-100-threads",
            "worker_count": health.worker_count,
            **checks,
            "game_count": health.game_count,
            "minimum_game_count": health.minimum_game_count,
            "fresh_audit_pairs": fresh_pairs,
            "fresh_audit_pair_ids": fresh_pair_ids,
            "fresh_audit_pair_ids_sha256": canonical_sha256(fresh_pair_ids),
            "fresh_audit_manifest_path": fresh_manifest_path,
            "fresh_audit_manifest_sha256": fresh_manifest_hash,
            "source_canary_report_sha256": source_canary_hash,
            "worker_evidence": [
                json.loads(canonical_json_bytes(value)) for value in health.workers
            ],
            "worker_evidence_sha256": canonical_sha256(list(health.workers)),
        }
        return _self_hashed(payload)

    def produce_rollout_report(
        self,
        generation_id: str,
        phase: str,
        *,
        raise_on_failure: bool = False,
    ) -> ProductionResult:
        """Publish a finalized, output-derived canary or intermediate report.

        Production callers leave ``raise_on_failure`` disabled so a watch scan
        durably reports FAIL and continues with unrelated work. Strict callers
        may request an exception after the same FAIL report is published.
        """

        if phase not in {"canary", "intermediate"}:
            raise PromotionAuditorError("phase must be canary or intermediate")
        self._validate_static_inputs()
        self._prepare_output_layout()
        intent = self._load_intent(generation_id)
        health = self.derive_rollout_health(generation_id, phase)
        artifact_paths: Tuple[Path, ...] = ()
        canary_audit: Optional[Mapping[str, Any]] = None
        canary_pass: Optional[Mapping[str, Any]] = None
        reused = True
        if phase == "canary":
            audit, runner, statistics, audit_path, artifacts_reused = (
                self._canary_artifacts(intent)
            )
            canary_audit = {
                **audit,
                "fresh_audit_manifest_path": str(audit_path.resolve()),
                "fresh_audit_manifest_sha256": sha256_file(audit_path),
            }
            artifact_paths = (runner, statistics, audit_path)
            reused = artifacts_reused
        else:
            canary_pass = self._load_canary_pass(intent)

        final_health = self.derive_rollout_health(generation_id, phase)
        if final_health != health:
            raise PromotionAuditorError(
                f"{phase} worker evidence changed during report production"
            )
        report = self._rollout_report(
            intent,
            final_health,
            canary_audit=canary_audit,
            canary_pass=canary_pass,
        )
        decision = report["decision"]
        output = self.runtime.rollout_report_inbox / f"{generation_id}.{phase}.json"
        report_reused = publish_canonical_json(output, report)
        self.failure_hook(f"{phase}-rollout-report-published")
        result = ProductionResult(
            kind=f"rollout-{phase}",
            decision=decision,
            output_path=output,
            output_sha256=sha256_file(output),
            reused=reused and report_reused,
            artifact_paths=artifact_paths,
        )
        if decision == "FAIL" and raise_on_failure:
            raise AuditDecisionError(
                f"{phase} evidence derived and published a finalized FAIL report",
                evidence={
                    **report,
                    "output_path": str(output.resolve()),
                    "output_sha256": result.output_sha256,
                },
            )
        return result

    def _validate_deep_request(
        self, request_path: Path
    ) -> Tuple[Dict[str, Any], str, Tuple[EvaluationJob, ...]]:
        request_path = _require_absolute_under(
            Path(request_path), self.runtime.audit_queue, "deep-audit request"
        )
        request = _load_canonical_object(request_path, "deep-audit request")
        required_fields = {
            "schema_version",
            "contract",
            "generation_id",
            "candidate_hash",
            "previous_champion_hash",
            "policy_path",
            "policy_hash",
            "policy_version",
            "suite_manifest_path",
            "suite_manifest_hash",
            "audit_schedule_path",
            "audit_schedule_hash",
            "audit_schedule_id",
            "audit_bank_hash",
            "activation_event_hash",
            "scheduled_at_utc",
            "reasons",
            "audit_contract",
            "audit_banks",
            "visit_tiers",
            "controls",
            "control_model_hashes",
            "b28_model_path",
            "audit_cells",
        }
        if set(request) != required_fields:
            raise PromotionAuditorError(
                "deep-audit request fields differ from the v2 contract"
            )
        generation_id = _safe_generation_id(request.get("generation_id"))
        if request_path.name != f"{generation_id}.json":
            raise PromotionAuditorError(
                "deep-audit request filename contradicts generation_id"
            )
        intent = self._load_intent(generation_id)
        request_hash = sha256_file(request_path)
        expected = {
            "schema_version": 2,
            "contract": DEEP_REQUEST_CONTRACT,
            "generation_id": generation_id,
            "candidate_hash": intent["candidate_hash"],
            "previous_champion_hash": intent["tested_champion_hash"],
            "policy_path": str(self.runtime.policy_path),
            "policy_hash": self.runtime.policy_hash,
            "policy_version": POLICY_VERSION,
            "suite_manifest_path": str(self.runtime.suite_manifest_path),
            "suite_manifest_hash": self.runtime.suite_manifest_hash,
            "audit_schedule_path": str(self.runtime.audit_schedule_path),
            "audit_schedule_hash": self.runtime.audit_schedule_hash,
        }
        if any(request.get(key) != value for key, value in expected.items()):
            raise PromotionAuditorError(
                "deep-audit request contradicts controller transaction/runtime"
            )
        _sha256(
            request.get("activation_event_hash"),
            "deep-audit activation event hash",
        )
        reasons = request.get("reasons")
        if (
            not isinstance(reasons, list)
            or not reasons
            or reasons != sorted(set(reasons))
            or any(not isinstance(reason, str) or not reason for reason in reasons)
        ):
            raise PromotionAuditorError("deep-audit reasons are invalid")
        policy_contract = self.runtime.policy.get("evaluation_stages", {}).get(
            "deep_audit"
        )
        if not isinstance(policy_contract, Mapping):
            raise PromotionAuditorError("frozen policy has no deep-audit contract")
        if request.get("audit_contract") != policy_contract:
            raise PromotionAuditorError("deep-audit request policy contract changed")
        visits = request.get("visit_tiers")
        controls = request.get("controls")
        if (
            visits != policy_contract.get("visits")
            or visits != [2000, 8000]
            or controls != policy_contract.get("controls")
            or controls != ["candidate", "champion", "original", "b28"]
        ):
            raise PromotionAuditorError("deep-audit visit/control matrix is invalid")
        control_hashes = request.get("control_model_hashes")
        if not isinstance(control_hashes, Mapping):
            raise PromotionAuditorError("deep-audit control hashes are missing")
        expected_controls = {
            "candidate": intent["candidate_hash"],
            "champion": intent["tested_champion_hash"],
            "original": self.runtime.original_hash,
            "b28": control_hashes.get("b28"),
        }
        if (
            dict(control_hashes) != expected_controls
            or any(
                not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
                for value in control_hashes.values()
            )
            or len(set(control_hashes.values())) != len(control_hashes)
        ):
            raise PromotionAuditorError(
                "deep-audit control model identities are invalid"
            )
        b28_path_value = _nonempty(request.get("b28_model_path"), "deep-audit b28 path")
        b28_path = _require_absolute_under(
            Path(b28_path_value), self.runtime.promotion_root, "b28 model"
        )
        _regular_file(b28_path, control_hashes["b28"], "b28 control model")
        model_paths = {
            "candidate": self._candidate_model(generation_id, intent["candidate_hash"]),
            "champion": self._previous_champion_model(
                generation_id, intent["tested_champion_hash"]
            ),
            "original": self.runtime.original_model_path,
            "b28": b28_path,
        }

        banks = request.get("audit_banks")
        if not isinstance(banks, list) or not banks:
            raise PromotionAuditorError("deep-audit request has no audit banks")
        policy_counts = {
            "ordinary": policy_contract.get("ordinary_color_pairs"),
            "lead-40": policy_contract.get("lead_40_color_pairs"),
            "lead-80": policy_contract.get("lead_80_color_pairs"),
        }
        if [bank.get("label") for bank in banks if isinstance(bank, Mapping)] != [
            "ordinary",
            "lead-40",
            "lead-80",
        ]:
            raise PromotionAuditorError("deep-audit bank order is invalid")
        bound_banks: Dict[str, Mapping[str, Any]] = {}
        for bank in banks:
            if not isinstance(bank, Mapping):
                raise PromotionAuditorError("deep-audit bank binding is malformed")
            label = bank.get("label")
            expected_pairs = _positive_int(
                policy_counts.get(label), f"deep-audit {label} policy pairs"
            )
            schedule_path_value = _nonempty(
                bank.get("schedule_path"), f"deep-audit {label} schedule path"
            )
            schedule_path = _require_absolute_under(
                Path(schedule_path_value),
                self.runtime.suite_manifest_path.parent,
                f"deep-audit {label} schedule",
            )
            schedule_hash = _sha256(
                bank.get("schedule_hash"),
                f"deep-audit {label} schedule hash",
            )
            frozen = self._bank_for_schedule(schedule_hash, expected_path=schedule_path)
            expected_bank = {
                "label": label,
                "qualified_name": frozen["qualified_name"],
                "schedule_path": str(schedule_path),
                "schedule_hash": schedule_hash,
                "schedule_id": frozen["schedule_id"],
                "bank_hash": frozen["bank_hash"],
                "color_pairs": expected_pairs,
            }
            if dict(bank) != expected_bank or frozen["pair_count"] != expected_pairs:
                raise PromotionAuditorError(
                    f"deep-audit {label} bank contradicts frozen suite"
                )
            bound_banks[label] = {**frozen, **expected_bank}
        if (
            request.get("audit_schedule_id") != banks[0]["schedule_id"]
            or request.get("audit_bank_hash") != banks[0]["bank_hash"]
        ):
            raise PromotionAuditorError("deep-audit primary bank binding is invalid")

        request_cells = request.get("audit_cells")
        if not isinstance(request_cells, list):
            raise PromotionAuditorError("deep-audit request cells are missing")
        expected_cells = []
        for bank in banks:
            for visit_count in visits:
                for control in controls:
                    payload = {
                        "label": bank["label"],
                        "visit_count": visit_count,
                        "control": control,
                        "control_model_hash": control_hashes[control],
                        "schedule_hash": bank["schedule_hash"],
                        "bank_hash": bank["bank_hash"],
                        "color_pairs": bank["color_pairs"],
                    }
                    expected_cells.append(
                        {
                            "cell_id": "deep-audit-cell-" + canonical_sha256(payload),
                            **payload,
                        }
                    )
        if request_cells != expected_cells:
            raise PromotionAuditorError(
                "deep-audit cells do not match the frozen Cartesian matrix"
            )

        jobs = []
        for cell in expected_cells:
            bank = bound_banks[cell["label"]]
            jobs.append(
                self._build_job(
                    kind="deep-audit",
                    job_id=cell["cell_id"],
                    generation_id=generation_id,
                    candidate_hash=intent["candidate_hash"],
                    reference_hash=control_hashes[cell["control"]],
                    reference_model_path=model_paths[cell["control"]],
                    schedule_path=Path(bank["schedule_path"]),
                    schedule_hash=cell["schedule_hash"],
                    schedule_id=bank["schedule_id"],
                    suite_name=bank["name"],
                    suite_bank_hash=cell["bank_hash"],
                    color_pairs=cell["color_pairs"],
                    max_visits=cell["visit_count"],
                    request_hash=request_hash,
                    cell=cell,
                )
            )
        return request, request_hash, tuple(jobs)

    def _deep_runner_value(
        self,
        job: EvaluationJob,
        observation: EvaluationObservation,
        lease_proof: Mapping[str, Any],
    ) -> Dict[str, Any]:
        assert job.cell is not None
        assert job.request_hash is not None
        return _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": DEEP_RUNNER_CONTRACT,
                "finalized": True,
                "cell": dict(job.cell),
                "decision": observation.decision,
                "policy_hash": self.runtime.policy_hash,
                "audit_request_hash": job.request_hash,
                "results_path": str(observation.results_path.resolve()),
                "results_sha256": observation.results_sha256,
                "moves_path": str(observation.moves_path.resolve()),
                "moves_sha256": observation.moves_sha256,
                "source_runner_manifest_path": (
                    None
                    if observation.source_manifest_path is None
                    else str(observation.source_manifest_path.resolve())
                ),
                "source_runner_manifest_sha256": observation.source_manifest_sha256,
                "gpu_lease_proof": json.loads(canonical_json_bytes(lease_proof)),
                "gpu_lease_proof_sha256": canonical_sha256(lease_proof),
            }
        )

    def _validate_deep_runner(
        self, job: EvaluationJob, path: Path
    ) -> Tuple[Dict[str, Any], EvaluationObservation]:
        value = _load_canonical_object(path, "deep-audit runner manifest")
        _validate_self_hash(value, "deep-audit runner manifest")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "contract": DEEP_RUNNER_CONTRACT,
            "finalized": True,
            "cell": dict(job.cell or {}),
            "policy_hash": self.runtime.policy_hash,
            "audit_request_hash": job.request_hash,
        }
        if any(value.get(key) != item for key, item in expected.items()):
            raise PromotionAuditorError(
                "deep-audit runner manifest contradicts request"
            )
        source_value = value.get("source_runner_manifest_path")
        observation = self._observe(
            job,
            EvaluationArtifacts(
                results_path=Path(
                    _nonempty(value.get("results_path"), "deep results path")
                ),
                moves_path=Path(_nonempty(value.get("moves_path"), "deep moves path")),
                source_manifest_path=(
                    None
                    if source_value is None
                    else Path(_nonempty(source_value, "deep source manifest path"))
                ),
            ),
        )
        comparisons = {
            "decision": observation.decision,
            "results_sha256": observation.results_sha256,
            "moves_sha256": observation.moves_sha256,
            "source_runner_manifest_sha256": observation.source_manifest_sha256,
        }
        if any(value.get(key) != item for key, item in comparisons.items()):
            raise PromotionAuditorError(
                "deep-audit runner decision is not output-derived"
            )
        proof = value.get("gpu_lease_proof")
        if not isinstance(proof, Mapping) or value.get(
            "gpu_lease_proof_sha256"
        ) != canonical_sha256(proof):
            raise PromotionAuditorError("deep-audit GPU lease proof is invalid")
        return value, observation

    def _deep_statistics_value(
        self,
        job: EvaluationJob,
        observation: EvaluationObservation,
    ) -> Dict[str, Any]:
        assert job.cell is not None
        assert job.request_hash is not None
        return _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": DEEP_STATISTICS_CONTRACT,
                "finalized": True,
                "cell": dict(job.cell),
                "decision": observation.decision,
                "policy_hash": self.runtime.policy_hash,
                "audit_request_hash": job.request_hash,
                "results_sha256": observation.results_sha256,
                "moves_sha256": observation.moves_sha256,
                "safety_failures": observation.safety_failures,
                "candidate_win_rate": observation.candidate_win_rate,
                "minimum_candidate_win_rate": observation.minimum_candidate_win_rate,
                "true_no_results": observation.true_no_results,
                "true_no_result_rate": observation.true_no_result_rate,
            }
        )

    def produce_deep_audit_report(self, request_path: Path) -> ProductionResult:
        """Execute/recover an exact queued matrix and publish its v2 report."""

        self._validate_static_inputs()
        self._prepare_output_layout()
        request, request_hash, jobs = self._validate_deep_request(request_path)
        generation_id = request["generation_id"]
        artifact_root = self.runtime.audit_artifacts / generation_id
        _ensure_directory(artifact_root, self.runtime.promotion_root)

        observations: Dict[str, EvaluationObservation] = {}
        runner_values: Dict[str, Mapping[str, Any]] = {}
        runner_paths: Dict[str, Path] = {}
        pending = []
        reused = True
        for job in jobs:
            cell_root = artifact_root / job.job_id
            _ensure_directory(cell_root, self.runtime.promotion_root)
            runner_path = cell_root / "runner-manifest.json"
            runner_paths[job.job_id] = runner_path
            if runner_path.exists():
                runner_value, observation = self._validate_deep_runner(job, runner_path)
                runner_values[job.job_id] = runner_value
                observations[job.job_id] = observation
            else:
                pending.append(job)
                reused = False
        if pending:
            fresh, proof = self._run_pending_jobs(pending)
            for job in pending:
                runner_value = self._deep_runner_value(job, fresh[job.job_id], proof)
                publish_canonical_json(runner_paths[job.job_id], runner_value)
                self.failure_hook(f"deep-runner-published:{job.job_id}")
                runner_value, observation = self._validate_deep_runner(
                    job, runner_paths[job.job_id]
                )
                runner_values[job.job_id] = runner_value
                observations[job.job_id] = observation

        self._validate_static_inputs()
        if (
            sha256_file(request_path) != request_hash
            or _load_canonical_object(request_path, "deep-audit request") != request
        ):
            raise PromotionAuditorError("deep-audit request changed during evaluation")
        cells = []
        artifact_paths: List[Path] = []
        for job in jobs:
            observation = observations[job.job_id]
            runner_path = runner_paths[job.job_id]
            statistics_path = runner_path.with_name("statistics.json")
            statistics = self._deep_statistics_value(job, observation)
            statistics_reused = publish_canonical_json(statistics_path, statistics)
            reused = reused and statistics_reused
            artifact_paths.extend((runner_path, statistics_path))
            cells.append(
                {
                    **dict(job.cell or {}),
                    "decision": observation.decision,
                    "runner_manifest_path": str(runner_path.resolve()),
                    "runner_manifest_sha256": sha256_file(runner_path),
                    "statistics_artifact_path": str(statistics_path.resolve()),
                    "statistics_artifact_sha256": sha256_file(statistics_path),
                }
            )
        decision = (
            "PASS" if all(cell["decision"] == "PASS" for cell in cells) else "FAIL"
        )
        report = _self_hashed(
            {
                "schema_version": 2,
                "contract": DEEP_REPORT_CONTRACT,
                "finalized": True,
                "decision": decision,
                "rollback_required": decision == "FAIL",
                "generation_id": generation_id,
                "candidate_hash": request["candidate_hash"],
                "policy_hash": self.runtime.policy_hash,
                "audit_request_hash": request_hash,
                "control_model_hashes": dict(request["control_model_hashes"]),
                "cells": cells,
            }
        )
        if sha256_file(request_path) != request_hash:
            raise PromotionAuditorError(
                "deep-audit request changed before report publication"
            )
        output = self.runtime.audit_outbox / f"{generation_id}.json"
        report_reused = publish_canonical_json(output, report)
        self.failure_hook("deep-audit-report-published")
        return ProductionResult(
            kind="deep-audit",
            decision=decision,
            output_path=output,
            output_sha256=sha256_file(output),
            reused=reused and report_reused,
            artifact_paths=tuple(artifact_paths),
        )

    def run_once(self) -> Mapping[str, Any]:
        """Scan controller-authored work and advance every ready audit once."""

        self._validate_static_inputs()
        self._prepare_output_layout()
        produced = []
        pending = []
        for intent_path in sorted(self.runtime.transactions.glob("*/intent.json")):
            generation_id = intent_path.parent.name
            transaction = intent_path.parent
            try:
                if not (transaction / "canary-pass.json").exists():
                    produced.append(
                        self.produce_rollout_report(generation_id, "canary").to_dict()
                    )
                    continue
                if not (transaction / "intermediate-pass.json").exists():
                    produced.append(
                        self.produce_rollout_report(
                            generation_id, "intermediate"
                        ).to_dict()
                    )
            except AuditNotReady as exc:
                pending.append(
                    {
                        "kind": "rollout",
                        "generation_id": generation_id,
                        "reason": str(exc),
                    }
                )
        for request_path in sorted(self.runtime.audit_queue.glob("*.json")):
            generation_id = request_path.stem
            recorded = (
                self.runtime.promotion_root
                / "audits"
                / "reports"
                / f"{generation_id}.json"
            )
            if recorded.exists():
                continue
            produced.append(self.produce_deep_audit_report(request_path).to_dict())
        return {
            "schema_version": SCHEMA_VERSION,
            "contract": "risk-score-promotion-auditor-scan-v1",
            "produced": produced,
            "pending": pending,
        }


@contextlib.contextmanager
def configured_gpu_handoff(
    config_path: Path,
) -> Iterable[Mapping[str, Any]]:
    """Acquire the existing trainer/evaluator lease for auditor evaluations."""

    from risk_score.gpu_lease import (
        GpuLeaseManager,
        ProcessIdentity,
    )
    from risk_score.gpu_lease import (
        RuntimeConfig as GpuRuntimeConfig,
    )
    from risk_score.promotion_host import same_process

    config = GpuRuntimeConfig.from_json_file(Path(config_path))
    manager = GpuLeaseManager(config)
    proof: Dict[str, Any] = {}
    trainer_identity = None
    if manager.read_record() is None:
        identity_path = config.promotion_root / "supervisor" / "trainer.json"
        identity_value = _load_canonical_object(
            identity_path, "supervised trainer identity"
        )
        identity = identity_value.get("process_identity")
        if not isinstance(identity, Mapping) or not same_process(identity):
            raise PromotionAuditorError(
                "first auditor GPU handoff requires a live supervised trainer"
            )
        trainer_identity = ProcessIdentity.capture(identity["pid"])
    with manager.exclusive_handoff(trainer_identity) as record:
        proof.update(
            {
                "lease_id": record.lease_id,
                "expected_gpu_uuid": record.expected_gpu_uuid,
                "handoff_checkpoint_hash": (
                    None
                    if record.handoff_checkpoint is None
                    else record.handoff_checkpoint.sha256
                ),
                "clean_observations": [
                    {
                        "gpu_uuid": record.expected_gpu_uuid,
                        "processes": [],
                        "observed_at": observed_at,
                    }
                    for observed_at in record.lease_clean_observation_times
                ],
                "trainer_restored": False,
                "restored_trainer_identity": {},
            }
        )
        yield proof
    final = manager.read_record()
    if final is None:
        raise PromotionAuditorError(
            "GPU lease state disappeared during trainer restoration"
        )
    restored = final.restored_trainer or final.trainer
    proof.update(
        {
            "handoff_checkpoint_hash": (
                None
                if final.handoff_checkpoint is None
                else final.handoff_checkpoint.sha256
            ),
            "trainer_restored": final.restoration_status in {"restored", "not_needed"},
            "restored_trainer_identity": (
                {} if restored is None else restored.to_dict()
            ),
            "release_clean_observation_count": final.release_clean_observation_count,
        }
    )
    if proof["trainer_restored"] is not True:
        raise PromotionAuditorError(
            "GPU lease did not prove successful trainer restoration"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--katago", required=True, type=Path)
    parser.add_argument("--mode", choices=("once", "watch"), default="once")
    parser.add_argument("--interval", type=float)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    lock: Optional[ControllerLock] = None
    try:
        from risk_score.gpu_lease import RuntimeConfig as GpuRuntimeConfig
        from risk_score.promotion_controller import RuntimeConfig

        controller_runtime = RuntimeConfig.load(args.runtime_config)
        if not controller_runtime.controller.mutation_enabled:
            raise PromotionAuditorError(
                "production auditor requires mutationEnabled=true"
            )
        if args.interval is not None and args.interval <= 0:
            raise PromotionAuditorError("--interval must be positive")
        gpu_runtime = GpuRuntimeConfig.from_json_file(
            controller_runtime.gpu_lease_config_path
        )
        runtime = AuditorRuntime.from_controller_runtime(controller_runtime)
        auditor = PromotionAuditor(
            runtime,
            katago_binary=args.katago,
            lease_factory=lambda: configured_gpu_handoff(runtime.gpu_lease_config_path),
            shards=args.shards,
            max_parallel=args.max_parallel,
            max_attempts=args.max_attempts,
            gpu_index=gpu_runtime.gpu_index,
        )
        lock = ControllerLock(
            runtime.promotion_root / "auditor.lock",
            owner="promotion-auditor",
        ).acquire()
        interval = (
            args.interval
            if args.interval is not None
            else controller_runtime.controller.poll_interval_seconds
        )
        while True:
            result = auditor.run_once()
            print(
                json.dumps(result, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
            if args.mode != "watch":
                break
            time.sleep(interval)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if lock is not None:
            lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
