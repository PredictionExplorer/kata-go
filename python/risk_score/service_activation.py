#!/usr/bin/env python3
"""Install and verify hash-bound closed-loop systemd services."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256
from risk_score.promotion_host import HostCommandError, atomic_write_json


SERVICE_SPEC_CONTRACT = "risk-score-host-services-v2"
TARGET_UNIT = "katago-risk-training.target"
EXPECTED_SERVICE_UNITS = frozenset(
    {
        "katago-risk-promotion-host.service",
        "katago-risk-promotion-controller.service",
        "katago-risk-promotion-auditor.service",
        "katago-risk-promotion-feedback.service",
        "katago-risk-shuffler.service",
        "katago-risk-exporter.service",
    }
)
SERVICE_UNIT_NAMES = {
    "supervisor": "katago-risk-promotion-host.service",
    "controller": "katago-risk-promotion-controller.service",
    "auditor": "katago-risk-promotion-auditor.service",
    "feedback": "katago-risk-promotion-feedback.service",
    "shuffler": "katago-risk-shuffler.service",
    "exporter": "katago-risk-exporter.service",
}
REQUIRED_SERVICE_KEYS = frozenset({"supervisor", "controller", "feedback"})
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _load_canonical(path: Path, role: str) -> Mapping[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise HostCommandError(f"{role} must be a regular non-symlink file")
    data = source.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HostCommandError(f"{role} is invalid JSON: {exc}") from exc
    if (
        not isinstance(value, dict)
        or data != (canonical_json(value) + "\n").encode("utf-8")
    ):
        raise HostCommandError(f"{role} must be canonical JSON")
    return value


def _validated_units(spec_path: Path) -> Dict[str, Mapping[str, Any]]:
    value = _load_canonical(spec_path, "service specification")
    if value.get("contract") != SERVICE_SPEC_CONTRACT:
        raise HostCommandError("service specification contract is unsupported")
    services = value.get("services")
    if (
        not isinstance(services, Mapping)
        or not REQUIRED_SERVICE_KEYS.issubset(services)
        or not set(services).issubset(SERVICE_UNIT_NAMES)
    ):
        raise HostCommandError("service specification service inventory is invalid")
    expected_names = {SERVICE_UNIT_NAMES[name] for name in services} | {TARGET_UNIT}
    raw_units = value.get("systemd_units")
    if not isinstance(raw_units, Mapping):
        raise HostCommandError("service specification has no systemd units")
    units: Dict[str, Mapping[str, Any]] = {}
    for record in raw_units.values():
        if not isinstance(record, Mapping):
            raise HostCommandError("systemd unit record is malformed")
        raw_path = record.get("path")
        expected_hash = record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
            raise HostCommandError("systemd unit identity is incomplete")
        path = Path(raw_path)
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or file_sha256(path) != expected_hash
        ):
            raise HostCommandError(f"systemd unit changed or is unsafe: {path}")
        name = path.name
        if name in units or name not in expected_names:
            raise HostCommandError(f"unexpected or duplicate systemd unit: {name}")
        units[name] = {
            "source": str(path.resolve()),
            "sha256": expected_hash,
        }
    if set(units) != expected_names:
        raise HostCommandError("service specification unit inventory is incomplete")
    return units


def plan_service_activation(
    *, spec_path: Path, destination: Path
) -> Mapping[str, Any]:
    target = Path(destination)
    if not target.is_absolute() or target.is_symlink() or not target.is_dir():
        raise HostCommandError(
            "systemd destination must be an absolute non-symlink directory"
        )
    units = _validated_units(spec_path)
    actions = []
    for name in sorted(units):
        destination_path = target / name
        if destination_path.is_symlink() or (
            destination_path.exists() and not destination_path.is_file()
        ):
            raise HostCommandError(f"unsafe installed unit path: {destination_path}")
        current_hash = (
            file_sha256(destination_path) if destination_path.is_file() else None
        )
        actions.append(
            {
                "unit": name,
                "source": units[name]["source"],
                "destination": str(destination_path),
                "expected_sha256": units[name]["sha256"],
                "current_sha256": current_hash,
                "action": (
                    "unchanged"
                    if current_hash == units[name]["sha256"]
                    else "update"
                    if current_hash is not None
                    else "install"
                ),
            }
        )
    plan = {
        "schema_version": 1,
        "contract": "risk-score-systemd-activation-plan-v1",
        "service_spec": str(Path(spec_path).resolve()),
        "service_spec_sha256": file_sha256(spec_path),
        "destination": str(target.resolve()),
        "target_unit": TARGET_UNIT,
        "actions": actions,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def _atomic_install(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as source_handle, temporary.open("wb") as output:
            shutil.copyfileobj(source_handle, output, 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _run(
    command_runner: CommandRunner, argv: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    completed = command_runner(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        raise HostCommandError(
            f"command failed ({' '.join(argv)}): {completed.stderr.strip()}"
        )
    return completed


def apply_service_activation(
    *,
    spec_path: Path,
    destination: Path,
    receipt_path: Path,
    systemctl: Sequence[str] = ("systemctl",),
    command_runner: CommandRunner = subprocess.run,
) -> Mapping[str, Any]:
    if not systemctl or any(not isinstance(part, str) or not part for part in systemctl):
        raise HostCommandError("systemctl command must be a nonempty argv sequence")
    plan = plan_service_activation(spec_path=spec_path, destination=destination)
    for action in plan["actions"]:
        if action["action"] == "unchanged":
            continue
        _atomic_install(Path(action["source"]), Path(action["destination"]))
        if file_sha256(Path(action["destination"])) != action["expected_sha256"]:
            raise HostCommandError(f"installed unit hash mismatch: {action['unit']}")
    _run(command_runner, [*systemctl, "daemon-reload"])
    _run(command_runner, [*systemctl, "enable", "--now", TARGET_UNIT])
    active = {}
    installed_service_units = sorted(
        action["unit"]
        for action in plan["actions"]
        if action["unit"].endswith(".service")
    )
    for unit in [TARGET_UNIT, *installed_service_units]:
        result = _run(command_runner, [*systemctl, "is-active", unit])
        active[unit] = result.stdout.strip() or "active"
    _run(command_runner, [*systemctl, "is-enabled", TARGET_UNIT])
    receipt = {
        "schema_version": 1,
        "contract": "risk-score-systemd-activation-receipt-v1",
        "plan_sha256": plan["plan_sha256"],
        "service_spec_sha256": plan["service_spec_sha256"],
        "target_unit": TARGET_UNIT,
        "installed_units": {
            action["unit"]: {
                "path": action["destination"],
                "sha256": file_sha256(Path(action["destination"])),
            }
            for action in plan["actions"]
        },
        "active": active,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    atomic_write_json(receipt_path, receipt)
    return receipt


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument(
        "--destination", type=Path, default=Path("/etc/systemd/system")
    )
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.apply:
            if args.receipt is None:
                raise HostCommandError("--receipt is required with --apply")
            result = apply_service_activation(
                spec_path=args.spec,
                destination=args.destination,
                receipt_path=args.receipt,
            )
        else:
            result = plan_service_activation(
                spec_path=args.spec, destination=args.destination
            )
    except (OSError, ValueError, HostCommandError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
