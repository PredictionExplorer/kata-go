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
from risk_score.promotion_host import HostCommandError


SERVICE_SPEC_CONTRACT = "risk-score-host-services-v2"
TARGET_UNIT = "katago-risk-training.target"
SERVICE_UNIT_NAMES = {
    "supervisor": "katago-risk-promotion-host.service",
    "controller": "katago-risk-promotion-controller.service",
    "auditor": "katago-risk-promotion-auditor.service",
    "feedback": "katago-risk-promotion-feedback.service",
    "shuffler": "katago-risk-shuffler.service",
    "exporter": "katago-risk-exporter.service",
    "reconcile": "katago-risk-boot-reconcile.service",
}
EXPECTED_SERVICE_UNITS = frozenset(SERVICE_UNIT_NAMES.values())
REQUIRED_SERVICE_KEYS = frozenset({"supervisor", "controller", "feedback"})
SHADOW_SERVICE_KEYS = frozenset(
    {"supervisor", "controller", "feedback", "shuffler", "exporter"}
)
AUTOMATIC_SERVICE_KEYS = frozenset(SERVICE_UNIT_NAMES)
OWNED_SERVICE_UNITS = frozenset(SERVICE_UNIT_NAMES.values())
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
    mutation_enabled = value.get("mutation_enabled")
    if not isinstance(mutation_enabled, bool):
        raise HostCommandError("service specification mutation mode is invalid")
    services = value.get("services")
    if (
        not isinstance(services, Mapping)
        or not REQUIRED_SERVICE_KEYS.issubset(services)
    ):
        raise HostCommandError("service specification service inventory is invalid")
    service_keys = set(services)
    if mutation_enabled:
        valid_inventory = service_keys == AUTOMATIC_SERVICE_KEYS
    else:
        valid_inventory = service_keys.issubset(SHADOW_SERVICE_KEYS)
    if not valid_inventory:
        raise HostCommandError("service specification service inventory is invalid")
    expected_names = {SERVICE_UNIT_NAMES[name] for name in services} | {TARGET_UNIT}
    raw_units = value.get("systemd_units")
    expected_unit_keys = service_keys | {"target"}
    if not isinstance(raw_units, Mapping):
        raise HostCommandError("service specification has no systemd units")
    if set(raw_units) != expected_unit_keys:
        raise HostCommandError("service specification unit inventory is incomplete")
    units: Dict[str, Mapping[str, Any]] = {}
    for key, record in raw_units.items():
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
        expected_name = (
            TARGET_UNIT if key == "target" else SERVICE_UNIT_NAMES.get(key)
        )
        if name in units or name != expected_name or name not in expected_names:
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
    removals = []
    for name in sorted(OWNED_SERVICE_UNITS - set(units)):
        destination_path = target / name
        if not destination_path.exists() and not destination_path.is_symlink():
            continue
        if destination_path.is_symlink() or not destination_path.is_file():
            raise HostCommandError(f"unsafe installed unit path: {destination_path}")
        removals.append(
            {
                "unit": name,
                "destination": str(destination_path),
                "current_sha256": file_sha256(destination_path),
                "action": "remove",
            }
        )
    plan = {
        "schema_version": 1,
        "contract": "risk-score-systemd-activation-plan-v1",
        "service_spec": str(Path(spec_path).resolve()),
        "service_spec_sha256": file_sha256(spec_path),
        "destination": str(target.resolve()),
        "target_unit": TARGET_UNIT,
        "unit_inventory": sorted(units),
        "actions": actions,
        "removals": removals,
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


def _atomic_remove(path: Path, expected_hash: str) -> None:
    target = Path(path)
    if (
        target.is_symlink()
        or not target.is_file()
        or file_sha256(target) != expected_hash
    ):
        raise HostCommandError(f"owned unit changed before removal: {target}")
    target.unlink()
    directory = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if target.exists() or target.is_symlink():
        raise HostCommandError(f"owned unit removal failed: {target}")


def _atomic_write_receipt(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise HostCommandError(f"unsafe activation receipt path: {target}")
    data = (canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
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


def _run_inactive(
    command_runner: CommandRunner, argv: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    completed = command_runner(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    status = completed.stdout.strip()
    if completed.returncode == 0 or status not in {"inactive", "unknown", "not-found"}:
        raise HostCommandError(
            f"unit remained active ({' '.join(argv)}): "
            f"{status or completed.stderr.strip()}"
        )
    return completed


def _load_activation_receipt(path: Path) -> Optional[Mapping[str, Any]]:
    source = Path(path)
    if not source.exists() and not source.is_symlink():
        return None
    value = _load_canonical(source, "activation receipt")
    if value.get("contract") != "risk-score-systemd-activation-receipt-v1":
        raise HostCommandError("activation receipt contract is unsupported")
    payload = dict(value)
    expected_hash = payload.pop("receipt_sha256", None)
    if expected_hash != canonical_sha256(payload):
        raise HostCommandError("activation receipt self-hash is invalid")
    inventory = value.get("unit_inventory")
    installed = value.get("installed_units")
    if (
        not isinstance(inventory, list)
        or any(not isinstance(name, str) for name in inventory)
        or len(inventory) != len(set(inventory))
        or not isinstance(installed, Mapping)
        or set(installed) != set(inventory)
        or not set(inventory).issubset(OWNED_SERVICE_UNITS | {TARGET_UNIT})
    ):
        raise HostCommandError("activation receipt unit inventory is invalid")
    for name, record in installed.items():
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("sha256"), str)
        ):
            raise HostCommandError(f"activation receipt unit is malformed: {name}")
    removed = value.get("removed_units", {})
    if (
        not isinstance(removed, Mapping)
        or not set(removed).issubset(OWNED_SERVICE_UNITS)
    ):
        raise HostCommandError("activation receipt removed inventory is invalid")
    return value


def _validate_noop_receipt(
    receipt: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    if (
        receipt.get("service_spec_sha256") != plan["service_spec_sha256"]
        or receipt.get("target_unit") != TARGET_UNIT
        or receipt.get("unit_inventory") != plan["unit_inventory"]
    ):
        raise HostCommandError("activation receipt does not match service specification")
    installed = receipt["installed_units"]
    for action in plan["actions"]:
        record = installed.get(action["unit"])
        destination = Path(action["destination"])
        if (
            not isinstance(record, Mapping)
            or record.get("path") != action["destination"]
            or record.get("sha256") != action["expected_sha256"]
            or destination.is_symlink()
            or not destination.is_file()
            or file_sha256(destination) != action["expected_sha256"]
        ):
            raise HostCommandError(
                f"activation receipt installed hash mismatch: {action['unit']}"
            )


def _validate_removal_ownership(
    receipt: Optional[Mapping[str, Any]], removal: Mapping[str, Any]
) -> None:
    if receipt is None:
        raise HostCommandError("owned unit removal requires an activation receipt")
    record = receipt["installed_units"].get(removal["unit"])
    if (
        not isinstance(record, Mapping)
        or record.get("path") != removal["destination"]
        or record.get("sha256") != removal["current_sha256"]
    ):
        raise HostCommandError(
            f"activation receipt does not own removed unit: {removal['unit']}"
        )


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
    previous_receipt = _load_activation_receipt(receipt_path)
    exact_noop = (
        previous_receipt is not None
        and previous_receipt.get("service_spec_sha256")
        == plan["service_spec_sha256"]
        and not plan["removals"]
        and all(action["action"] == "unchanged" for action in plan["actions"])
    )
    removed_units = {}
    if exact_noop:
        _validate_noop_receipt(previous_receipt, plan)
    else:
        for removal in plan["removals"]:
            _validate_removal_ownership(previous_receipt, removal)
            _run(command_runner, [*systemctl, "stop", removal["unit"]])
            _run(command_runner, [*systemctl, "disable", removal["unit"]])
        for action in plan["actions"]:
            if action["action"] == "unchanged":
                continue
            _atomic_install(Path(action["source"]), Path(action["destination"]))
            if file_sha256(Path(action["destination"])) != action["expected_sha256"]:
                raise HostCommandError(f"installed unit hash mismatch: {action['unit']}")
        for removal in plan["removals"]:
            _atomic_remove(
                Path(removal["destination"]), str(removal["current_sha256"])
            )
            removed_units[removal["unit"]] = {
                "path": removal["destination"],
                "sha256": removal["current_sha256"],
            }
        _run(command_runner, [*systemctl, "daemon-reload"])
        for removal in plan["removals"]:
            destination_path = Path(removal["destination"])
            if destination_path.exists() or destination_path.is_symlink():
                raise HostCommandError(
                    f"removed unit remains installed: {removal['unit']}"
                )
            _run_inactive(
                command_runner, [*systemctl, "is-active", removal["unit"]]
            )
        _run(command_runner, [*systemctl, "enable", TARGET_UNIT])
        # Every generated service is PartOf the aggregate target. Restarting
        # applies changed unit/spec hashes and starts an inactive deployment.
        _run(command_runner, [*systemctl, "restart", TARGET_UNIT])
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
        "unit_inventory": plan["unit_inventory"],
        "removed_units": removed_units,
        "restart_occurred": not exact_noop,
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
    _atomic_write_receipt(receipt_path, receipt)
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
