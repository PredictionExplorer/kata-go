import json
import subprocess
from pathlib import Path

import pytest

from risk_score.position_samples import canonical_json, file_sha256
from risk_score.promotion_host import HostCommandError
from risk_score.service_activation import (
    EXPECTED_SERVICE_UNITS,
    SERVICE_UNIT_NAMES,
    TARGET_UNIT,
    apply_service_activation,
    plan_service_activation,
)


def service_spec(tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    units = {}
    for name in sorted(EXPECTED_SERVICE_UNITS | {TARGET_UNIT}):
        path = generated / name
        path.write_text(
            f"[Unit]\nDescription={name}\n", encoding="utf-8"
        )
        units[name] = {"path": str(path), "sha256": file_sha256(path)}
    spec = tmp_path / "promotion-services.json"
    spec.write_text(
        canonical_json(
            {
                "schema_version": 2,
                "contract": "risk-score-host-services-v2",
                "services": {
                    name: {"argv": ["/bin/true"]}
                    for name in SERVICE_UNIT_NAMES
                },
                "systemd_units": units,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return spec, units


def test_activation_plan_is_complete_and_idempotent(tmp_path):
    spec, units = service_spec(tmp_path)
    destination = tmp_path / "systemd"
    destination.mkdir()

    first = plan_service_activation(
        spec_path=spec, destination=destination
    )
    assert first["target_unit"] == TARGET_UNIT
    assert {row["unit"] for row in first["actions"]} == set(units)
    assert {row["action"] for row in first["actions"]} == {"install"}

    for record in units.values():
        source = Path(record["path"])
        (destination / source.name).write_bytes(source.read_bytes())
    second = plan_service_activation(
        spec_path=spec, destination=destination
    )
    assert {row["action"] for row in second["actions"]} == {"unchanged"}


def test_apply_installs_enables_and_verifies_every_unit(tmp_path):
    spec, units = service_spec(tmp_path)
    destination = tmp_path / "systemd"
    destination.mkdir()
    receipt = tmp_path / "activation.json"
    commands = []

    def runner(argv, **kwargs):
        commands.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv, 0, stdout="active\n", stderr=""
        )

    result = apply_service_activation(
        spec_path=spec,
        destination=destination,
        receipt_path=receipt,
        command_runner=runner,
    )

    assert receipt.is_file()
    assert set(result["installed_units"]) == set(units)
    assert all(
        file_sha256(destination / name) == record["sha256"]
        for name, record in units.items()
    )
    argv = [row[0] for row in commands]
    assert ["systemctl", "daemon-reload"] in argv
    assert ["systemctl", "enable", "--now", TARGET_UNIT] in argv
    assert {
        command[-1]
        for command in argv
        if command[:2] == ["systemctl", "is-active"]
    } == EXPECTED_SERVICE_UNITS | {TARGET_UNIT}
    assert all(options["shell"] is False for _, options in commands)


def test_activation_refuses_changed_generated_unit(tmp_path):
    spec, units = service_spec(tmp_path)
    Path(units[next(iter(units))]["path"]).write_text(
        "changed\n", encoding="utf-8"
    )
    destination = tmp_path / "systemd"
    destination.mkdir()

    with pytest.raises(HostCommandError, match="changed or is unsafe"):
        plan_service_activation(spec_path=spec, destination=destination)


def test_activation_accepts_shadow_inventory_without_auditor(tmp_path):
    spec, units = service_spec(tmp_path)
    value = json.loads(spec.read_text())
    value["services"].pop("auditor")
    value["systemd_units"].pop("katago-risk-promotion-auditor.service")
    spec.write_text(canonical_json(value) + "\n", encoding="utf-8")
    destination = tmp_path / "systemd"
    destination.mkdir()

    plan = plan_service_activation(spec_path=spec, destination=destination)
    assert {row["unit"] for row in plan["actions"]} == (
        set(units) - {"katago-risk-promotion-auditor.service"}
    )


def test_activation_refuses_unsafe_installed_symlink(tmp_path):
    spec, _ = service_spec(tmp_path)
    destination = tmp_path / "systemd"
    destination.mkdir()
    target = tmp_path / "elsewhere"
    target.write_text("foreign\n", encoding="utf-8")
    (destination / TARGET_UNIT).symlink_to(target)

    with pytest.raises(HostCommandError, match="unsafe installed unit"):
        plan_service_activation(spec_path=spec, destination=destination)


def test_apply_stops_on_systemctl_failure_without_receipt(tmp_path):
    spec, _ = service_spec(tmp_path)
    destination = tmp_path / "systemd"
    destination.mkdir()
    receipt = tmp_path / "activation.json"

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv, 1, stdout="", stderr="denied"
        )

    with pytest.raises(HostCommandError, match="command failed"):
        apply_service_activation(
            spec_path=spec,
            destination=destination,
            receipt_path=receipt,
            command_runner=runner,
        )
    assert not receipt.exists()
