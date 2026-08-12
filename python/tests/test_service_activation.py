import json
import subprocess
from pathlib import Path

import pytest

from risk_score.position_samples import canonical_json, file_sha256
from risk_score.promotion_host import HostCommandError
from risk_score.service_activation import (
    AUTONOMY_SERVICE_SPEC_CONTRACT,
    AUTONOMY_SERVICE_UNIT_NAMES,
    EXPECTED_SERVICE_UNITS,
    FULL_EXPECTED_SERVICE_UNITS,
    FULL_SERVICE_UNIT_NAMES,
    SERVICE_UNIT_NAMES,
    TARGET_UNIT,
    apply_service_activation,
    plan_service_activation,
)


def service_spec(tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    unit_names = {**SERVICE_UNIT_NAMES, "target": TARGET_UNIT}
    unit_records = {}
    units = {}
    for key, name in sorted(unit_names.items()):
        path = generated / name
        path.write_text(
            f"[Unit]\nDescription={name}\n", encoding="utf-8"
        )
        record = {"path": str(path), "sha256": file_sha256(path)}
        unit_records[key] = record
        units[name] = record
    spec = tmp_path / "promotion-services.json"
    spec.write_text(
        canonical_json(
            {
                "schema_version": 2,
                "contract": "risk-score-host-services-v2",
                "mutation_enabled": True,
                "services": {
                    name: {"argv": ["/bin/true"]}
                    for name in SERVICE_UNIT_NAMES
                },
                "systemd_units": unit_records,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return spec, units


def shadow_service_spec(spec):
    value = json.loads(spec.read_text())
    value["mutation_enabled"] = False
    for name in ("auditor", "reconcile"):
        value["services"].pop(name)
        value["systemd_units"].pop(name)
    spec.write_text(canonical_json(value) + "\n", encoding="utf-8")


def full_autonomy_service_spec(tmp_path):
    generated = tmp_path / "generated-full"
    generated.mkdir()
    unit_names = {**FULL_SERVICE_UNIT_NAMES, "target": TARGET_UNIT}
    unit_records = {}
    units = {}
    for key, name in sorted(unit_names.items()):
        path = generated / name
        path.write_text(
            f"[Unit]\nDescription={name}\n", encoding="utf-8"
        )
        record = {"path": str(path), "sha256": file_sha256(path)}
        unit_records[key] = record
        units[name] = record
    service_inputs = {}
    inputs = tmp_path / "autonomy-inputs"
    inputs.mkdir()
    for name in (
        "autonomy_policy",
        "executor_spec",
        "adaptive_spec",
        "suite_registry_spec",
    ):
        path = inputs / f"{name}.json"
        path.write_text(canonical_json({"name": name}) + "\n", encoding="utf-8")
        service_inputs[name] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
    spec = tmp_path / "autonomy-services.json"
    spec.write_text(
        canonical_json(
            {
                "schema_version": 3,
                "contract": AUTONOMY_SERVICE_SPEC_CONTRACT,
                "mutation_enabled": True,
                "full_autonomy": True,
                "service_inputs": service_inputs,
                "services": {
                    name: {"argv": ["/bin/true"]}
                    for name in FULL_SERVICE_UNIT_NAMES
                },
                "systemd_units": unit_records,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return spec, units


def recording_runner(commands, *, inactive_units=()):
    inactive = set(inactive_units)

    def runner(argv, **kwargs):
        commands.append((argv, kwargs))
        if argv[1:2] == ["is-active"] and argv[-1] in inactive:
            return subprocess.CompletedProcess(
                argv, 3, stdout="inactive\n", stderr=""
            )
        return subprocess.CompletedProcess(
            argv, 0, stdout="active\n", stderr=""
        )

    return runner


def test_activation_plan_is_complete_and_idempotent(tmp_path):
    spec, units = service_spec(tmp_path)
    destination = tmp_path / "systemd"
    destination.mkdir()

    first = plan_service_activation(
        spec_path=spec, destination=destination
    )
    assert first["target_unit"] == TARGET_UNIT
    assert {row["unit"] for row in first["actions"]} == set(units)
    assert first["unit_inventory"] == sorted(units)
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
    assert result["unit_inventory"] == sorted(units)
    assert result["removed_units"] == {}
    assert result["restart_occurred"] is True
    assert all(
        file_sha256(destination / name) == record["sha256"]
        for name, record in units.items()
    )
    argv = [row[0] for row in commands]
    assert ["systemctl", "daemon-reload"] in argv
    assert ["systemctl", "enable", TARGET_UNIT] in argv
    assert ["systemctl", "restart", TARGET_UNIT] in argv
    assert {
        command[-1]
        for command in argv
        if command[:2] == ["systemctl", "is-active"]
    } == EXPECTED_SERVICE_UNITS | {TARGET_UNIT}
    assert all(options["shell"] is False for _, options in commands)


def test_exact_noop_only_verifies_without_restarting(tmp_path):
    spec, units = service_spec(tmp_path)
    destination = tmp_path / "systemd"
    destination.mkdir()
    receipt = tmp_path / "activation.json"
    commands = []
    runner = recording_runner(commands)
    apply_service_activation(
        spec_path=spec,
        destination=destination,
        receipt_path=receipt,
        command_runner=runner,
    )

    commands.clear()
    result = apply_service_activation(
        spec_path=spec,
        destination=destination,
        receipt_path=receipt,
        command_runner=runner,
    )

    argv = [row[0] for row in commands]
    assert result["restart_occurred"] is False
    assert result["removed_units"] == {}
    assert not any(command[1] in {"daemon-reload", "enable", "restart"} for command in argv)
    assert {
        command[-1]
        for command in argv
        if command[:2] == ["systemctl", "is-active"]
    } == set(units)
    assert ["systemctl", "is-enabled", TARGET_UNIT] in argv


def test_changed_service_spec_restarts_partof_target(tmp_path):
    spec, _ = service_spec(tmp_path)
    destination = tmp_path / "systemd"
    destination.mkdir()
    receipt = tmp_path / "activation.json"
    commands = []
    runner = recording_runner(commands)
    apply_service_activation(
        spec_path=spec,
        destination=destination,
        receipt_path=receipt,
        command_runner=runner,
    )
    value = json.loads(spec.read_text())
    value["services"]["controller"]["activation_revision"] = 2
    spec.write_text(canonical_json(value) + "\n", encoding="utf-8")

    commands.clear()
    result = apply_service_activation(
        spec_path=spec,
        destination=destination,
        receipt_path=receipt,
        command_runner=runner,
    )

    argv = [row[0] for row in commands]
    assert result["restart_occurred"] is True
    assert ["systemctl", "daemon-reload"] in argv
    assert ["systemctl", "restart", TARGET_UNIT] in argv


def test_activation_refuses_changed_generated_unit(tmp_path):
    spec, units = service_spec(tmp_path)
    Path(units[next(iter(units))]["path"]).write_text(
        "changed\n", encoding="utf-8"
    )
    destination = tmp_path / "systemd"
    destination.mkdir()

    with pytest.raises(HostCommandError, match="changed or is unsafe"):
        plan_service_activation(spec_path=spec, destination=destination)


def test_activation_accepts_exact_shadow_inventory(tmp_path):
    spec, units = service_spec(tmp_path)
    shadow_service_spec(spec)
    destination = tmp_path / "systemd"
    destination.mkdir()

    plan = plan_service_activation(spec_path=spec, destination=destination)
    assert {row["unit"] for row in plan["actions"]} == (
        set(units)
        - {
            "katago-risk-promotion-auditor.service",
            "katago-risk-boot-reconcile.service",
        }
    )


def test_activation_accepts_full_autonomy_inventory(tmp_path):
    spec, units = full_autonomy_service_spec(tmp_path)
    destination = tmp_path / "systemd"
    destination.mkdir()

    plan = plan_service_activation(
        spec_path=spec, destination=destination
    )

    assert set(plan["unit_inventory"]) == FULL_EXPECTED_SERVICE_UNITS | {
        TARGET_UNIT
    }
    assert {row["unit"] for row in plan["actions"]} == set(units)


def test_activation_refuses_incomplete_full_autonomy_inventory(tmp_path):
    spec, _ = full_autonomy_service_spec(tmp_path)
    value = json.loads(spec.read_text())
    key = next(iter(AUTONOMY_SERVICE_UNIT_NAMES))
    value["services"].pop(key)
    value["systemd_units"].pop(key)
    spec.write_text(canonical_json(value) + "\n", encoding="utf-8")
    destination = tmp_path / "systemd"
    destination.mkdir()

    with pytest.raises(HostCommandError, match="service inventory is invalid"):
        plan_service_activation(spec_path=spec, destination=destination)


def test_activation_refuses_changed_full_autonomy_input(tmp_path):
    spec, _ = full_autonomy_service_spec(tmp_path)
    value = json.loads(spec.read_text())
    path = Path(value["service_inputs"]["adaptive_spec"]["path"])
    path.write_text(canonical_json({"changed": True}) + "\n", encoding="utf-8")
    destination = tmp_path / "systemd"
    destination.mkdir()

    with pytest.raises(HostCommandError, match="changed or is unsafe"):
        plan_service_activation(spec_path=spec, destination=destination)


def test_apply_full_autonomy_verifies_every_required_service(tmp_path):
    spec, units = full_autonomy_service_spec(tmp_path)
    destination = tmp_path / "systemd"
    destination.mkdir()
    receipt = tmp_path / "activation.json"
    commands = []

    result = apply_service_activation(
        spec_path=spec,
        destination=destination,
        receipt_path=receipt,
        command_runner=recording_runner(commands),
    )

    assert set(result["installed_units"]) == set(units)
    checked = {
        command[0][-1]
        for command in commands
        if command[0][:2] == ["systemctl", "is-active"]
    }
    assert checked == FULL_EXPECTED_SERVICE_UNITS | {TARGET_UNIT}


def test_automatic_to_shadow_removes_only_omitted_owned_units(tmp_path):
    spec, units = service_spec(tmp_path)
    destination = tmp_path / "systemd"
    destination.mkdir()
    receipt = tmp_path / "activation.json"
    initial_commands = []
    apply_service_activation(
        spec_path=spec,
        destination=destination,
        receipt_path=receipt,
        command_runner=recording_runner(initial_commands),
    )
    unknown = destination / "unrelated.service"
    unknown.write_text("foreign\n", encoding="utf-8")
    shadow_service_spec(spec)
    removed = {
        "katago-risk-promotion-auditor.service",
        "katago-risk-boot-reconcile.service",
    }
    commands = []

    result = apply_service_activation(
        spec_path=spec,
        destination=destination,
        receipt_path=receipt,
        command_runner=recording_runner(commands, inactive_units=removed),
    )

    assert set(result["removed_units"]) == removed
    assert result["restart_occurred"] is True
    assert set(result["installed_units"]) == set(units) - removed
    assert all(not (destination / name).exists() for name in removed)
    assert unknown.read_text(encoding="utf-8") == "foreign\n"
    argv = [row[0] for row in commands]
    for name in removed:
        assert ["systemctl", "stop", name] in argv
        assert ["systemctl", "disable", name] in argv
        assert ["systemctl", "is-active", name] in argv
    daemon_index = argv.index(["systemctl", "daemon-reload"])
    restart_index = argv.index(["systemctl", "restart", TARGET_UNIT])
    assert all(
        daemon_index < argv.index(["systemctl", "is-active", name]) < restart_index
        for name in removed
    )
    assert not any("unrelated.service" in command for command in argv)


def test_transition_stop_failure_preserves_units_and_receipt(tmp_path):
    spec, _ = service_spec(tmp_path)
    destination = tmp_path / "systemd"
    destination.mkdir()
    receipt = tmp_path / "activation.json"
    apply_service_activation(
        spec_path=spec,
        destination=destination,
        receipt_path=receipt,
        command_runner=recording_runner([]),
    )
    previous_receipt = receipt.read_bytes()
    shadow_service_spec(spec)
    commands = []

    def runner(argv, **kwargs):
        commands.append((argv, kwargs))
        if argv[1:2] == ["stop"]:
            return subprocess.CompletedProcess(
                argv, 1, stdout="", stderr="stop failed"
            )
        return subprocess.CompletedProcess(
            argv, 0, stdout="active\n", stderr=""
        )

    with pytest.raises(HostCommandError, match="stop failed"):
        apply_service_activation(
            spec_path=spec,
            destination=destination,
            receipt_path=receipt,
            command_runner=runner,
        )
    assert (destination / "katago-risk-promotion-auditor.service").is_file()
    assert (destination / "katago-risk-boot-reconcile.service").is_file()
    assert receipt.read_bytes() == previous_receipt
    assert ["systemctl", "restart", TARGET_UNIT] not in [row[0] for row in commands]


def test_transition_refuses_restart_if_removed_unit_stays_active(tmp_path):
    spec, _ = service_spec(tmp_path)
    destination = tmp_path / "systemd"
    destination.mkdir()
    receipt = tmp_path / "activation.json"
    apply_service_activation(
        spec_path=spec,
        destination=destination,
        receipt_path=receipt,
        command_runner=recording_runner([]),
    )
    previous_receipt = receipt.read_bytes()
    shadow_service_spec(spec)
    commands = []

    with pytest.raises(HostCommandError, match="remained active"):
        apply_service_activation(
            spec_path=spec,
            destination=destination,
            receipt_path=receipt,
            command_runner=recording_runner(commands),
        )
    assert receipt.read_bytes() == previous_receipt
    assert ["systemctl", "restart", TARGET_UNIT] not in [row[0] for row in commands]


def test_noop_refuses_noncanonical_receipt_before_systemctl(tmp_path):
    spec, _ = service_spec(tmp_path)
    destination = tmp_path / "systemd"
    destination.mkdir()
    receipt = tmp_path / "activation.json"
    apply_service_activation(
        spec_path=spec,
        destination=destination,
        receipt_path=receipt,
        command_runner=recording_runner([]),
    )
    receipt.write_text(json.dumps(json.loads(receipt.read_text())), encoding="utf-8")
    commands = []

    with pytest.raises(HostCommandError, match="must be canonical JSON"):
        apply_service_activation(
            spec_path=spec,
            destination=destination,
            receipt_path=receipt,
            command_runner=recording_runner(commands),
        )
    assert commands == []


def test_activation_refuses_automatic_inventory_without_reconcile(tmp_path):
    spec, _ = service_spec(tmp_path)
    value = json.loads(spec.read_text())
    value["services"].pop("reconcile")
    value["systemd_units"].pop("reconcile")
    spec.write_text(canonical_json(value) + "\n", encoding="utf-8")
    destination = tmp_path / "systemd"
    destination.mkdir()

    with pytest.raises(HostCommandError, match="service inventory is invalid"):
        plan_service_activation(spec_path=spec, destination=destination)


def test_activation_refuses_reconcile_inventory_in_shadow(tmp_path):
    spec, _ = service_spec(tmp_path)
    value = json.loads(spec.read_text())
    value["mutation_enabled"] = False
    value["services"].pop("auditor")
    value["systemd_units"].pop("auditor")
    spec.write_text(canonical_json(value) + "\n", encoding="utf-8")
    destination = tmp_path / "systemd"
    destination.mkdir()

    with pytest.raises(HostCommandError, match="service inventory is invalid"):
        plan_service_activation(spec_path=spec, destination=destination)


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
