import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk_score import autonomy_lease_probe as probe_module
from risk_score.autonomy_lease_drill import AutonomyLeaseDrill
from risk_score.autonomy_lease_probe import (
    MAX_ANALYSIS_STDOUT_BYTES,
    PROBE_RECEIPT_CONTRACT,
    WORKER_READY_CONTRACT,
    AutonomyLeaseProbe,
    BoundedSubprocessRunner,
    CommandResult,
    GpuObservation,
    GpuProcess,
    LinuxProcessIdentity,
    NvidiaSmiComputeProbe,
    ProbeError,
    build_probe_spec,
    canonical_json,
    canonical_sha256,
    evaluator_probe_argv,
    evaluator_probe_outer_timeout_seconds,
    evaluator_sentinel_launch_argv,
    lease_publisher_commands,
    load_probe_spec,
    main,
    publish_probe_spec,
)
from risk_score.gpu_lease import ProcessIdentity

GPU_UUID = "GPU-autonomy-probe-test"
GPU_INDEX = 7
SENTINEL_PID = 401
ANALYSIS_PID = 502
DETERMINISTIC_CONFIG = """\
forDeterministicTesting = true
numAnalysisThreads = 1
numSearchThreadsPerAnalysisThread = 1
nnRandomize = false
rootNoiseEnabled = false
rootNumSymmetriesToSample = 1
useUncertainty = false
cpuctUtilityStdevScale = 0
reportAnalysisWinratesAs = SIDETOMOVE
cudaDeviceToUse = 0
"""
QUERY = {
    "boardXSize": 19,
    "boardYSize": 19,
    "id": "autonomy-lease-probe-work-1",
    "initialPlayer": "B",
    "initialStones": [],
    "komi": 7.5,
    "maxVisits": 4,
    "moves": [],
    "rules": "tromp-taylor",
}


def analysis_response(query_id=QUERY["id"]):
    return {
        "id": query_id,
        "rootInfo": {
            "scoreLead": 0.0,
            "utility": 0.0,
            "visits": 4,
            "winrate": 0.5,
        },
        "moveInfos": [{"move": "D4", "visits": 4}],
    }


def make_executable(path, contents):
    path.write_bytes(contents)
    path.chmod(0o755)
    return path


def identity(pid, *, marker="stable", process_group_id=None):
    return LinuxProcessIdentity(
        pid=pid,
        start_time_ticks=pid * 10,
        process_group_id=(
            os.getpgrp() if process_group_id is None else process_group_id
        ),
        boot_id="boot-probe-test",
        command_sha256=hashlib.sha256(f"{pid}:{marker}".encode("ascii")).hexdigest(),
        cgroup=f"/probe/{pid}/{marker}",
    )


def publish_ready(path, sentinel_identity, sentinel_module_hash):
    value = {
        "schema_version": 1,
        "contract": WORKER_READY_CONTRACT,
        "gpu_uuid": GPU_UUID,
        "gpu_index": GPU_INDEX,
        "pid": sentinel_identity.pid,
        "process_identity": sentinel_identity.to_dict(),
        "module_sha256": sentinel_module_hash,
    }
    value["ready_sha256"] = canonical_sha256(value)
    path.write_text(canonical_json(value) + "\n")
    return value


def make_fixture(
    tmp_path,
    *,
    timeout_seconds=5.0,
    cleanup_margin_seconds=1.0,
    poll_interval_seconds=0.01,
):
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    nvidia_smi = make_executable(inputs / "nvidia-smi", b"nvidia-smi")
    katago = make_executable(inputs / "katago", b"katago")
    model = inputs / "model.bin.gz"
    model.write_bytes(b"model")
    config = inputs / "analysis.cfg"
    config.write_text(DETERMINISTIC_CONFIG)
    query = inputs / "query.jsonl"
    query.write_text(canonical_json(QUERY) + "\n")
    ready = inputs / "sentinel-ready.json"
    spec_path = inputs / "probe-spec.json"
    spec = publish_probe_spec(
        spec_path,
        nvidia_smi=nvidia_smi,
        katago=katago,
        model=model,
        config=config,
        query=query,
        expected_gpu_uuid=GPU_UUID,
        expected_gpu_index=GPU_INDEX,
        sentinel_readiness_path=ready,
        timeout_seconds=timeout_seconds,
        cleanup_margin_seconds=cleanup_margin_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    sentinel = identity(SENTINEL_PID, process_group_id=SENTINEL_PID)
    publish_ready(ready, sentinel, spec.sentinel_module.sha256)
    return SimpleNamespace(
        inputs=inputs,
        nvidia_smi=nvidia_smi,
        katago=katago,
        model=model,
        config=config,
        query=query,
        ready=ready,
        spec_path=spec_path,
        spec=spec,
        sentinel=sentinel,
        child=identity(ANALYSIS_PID),
    )


def gpu_observation(*pids, gpu_uuid=GPU_UUID, gpu_index=GPU_INDEX):
    return GpuObservation(
        gpu_uuid=gpu_uuid,
        gpu_index=gpu_index,
        processes=tuple(
            GpuProcess(pid=pid, process_name=f"owner-{pid}") for pid in pids
        ),
    )


class SequenceGpuProbe:
    def __init__(self, observations, *, callbacks=None):
        self.observations = list(observations)
        self.callbacks = list(callbacks or ())
        self.calls = []

    def observe(self, *, timeout_seconds):
        self.calls.append(timeout_seconds)
        if self.callbacks:
            callback = self.callbacks.pop(0)
            if callback is not None:
                callback()
        if len(self.observations) > 1:
            return self.observations.pop(0)
        if self.observations:
            return self.observations[0]
        raise AssertionError("GPU probe called too many times")


class FakeIdentityProbe:
    def __init__(self, fixture):
        self.current = {fixture.sentinel.pid: fixture.sentinel}
        self.descendant_pids = ()

    def capture(self, pid):
        return self.current.get(pid)

    def descendants(self, pid):
        assert pid == ANALYSIS_PID
        return tuple(self.descendant_pids)


class FakeAnalysisProcess:
    def __init__(
        self,
        identities,
        child,
        *,
        statuses=None,
        result=None,
        finish_callback=None,
        reuse_after_finish=False,
    ):
        self.identities = identities
        self.child = child
        self.statuses = list(statuses or (None, 0))
        self.result = result or CommandResult(
            0, (canonical_json(analysis_response()) + "\n").encode()
        )
        self.finish_callback = finish_callback
        self.reuse_after_finish = reuse_after_finish
        self.killed = False
        identities.current[child.pid] = child

    @property
    def pid(self):
        return self.child.pid

    def poll(self):
        if len(self.statuses) > 1:
            return self.statuses.pop(0)
        return self.statuses[0]

    def finish(self, *, timeout_seconds):
        assert timeout_seconds > 0
        if self.finish_callback is not None:
            self.finish_callback()
        self.identities.current.pop(self.pid, None)
        if self.reuse_after_finish:
            self.identities.current[self.pid] = identity(self.pid, marker="reused")
        return self.result

    def kill_and_wait(self, *, timeout_seconds):
        assert timeout_seconds > 0
        self.killed = True
        if self.identities.current.get(self.pid) == self.child:
            self.identities.current.pop(self.pid)


class FakeAnalysisLauncher:
    def __init__(self, identities, child, **process_options):
        self.identities = identities
        self.child = child
        self.process_options = process_options
        self.calls = []
        self.process = None

    def start(
        self,
        argv,
        *,
        input_bytes,
        max_stdout_bytes,
        max_stderr_bytes,
        env,
    ):
        self.calls.append(
            {
                "argv": tuple(argv),
                "input_bytes": input_bytes,
                "max_stdout_bytes": max_stdout_bytes,
                "max_stderr_bytes": max_stderr_bytes,
                "env": env,
            }
        )
        self.process = FakeAnalysisProcess(
            self.identities, self.child, **self.process_options
        )
        return self.process


def happy_observations():
    return [
        gpu_observation(SENTINEL_PID),
        gpu_observation(SENTINEL_PID, ANALYSIS_PID),
        gpu_observation(SENTINEL_PID, ANALYSIS_PID),
        gpu_observation(SENTINEL_PID),
        gpu_observation(SENTINEL_PID),
        gpu_observation(SENTINEL_PID),
    ]


def run_fixture(fixture, *, observations=None, process_options=None):
    identities = FakeIdentityProbe(fixture)
    launcher = FakeAnalysisLauncher(
        identities, fixture.child, **(process_options or {})
    )
    gpu_probe = SequenceGpuProbe(observations or happy_observations())
    receipt = AutonomyLeaseProbe(
        fixture.spec,
        gpu_probe=gpu_probe,
        analysis_launcher=launcher,
        identity_probe=identities,
        sleep=lambda _seconds: None,
    ).run()
    return receipt, launcher, gpu_probe, identities


def drill_identity(pid):
    return ProcessIdentity(
        pid=pid,
        start_time_ticks=pid * 10,
        process_group_id=pid,
        boot_id="boot-probe-test",
        command_sha256=hashlib.sha256(str(pid).encode()).hexdigest(),
        cgroup=f"/probe/{pid}",
    )


def test_happy_path_receipt_lists_only_persistent_sentinel_and_is_drill_accepted(
    tmp_path,
):
    fixture = make_fixture(tmp_path)

    receipt, launcher, gpu_probe, _identities = run_fixture(fixture)

    assert receipt == {
        "schema_version": 1,
        "contract": PROBE_RECEIPT_CONTRACT,
        "gpu_uuid": GPU_UUID,
        "evaluator_pids": [SENTINEL_PID],
        "model_sha256": fixture.spec.model.sha256,
        "config_sha256": fixture.spec.config.sha256,
        "completed_work_count": 1,
        "receipt_sha256": receipt["receipt_sha256"],
    }
    payload = dict(receipt)
    assert payload.pop("receipt_sha256") == canonical_sha256(payload)
    assert len(gpu_probe.calls) == 6
    call = launcher.calls[0]
    assert call["argv"] == fixture.spec.analysis_argv
    assert call["input_bytes"] == fixture.query.read_bytes()
    assert call["env"]["CUDA_VISIBLE_DEVICES"] == GPU_UUID
    assert launcher.process.killed

    class ReceiptRunner:
        def run(self, argv, *, timeout=None):
            return SimpleNamespace(
                returncode=0,
                stdout=canonical_json(receipt) + "\n",
                stderr="",
            )

    drill_spec = SimpleNamespace(
        evaluator_probe_argv=("probe",),
        evaluator_probe_timeout_seconds=5.0,
        evaluator_probe_model_sha256=fixture.spec.model.sha256,
        evaluator_probe_config_sha256=fixture.spec.config.sha256,
        evaluator_probe_minimum_completed_work=1,
        expected_gpu_uuid=GPU_UUID,
    )
    drill = AutonomyLeaseDrill(drill_spec)
    drill._evaluator_identities = (drill_identity(SENTINEL_PID),)
    drill._run_probe(SimpleNamespace(runner=ReceiptRunner()))


def test_spec_and_publisher_commands_bind_modules_readiness_and_process_count(
    tmp_path,
):
    fixture = make_fixture(tmp_path)
    raw = json.loads(fixture.spec_path.read_bytes())
    body = dict(raw)
    supplied = body.pop("spec_sha256")

    assert supplied == canonical_sha256(body)
    assert (
        load_probe_spec(
            fixture.spec_path, expected_spec_sha256=fixture.spec.identity
        ).identity
        == fixture.spec.identity
    )
    probe_argv = evaluator_probe_argv(
        fixture.spec, python_executable=Path(sys.executable).resolve()
    )
    sentinel_argv = evaluator_sentinel_launch_argv(
        fixture.spec, python_executable=Path(sys.executable).resolve()
    )
    commands = lease_publisher_commands(
        fixture.spec, python_executable=Path(sys.executable).resolve()
    )
    assert probe_argv[1] == str(fixture.spec.probe_module.path)
    assert probe_argv[2:4] == (
        "--expected-module-sha256",
        fixture.spec.probe_module.sha256,
    )
    assert sentinel_argv[1] == str(fixture.spec.sentinel_module.path)
    assert "--ready" in sentinel_argv
    assert commands["evaluator_process_count"] == 1
    assert commands["evaluator_launch_command"] == list(sentinel_argv)
    assert commands["evaluator_probe_argv"] == list(probe_argv)
    assert commands["probe_internal_timeout_seconds"] == 5.0
    assert commands["evaluator_probe_timeout_seconds"] == 15.0
    assert (
        evaluator_probe_outer_timeout_seconds(fixture.spec)
        > fixture.spec.timeout_seconds
    )


def test_no_sentinel_gpu_owner_fails_without_launching_analysis(tmp_path):
    fixture = make_fixture(
        tmp_path,
        timeout_seconds=0.05,
        cleanup_margin_seconds=0.01,
        poll_interval_seconds=0.01,
    )
    identities = FakeIdentityProbe(fixture)
    launcher = FakeAnalysisLauncher(identities, fixture.child)

    with pytest.raises(ProbeError) as raised:
        AutonomyLeaseProbe(
            fixture.spec,
            gpu_probe=SequenceGpuProbe([gpu_observation()]),
            analysis_launcher=launcher,
            identity_probe=identities,
        ).run()

    assert raised.value.code == "probe_timeout"
    assert launcher.calls == []


def test_unknown_or_trainer_overlap_is_rejected_during_analysis(tmp_path):
    fixture = make_fixture(tmp_path)
    observations = happy_observations()
    observations[1] = gpu_observation(SENTINEL_PID, ANALYSIS_PID, 999)

    with pytest.raises(ProbeError) as raised:
        run_fixture(fixture, observations=observations)

    assert raised.value.code == "unexpected_gpu_process"


def test_wrong_uuid_or_index_fails_closed(tmp_path):
    fixture = make_fixture(tmp_path)

    with pytest.raises(ProbeError) as raised:
        run_fixture(
            fixture,
            observations=[gpu_observation(SENTINEL_PID, gpu_uuid="GPU-wrong")],
        )
    assert raised.value.code == "gpu_uuid_mismatch"

    with pytest.raises(ProbeError) as raised:
        run_fixture(
            fixture,
            observations=[gpu_observation(SENTINEL_PID, gpu_index=GPU_INDEX - 1)],
        )
    assert raised.value.code == "gpu_uuid_mismatch"


def test_sentinel_identity_churn_is_rejected(tmp_path):
    fixture = make_fixture(tmp_path)
    identities = FakeIdentityProbe(fixture)
    launcher = FakeAnalysisLauncher(identities, fixture.child)
    gpu_probe = SequenceGpuProbe(happy_observations())
    original_observe = gpu_probe.observe

    def observe(*, timeout_seconds):
        result = original_observe(timeout_seconds=timeout_seconds)
        if len(gpu_probe.calls) == 1:
            identities.current[SENTINEL_PID] = identity(
                SENTINEL_PID, marker="reused", process_group_id=SENTINEL_PID
            )
        return result

    gpu_probe.observe = observe
    with pytest.raises(ProbeError) as raised:
        AutonomyLeaseProbe(
            fixture.spec,
            gpu_probe=gpu_probe,
            analysis_launcher=launcher,
            identity_probe=identities,
            sleep=lambda _seconds: None,
        ).run()

    assert raised.value.code == "process_identity_churn"


def test_analysis_descendant_and_pid_reuse_are_rejected(tmp_path):
    fixture = make_fixture(tmp_path / "descendant")
    identities = FakeIdentityProbe(fixture)
    identities.descendant_pids = (777,)
    launcher = FakeAnalysisLauncher(identities, fixture.child)
    with pytest.raises(ProbeError) as raised:
        AutonomyLeaseProbe(
            fixture.spec,
            gpu_probe=SequenceGpuProbe(happy_observations()),
            analysis_launcher=launcher,
            identity_probe=identities,
            sleep=lambda _seconds: None,
        ).run()
    assert raised.value.code == "analysis_descendant_process"

    reused = make_fixture(tmp_path / "reused")
    with pytest.raises(ProbeError) as raised:
        run_fixture(reused, process_options={"reuse_after_finish": True})
    assert raised.value.code == "process_identity_churn"


def test_analysis_timeout_uses_cleanup_reserve_and_kills_child(tmp_path):
    fixture = make_fixture(
        tmp_path,
        timeout_seconds=0.05,
        cleanup_margin_seconds=0.01,
        poll_interval_seconds=0.005,
    )
    identities = FakeIdentityProbe(fixture)
    launcher = FakeAnalysisLauncher(identities, fixture.child, statuses=(None,))

    class Clock:
        value = 0.0

        def __call__(self):
            return self.value

        def sleep(self, duration):
            self.value += duration

    clock = Clock()
    with pytest.raises(ProbeError) as raised:
        AutonomyLeaseProbe(
            fixture.spec,
            gpu_probe=SequenceGpuProbe(
                [
                    gpu_observation(SENTINEL_PID),
                    gpu_observation(SENTINEL_PID, ANALYSIS_PID),
                ]
            ),
            analysis_launcher=launcher,
            identity_probe=identities,
            clock=clock,
            sleep=clock.sleep,
        ).run()

    assert raised.value.code == "probe_command_timeout"
    assert launcher.process.killed
    assert identities.capture(ANALYSIS_PID) is None


@pytest.mark.parametrize("role", ["model", "config"])
def test_model_or_config_drift_during_work_is_rejected(tmp_path, role):
    fixture = make_fixture(tmp_path)
    target = getattr(fixture, role)

    with pytest.raises(ProbeError) as raised:
        run_fixture(
            fixture,
            process_options={"finish_callback": lambda: target.write_bytes(b"changed")},
        )

    assert raised.value.code == "bound_input_changed"


@pytest.mark.parametrize(
    "result",
    [
        CommandResult(9, stderr=b"analysis failed"),
        CommandResult(0, stdout=b"not-json\n"),
        CommandResult(
            0,
            stdout=(canonical_json(analysis_response("wrong-id")) + "\n").encode(),
        ),
        CommandResult(0, stdout=b"x" * (MAX_ANALYSIS_STDOUT_BYTES + 1)),
    ],
)
def test_analysis_failure_and_malformed_output_are_rejected(tmp_path, result):
    fixture = make_fixture(tmp_path)

    with pytest.raises(ProbeError) as raised:
        run_fixture(fixture, process_options={"result": result})

    expected = (
        "probe_command_failed"
        if result.returncode
        else (
            "probe_output_limit"
            if len(result.stdout) > MAX_ANALYSIS_STDOUT_BYTES
            else "malformed_probe_output"
        )
    )
    assert raised.value.code == expected


@pytest.mark.parametrize(
    ("query_update", "config_text"),
    [
        ({"priority": 1}, DETERMINISTIC_CONFIG),
        ({"analyzeTurns": [0]}, DETERMINISTIC_CONFIG),
        ({"reportDuringSearchEvery": 0.1}, DETERMINISTIC_CONFIG),
        (
            {},
            DETERMINISTIC_CONFIG.replace("numSearchThreadsPerAnalysisThread = 1\n", ""),
        ),
        ({}, DETERMINISTIC_CONFIG + "unknownSetting = 1\n"),
    ],
)
def test_strict_query_and_config_allowlists(tmp_path, query_update, config_text):
    fixture = make_fixture(tmp_path)
    fixture.query.write_text(canonical_json({**QUERY, **query_update}) + "\n")
    fixture.config.write_text(config_text)

    with pytest.raises(ProbeError) as raised:
        build_probe_spec(
            nvidia_smi=fixture.nvidia_smi,
            katago=fixture.katago,
            model=fixture.model,
            config=fixture.config,
            query=fixture.query,
            expected_gpu_uuid=GPU_UUID,
            expected_gpu_index=GPU_INDEX,
            sentinel_readiness_path=fixture.ready,
        )

    assert raised.value.code == "invalid_probe_spec"


def test_nvidia_probe_filters_unrelated_mig_rows_but_rejects_unknown_physical_owner(
    tmp_path,
):
    fixture = make_fixture(tmp_path)

    class NvidiaRunner:
        def run(self, argv, **_kwargs):
            if argv[1] == "--query-gpu=index,uuid":
                return CommandResult(
                    0,
                    stdout=f"{GPU_INDEX}, {GPU_UUID}\n".encode(),
                )
            return CommandResult(
                0,
                stdout=(
                    "MIG-GPU-other/1/2, 700, mig-worker\n"
                    f"{GPU_UUID}, {SENTINEL_PID}, sentinel\n"
                ).encode(),
            )

    observation = NvidiaSmiComputeProbe(
        nvidia_smi=fixture.spec.nvidia_smi,
        expected_gpu_uuid=GPU_UUID,
        expected_gpu_index=GPU_INDEX,
        runner=NvidiaRunner(),
    ).observe(timeout_seconds=2.0)

    assert [process.pid for process in observation.processes] == [SENTINEL_PID]


def test_relative_and_symlinked_paths_are_rejected(tmp_path):
    fixture = make_fixture(tmp_path / "base")
    with pytest.raises(ProbeError) as raised:
        build_probe_spec(
            nvidia_smi=Path("relative-nvidia-smi"),
            katago=fixture.katago,
            model=fixture.model,
            config=fixture.config,
            query=fixture.query,
            expected_gpu_uuid=GPU_UUID,
            expected_gpu_index=GPU_INDEX,
            sentinel_readiness_path=fixture.ready,
        )
    assert raised.value.code == "unsafe_path"

    model_link = tmp_path / "model-link"
    model_link.symlink_to(fixture.model)
    with pytest.raises(ProbeError) as raised:
        build_probe_spec(
            nvidia_smi=fixture.nvidia_smi,
            katago=fixture.katago,
            model=model_link,
            config=fixture.config,
            query=fixture.query,
            expected_gpu_uuid=GPU_UUID,
            expected_gpu_index=GPU_INDEX,
            sentinel_readiness_path=fixture.ready,
        )
    assert raised.value.code == "unsafe_path"


def test_bounded_subprocess_runner_enforces_timeout_and_output_cap():
    python = str(Path(sys.executable).resolve())
    runner = BoundedSubprocessRunner()

    with pytest.raises(subprocess.TimeoutExpired):
        runner.run(
            (python, "-c", "import time; time.sleep(1)"),
            input_bytes=b"",
            timeout_seconds=0.05,
            max_stdout_bytes=16,
            max_stderr_bytes=16,
        )
    with pytest.raises(ProbeError) as raised:
        runner.run(
            (python, "-c", "import os; os.write(1, b'x' * 128)"),
            input_bytes=b"",
            timeout_seconds=2.0,
            max_stdout_bytes=16,
            max_stderr_bytes=16,
        )
    assert raised.value.code == "command_output_limit"


def test_cli_success_writes_only_one_canonical_receipt(monkeypatch, capsys):
    receipt = {
        "schema_version": 1,
        "contract": PROBE_RECEIPT_CONTRACT,
        "gpu_uuid": GPU_UUID,
        "evaluator_pids": [SENTINEL_PID],
        "model_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "completed_work_count": 1,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    monkeypatch.setattr(probe_module, "run_probe", lambda *_args, **_kwargs: receipt)
    monkeypatch.setattr(
        probe_module,
        "_stable_file_sha256",
        lambda *_args, **_kwargs: "3" * 64,
    )

    result = main(
        [
            "--expected-module-sha256",
            "3" * 64,
            "--spec",
            "/absolute/probe.json",
            "--expected-spec-sha256",
            "4" * 64,
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == canonical_json(receipt) + "\n"
    assert captured.err == ""
