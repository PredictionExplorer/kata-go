import copy
import json
import signal
import sys
from pathlib import Path

import pytest
from katago.train.training_controls import build_validation_manifest
from risk_score.adaptive_training import (
    DEFAULT_POLICY_PATH,
    TRIAL_RESULT_CONTRACT,
    AdaptiveTrainingError,
    AdaptiveTrainingStore,
    atomic_create_json,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    load_policy,
    load_trial_result,
)
from risk_score.adaptive_trial_worker import (
    COMMAND_RECEIPT_CONTRACT,
    WORKER_SPEC_CONTRACT,
    AdaptiveTrialWorker,
    AmbiguousTrialState,
    FrozenInputChanged,
    WorkerSpecError,
    load_worker_spec,
    parse_args,
    publish_command_receipt,
    publish_curriculum_manifest,
    publish_worker_spec,
    translate_recipe,
)


def digest(label):
    return canonical_sha256({"label": label})


class FakeClock:
    def __init__(self, value=1_900_000_000.0):
        self.value = float(value)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)

    def sleep(self, seconds):
        self.advance(seconds)


class FakeProcess:
    def __init__(
        self,
        runner,
        stage,
        *,
        returncode=0,
        wait_for_sigint=False,
        stuck_after_sigint=False,
    ):
        self.runner = runner
        self.stage = stage
        self._returncode = returncode
        self.wait_for_sigint = wait_for_sigint
        self.stuck_after_sigint = stuck_after_sigint
        self.signaled = False

    def poll(self):
        if self.wait_for_sigint and not self.signaled:
            return None
        if self.stuck_after_sigint:
            return None
        return self._returncode

    def send_signal(self, sig):
        assert sig != signal.SIGKILL
        self.runner.signals.append((self.stage, sig))
        self.signaled = True
        if self.stage == "trainer" and not self.stuck_after_sigint:
            path = self.runner.worker.paths.checkpoint
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"stable drained checkpoint")
            self._returncode = -signal.SIGINT
            environment = self.runner.environments[-1]
            publish_command_receipt(
                Path(environment["RISK_SCORE_ADAPTIVE_RECEIPT_PATH"]),
                stage=self.stage,
                worker_spec_sha256=self.runner.worker.spec.spec_sha256,
                trial_manifest_path=self.runner.worker.context.path,
                trial_manifest_sha256=(self.runner.worker.context.manifest_sha256),
                trial_id=self.runner.worker.context.trial_id,
                work_id=self.runner.worker.work_id,
                round_index=self.runner.worker.round_index,
                argv=self.runner.argv[-1],
                inputs_sha256=environment["RISK_SCORE_ADAPTIVE_INPUTS_SHA256"],
                returncode=self._returncode,
                status="drained",
                outputs={},
            )

    def process_group_alive(self):
        return self.poll() is None


class FakeCommandRunner:
    def __init__(
        self,
        clock,
        *,
        failing_stage=None,
        invalid_evidence_stage=None,
        timeout_trainer=False,
        stuck_trainer=False,
        fail_on_spawn=False,
        tamper_receipt_stage=None,
    ):
        self.clock = clock
        self.failing_stage = failing_stage
        self.invalid_evidence_stage = invalid_evidence_stage
        self.timeout_trainer = timeout_trainer
        self.stuck_trainer = stuck_trainer
        self.fail_on_spawn = fail_on_spawn
        self.tamper_receipt_stage = tamper_receipt_stage
        self.worker = None
        self.stages = []
        self.argv = []
        self.environments = []
        self.signals = []

    @staticmethod
    def binding(path, *, resumable=False):
        value = {"path": str(path), "sha256": file_sha256(path)}
        if resumable:
            value["resumable"] = True
        return value

    def _evidence(self, stage):
        worker = self.worker
        source = "confirmation" if self.invalid_evidence_stage == stage else stage
        metric = (
            "discovery_powered_terminal_utility"
            if source == "discovery"
            else "fixed_validation_loss"
        )
        return {
            "artifact_sha256": digest(f"{stage}-artifact"),
            "finalized": True,
            "metrics": {metric: 3.5 if stage == "discovery" else 0.25},
            "round_index": worker.round_index,
            "sample_count": 128,
            "schema_version": 1,
            "source": source,
            "trial_id": worker.context.trial_id,
        }

    def _completed_outputs(self, stage):
        worker = self.worker
        paths = worker.paths
        if stage == "curriculum":
            paths.curriculum_directory.mkdir(parents=True, exist_ok=True)
            (paths.curriculum_directory / "train-000.npz").write_bytes(
                b"bounded curriculum"
            )
            publish_curriculum_manifest(
                paths.curriculum_manifest,
                directory=paths.curriculum_directory,
                worker_spec_sha256=worker.spec.spec_sha256,
                trial_manifest_sha256=worker.context.manifest_sha256,
                trial_id=worker.context.trial_id,
                round_index=worker.round_index,
                admitted_data_manifest=(
                    worker.context.admitted_data_manifest.to_dict()
                ),
                recipe_sha256=worker.context.recipe_sha256,
                shuffle_argv=worker.recipe_arguments.shuffle_argv,
            )
            return {"curriculum_manifest": self.binding(paths.curriculum_manifest)}
        if stage == "trainer":
            paths.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            paths.checkpoint.write_bytes(b"stable bounded checkpoint")
            return {
                "checkpoint": self.binding(
                    paths.checkpoint,
                    resumable=True,
                )
            }
        if stage == "export":
            paths.candidate_model.parent.mkdir(parents=True, exist_ok=True)
            paths.candidate_model.write_bytes(b"candidate model")
            paths.candidate_checkpoint.write_bytes(b"candidate checkpoint")
            return {
                "candidate_model": self.binding(paths.candidate_model),
                "candidate_checkpoint": self.binding(
                    paths.candidate_checkpoint,
                    resumable=True,
                ),
            }
        if stage == "model_probe":
            probe = {
                "config_sha256": digest("probe-config"),
                "contract": "risk-score-model-probe-v1",
                "finite": True,
                "gpu_uuid": "GPU-test",
                "katago_sha256": worker.spec.katago_binary.sha256,
                "model_sha256": file_sha256(paths.candidate_model),
                "schema_version": 1,
            }
            atomic_create_json(paths.model_probe, probe)
            return {"probe": self.binding(paths.model_probe)}
        if stage in {"fixed_validation", "discovery"}:
            path = (
                paths.fixed_validation_evidence
                if stage == "fixed_validation"
                else paths.discovery_evidence
            )
            atomic_create_json(path, self._evidence(stage))
            return {"evidence": self.binding(path)}
        raise AssertionError(stage)

    def spawn(self, argv, *, cwd, environment, log_path):
        if self.fail_on_spawn:
            raise AssertionError("read-only replay attempted a command")
        worker = self.worker
        assert worker is not None
        stage = environment["RISK_SCORE_ADAPTIVE_STAGE"]
        assert cwd == worker.paths.round_root
        assert environment["CUDA_VISIBLE_DEVICES"] == "7"
        assert environment["RISK_SCORE_ADAPTIVE_WORK_ID"] == worker.work_id
        assert (
            environment["RISK_SCORE_ADAPTIVE_WORKER_SPEC_SHA256"]
            == worker.spec.spec_sha256
        )
        self.stages.append(stage)
        self.argv.append(tuple(argv))
        self.environments.append(dict(environment))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.clock.advance(0.5)

        if stage == "trainer" and (self.timeout_trainer or self.stuck_trainer):
            return FakeProcess(
                self,
                stage,
                wait_for_sigint=True,
                stuck_after_sigint=self.stuck_trainer,
            )

        returncode = 9 if stage == self.failing_stage else 0
        outputs = {} if returncode else self._completed_outputs(stage)
        status = "failed" if returncode else "completed"
        receipt = publish_command_receipt(
            Path(environment["RISK_SCORE_ADAPTIVE_RECEIPT_PATH"]),
            stage=stage,
            worker_spec_sha256=worker.spec.spec_sha256,
            trial_manifest_path=worker.context.path,
            trial_manifest_sha256=worker.context.manifest_sha256,
            trial_id=worker.context.trial_id,
            work_id=worker.work_id,
            round_index=worker.round_index,
            argv=argv,
            inputs_sha256=environment["RISK_SCORE_ADAPTIVE_INPUTS_SHA256"],
            returncode=returncode,
            status=status,
            outputs=outputs,
        )
        assert receipt["contract"] == COMMAND_RECEIPT_CONTRACT
        if stage == self.tamper_receipt_stage:
            receipt_path = Path(environment["RISK_SCORE_ADAPTIVE_RECEIPT_PATH"])
            receipt_path.chmod(0o600)
            changed = dict(receipt)
            changed["inputs_sha256"] = "0" * 64
            receipt_path.write_bytes(canonical_json_bytes(changed) + b"\n")
        return FakeProcess(self, stage, returncode=returncode)


def command_template(stage):
    common = [
        "{python_executable}",
        "-m",
        f"test_adapter.{stage}",
        "--stage",
        "{stage}",
        "--receipt",
        "{receipt_path}",
        "--worker-spec-sha256",
        "{worker_spec_sha256}",
        "--trial-manifest-sha256",
        "{trial_manifest_sha256}",
        "--work-id",
        "{work_id}",
        "--round-index",
        "{round_index}",
        "--inputs-sha256",
        "{inputs_sha256}",
    ]
    stage_values = {
        "curriculum": [
            "--admitted-data",
            "{admitted_data_manifest_path}",
            "--admitted-data-sha256",
            "{admitted_data_manifest_sha256}",
            "--output",
            "{curriculum_data_path}",
            "--manifest",
            "{curriculum_manifest_path}",
        ],
        "trainer": [
            "--curriculum",
            "{curriculum_manifest_path}",
            "--initial-checkpoint",
            "{initial_checkpoint_path}",
            "--initial-checkpoint-sha256",
            "{initial_checkpoint_sha256}",
            "--checkpoint",
            "{checkpoint_path}",
            "--reservation",
            "{reservation_gpu_seconds}",
            "--deadline",
            "{deadline_unix}",
        ],
        "export": [
            "--checkpoint",
            "{checkpoint_path}",
            "--model",
            "{candidate_model_path}",
            "--candidate-checkpoint",
            "{candidate_checkpoint_path}",
        ],
        "model_probe": [
            "--katago",
            "{katago_binary}",
            "--model",
            "{candidate_model_path}",
            "--output",
            "{model_probe_path}",
        ],
        "fixed_validation": [
            "--katago",
            "{katago_binary}",
            "--model",
            "{candidate_model_path}",
            "--validation-manifest",
            "{fixed_validation_manifest_path}",
            "--validation-manifest-sha256",
            "{fixed_validation_manifest_sha256}",
            "--evidence",
            "{fixed_validation_evidence_path}",
        ],
        "discovery": [
            "--katago",
            "{katago_binary}",
            "--model",
            "{candidate_model_path}",
            "--evidence",
            "{discovery_evidence_path}",
        ],
    }
    return common + stage_values[stage]


def make_fixture(tmp_path, *, deadline_offset=100.0):
    clock = FakeClock()
    repository = (tmp_path / "deployed-repo").resolve()
    repository.mkdir()
    source_revision = "a" * 40

    katago = (tmp_path / "katago").resolve()
    katago.write_bytes(b"#!/bin/sh\nexit 0\n")
    katago.chmod(0o755)

    validation_directory = (tmp_path / "fixed-validation").resolve()
    validation_directory.mkdir()
    (validation_directory / "validation.npz").write_bytes(b"fixed")
    validation_manifest = (tmp_path / "fixed-validation.json").resolve()
    build_validation_manifest(validation_directory, validation_manifest)

    spec_path = (tmp_path / "adaptive-trial-worker.json").resolve()
    revision_reader = lambda _path: source_revision
    status_reader = lambda _path: ""
    templates = {
        stage: command_template(stage)
        for stage in (
            "curriculum",
            "trainer",
            "export",
            "model_probe",
            "fixed_validation",
            "discovery",
        )
    }
    spec = publish_worker_spec(
        spec_path,
        repository_path=repository,
        source_revision=source_revision,
        autonomy_policy_path=DEFAULT_POLICY_PATH.resolve(),
        python_executable=Path(sys.executable).resolve(),
        katago_binary=katago,
        fixed_validation_manifest_path=validation_manifest,
        curriculum_argv_template=templates["curriculum"],
        trainer_argv_template=templates["trainer"],
        export_argv_template=templates["export"],
        model_probe_argv_template=templates["model_probe"],
        fixed_validation_argv_template=templates["fixed_validation"],
        discovery_argv_template=templates["discovery"],
        poll_interval_seconds=0.1,
        drain_timeout_seconds=1.0,
        checkpoint_timeout_seconds=1.0,
        checkpoint_stable_seconds=0.0,
        revision_reader=revision_reader,
        repository_status_reader=status_reader,
    )

    champion = (tmp_path / "champion.ckpt").resolve()
    champion.write_bytes(b"immutable champion checkpoint")
    admitted = (tmp_path / "admitted.json").resolve()
    admitted.write_bytes(b'{"snapshot":"admitted"}\n')
    store = AdaptiveTrainingStore((tmp_path / "adaptive").resolve())
    plan = store.plan_epoch(
        admitted_samples=3_000_000,
        last_promotion_admitted_samples=0,
        candidate_queue_depth=0,
        parent_champion_model_sha256=digest("champion-model"),
        champion_checkpoint_path=champion,
        admitted_data_manifest_path=admitted,
        now=clock(),
    )
    trial_id = plan["trial_ids"][0]
    manifest_path = store.trials_dir / trial_id / "trial.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_path = Path(manifest["isolation_root"]) / "results" / "round-00.json"
    return {
        "admitted": admitted,
        "champion": champion,
        "clock": clock,
        "deadline": clock() + deadline_offset,
        "katago": katago,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "repository": repository,
        "result_path": result_path,
        "revision_reader": revision_reader,
        "source_revision": source_revision,
        "spec": spec,
        "spec_path": spec_path,
        "status_reader": status_reader,
        "templates": templates,
    }


def make_worker(fixture, runner):
    worker = AdaptiveTrialWorker(
        fixture["spec"],
        trial_manifest_path=fixture["manifest_path"],
        expected_trial_manifest_sha256=fixture["manifest"]["manifest_sha256"],
        work_id="adaptive-work-a",
        round_index=0,
        reservation_gpu_seconds=14_400,
        deadline_unix=fixture["deadline"],
        result_path=fixture["result_path"],
        command_runner=runner,
        clock=fixture["clock"],
        sleeper=fixture["clock"].sleep,
        revision_reader=fixture["revision_reader"],
        repository_status_reader=fixture["status_reader"],
        install_signal_handlers=False,
    )
    runner.worker = worker
    return worker


def write_spec(path, value):
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def test_recipe_translation_is_complete_and_rejects_protected_surfaces():
    policy = load_policy()
    recipe = {key: values[0] for key, values in policy["allowed_recipe_knobs"].items()}

    translated = translate_recipe(recipe, policy=policy)

    assert translated.recipe_sha256 == canonical_sha256(recipe)
    assert translated.shuffle_argv == (
        "--recent-window-samples",
        "3000000",
        "--recent-fraction",
        "0.75",
        "--historical-window-samples",
        "24000000",
        "--historical-fraction",
        "0.25",
    )
    assert translated.trainer_argv == (
        "--bucket-cap-samples",
        "1000000",
        "--bucket-ratio",
        "0.75",
        "--export-cadence-epochs",
        "1",
        "--learning-rate-scale",
        "0.75",
        "--learning-rate-schedule",
        "frozen-baseline",
        "--swa-cadence-samples",
        "250000",
    )

    for forbidden in (
        "objective",
        "game_rules",
        "architecture",
        "promotion_thresholds",
        "confirmation_inputs",
        "audit_inputs",
    ):
        changed = dict(recipe)
        changed[forbidden] = "forbidden"
        with pytest.raises(AdaptiveTrainingError) as raised:
            translate_recipe(changed, policy=policy)
        assert raised.value.code == "recipe_surface_forbidden"


@pytest.mark.parametrize(
    "argument,code",
    [
        ("{objective}", "forbidden_placeholder"),
        ("--confirmation-suite", "protected_surface_forbidden"),
        ("--bucket-ratio", "recipe_flag_in_template"),
    ],
)
def test_worker_spec_rejects_forbidden_placeholders_and_surfaces(
    tmp_path,
    argument,
    code,
):
    fixture = make_fixture(tmp_path)
    raw = copy.deepcopy(dict(fixture["spec"].raw))
    raw["trainer_argv_template"].append(argument)
    raw.pop("spec_sha256")
    raw["spec_sha256"] = canonical_sha256(raw)
    invalid = (tmp_path / f"invalid-{code}.json").resolve()
    write_spec(invalid, raw)

    with pytest.raises(WorkerSpecError) as raised:
        load_worker_spec(
            invalid,
            expected_spec_sha256=raw["spec_sha256"],
            revision_reader=fixture["revision_reader"],
            repository_status_reader=fixture["status_reader"],
        )
    assert raised.value.code == code


def test_success_runs_exact_stage_order_and_publishes_bound_result(tmp_path):
    fixture = make_fixture(tmp_path)
    runner = FakeCommandRunner(fixture["clock"])
    worker = make_worker(fixture, runner)

    result = worker.run()

    assert runner.stages == [
        "curriculum",
        "trainer",
        "export",
        "model_probe",
        "fixed_validation",
        "discovery",
    ]
    assert result["contract"] == TRIAL_RESULT_CONTRACT
    assert result["status"] == "completed"
    assert result["gpu_usage"]["gpu_id"] == "7"
    assert result["gpu_usage"]["ended_at_unix"] - result["gpu_usage"][
        "started_at_unix"
    ] == pytest.approx(3.0)
    assert {item["source"] for item in result["evidence"]} == {
        "fixed_validation",
        "discovery",
    }
    assert result["candidate_model"]["path"] == str(worker.paths.candidate_model)
    assert result["candidate_checkpoint"] == {
        "path": str(worker.paths.candidate_checkpoint),
        "resumable": True,
        "sha256": file_sha256(worker.paths.candidate_checkpoint),
    }
    loaded = load_trial_result(
        fixture["result_path"],
        expected_trial_id=fixture["manifest"]["trial_id"],
        expected_round_index=0,
        expected_work_id="adaptive-work-a",
        expected_gpu_id="7",
        expected_manifest_path=fixture["manifest_path"],
        expected_manifest_sha256=fixture["manifest"]["manifest_sha256"],
    )
    assert loaded == result
    assert all("--bucket-cap-samples" not in argv for argv in runner.argv[:-5])
    assert "--bucket-cap-samples" in runner.argv[1]
    assert "--recent-window-samples" in runner.argv[0]


def test_deadline_sends_sigint_drains_checkpoint_and_never_sigkills(tmp_path):
    fixture = make_fixture(tmp_path, deadline_offset=1.0)
    runner = FakeCommandRunner(fixture["clock"], timeout_trainer=True)
    worker = make_worker(fixture, runner)

    result = worker.run()

    assert runner.stages == ["curriculum", "trainer"]
    assert runner.signals == [("trainer", signal.SIGINT)]
    assert all(sig != signal.SIGKILL for _, sig in runner.signals)
    assert result["status"] == "failed"
    assert result["failure_reason"] == "gpu_deadline_exhausted"
    assert worker.paths.checkpoint.read_bytes() == b"stable drained checkpoint"
    assert result["gpu_usage"]["ended_at_unix"] <= fixture["deadline"]


def test_unfinished_graceful_drain_fails_closed_without_result(tmp_path):
    fixture = make_fixture(tmp_path, deadline_offset=1.0)
    runner = FakeCommandRunner(fixture["clock"], stuck_trainer=True)
    worker = make_worker(fixture, runner)

    with pytest.raises(AmbiguousTrialState) as raised:
        worker.run()

    assert raised.value.code == "graceful_drain_timeout"
    assert runner.signals == [("trainer", signal.SIGINT)]
    assert not fixture["result_path"].exists()


def test_invalid_holdout_evidence_is_not_finalized_into_result(tmp_path):
    fixture = make_fixture(tmp_path)
    runner = FakeCommandRunner(
        fixture["clock"],
        invalid_evidence_stage="fixed_validation",
    )
    worker = make_worker(fixture, runner)

    result = worker.run()

    assert runner.stages == [
        "curriculum",
        "trainer",
        "export",
        "model_probe",
        "fixed_validation",
    ]
    assert result["status"] == "failed"
    assert result["failure_reason"] == "invalid_tuning_evidence"
    assert result["evidence"] == []
    assert result["candidate_model"] is None
    assert result["candidate_checkpoint"] is None


def test_command_receipt_hash_drift_stops_before_next_stage(tmp_path):
    fixture = make_fixture(tmp_path)
    runner = FakeCommandRunner(
        fixture["clock"],
        tamper_receipt_stage="curriculum",
    )
    worker = make_worker(fixture, runner)

    result = worker.run()

    assert runner.stages == ["curriculum"]
    assert result["status"] == "failed"
    assert result["failure_reason"] == "invalid_command_receipt"
    assert result["evidence"] == []


def test_exact_result_replay_is_read_only_and_spawns_nothing(tmp_path):
    fixture = make_fixture(tmp_path)
    first_runner = FakeCommandRunner(fixture["clock"])
    first_worker = make_worker(fixture, first_runner)
    first = first_worker.run()
    before_bytes = fixture["result_path"].read_bytes()
    before_mtime = fixture["result_path"].stat().st_mtime_ns

    replay_runner = FakeCommandRunner(
        fixture["clock"],
        fail_on_spawn=True,
    )
    replay_worker = make_worker(fixture, replay_runner)
    replay = replay_worker.run()

    assert replay == first
    assert replay_runner.stages == []
    assert fixture["result_path"].read_bytes() == before_bytes
    assert fixture["result_path"].stat().st_mtime_ns == before_mtime


def test_crash_after_bound_receipt_resumes_without_duplicate_command(tmp_path):
    class SimulatedCrash(BaseException):
        pass

    fixture = make_fixture(tmp_path)
    original_start = fixture["clock"]()
    first_runner = FakeCommandRunner(fixture["clock"])
    first_worker = make_worker(fixture, first_runner)
    execute = first_worker._execute_stage

    def crash_after_curriculum(stage):
        receipt = execute(stage)
        if stage == "curriculum":
            raise SimulatedCrash
        return receipt

    first_worker._execute_stage = crash_after_curriculum
    with pytest.raises(SimulatedCrash):
        first_worker.run()
    assert first_runner.stages == ["curriculum"]
    assert not fixture["result_path"].exists()

    replay_runner = FakeCommandRunner(fixture["clock"])
    replay_worker = make_worker(fixture, replay_runner)
    result = replay_worker.run()

    assert replay_runner.stages == [
        "trainer",
        "export",
        "model_probe",
        "fixed_validation",
        "discovery",
    ]
    assert result["status"] == "completed"
    assert result["gpu_usage"]["started_at_unix"] == original_start
    assert result["gpu_usage"]["ended_at_unix"] - result["gpu_usage"][
        "started_at_unix"
    ] == pytest.approx(3.0)


def test_later_round_resumes_only_from_prior_bound_candidate_checkpoint(tmp_path):
    fixture = make_fixture(tmp_path)
    first_runner = FakeCommandRunner(fixture["clock"])
    first_worker = make_worker(fixture, first_runner)
    first = first_worker.run()
    prior_checkpoint = Path(first["candidate_checkpoint"]["path"])

    second_runner = FakeCommandRunner(fixture["clock"])
    second_result = (
        Path(fixture["manifest"]["isolation_root"]) / "results" / "round-01.json"
    )
    second = AdaptiveTrialWorker(
        fixture["spec"],
        trial_manifest_path=fixture["manifest_path"],
        expected_trial_manifest_sha256=fixture["manifest"]["manifest_sha256"],
        work_id="adaptive-work-b",
        round_index=1,
        reservation_gpu_seconds=28_800,
        deadline_unix=fixture["clock"]() + 100,
        result_path=second_result,
        command_runner=second_runner,
        clock=fixture["clock"],
        sleeper=fixture["clock"].sleep,
        revision_reader=fixture["revision_reader"],
        repository_status_reader=fixture["status_reader"],
        install_signal_handlers=False,
    )
    second_runner.worker = second

    result = second.run()

    assert result["status"] == "completed"
    assert second._initial_checkpoint.path == prior_checkpoint
    assert second._initial_checkpoint.sha256 == first["candidate_checkpoint"]["sha256"]
    trainer_argv = second_runner.argv[1]
    initial_index = trainer_argv.index("--initial-checkpoint") + 1
    assert trainer_argv[initial_index] == str(prior_checkpoint)


def test_child_failure_publishes_fail_result_only_after_stable_trainer(tmp_path):
    fixture = make_fixture(tmp_path)
    runner = FakeCommandRunner(fixture["clock"], failing_stage="export")
    worker = make_worker(fixture, runner)

    result = worker.run()

    assert runner.stages == ["curriculum", "trainer", "export"]
    assert result["status"] == "failed"
    assert result["failure_reason"] == "child_command_failed"
    assert result["candidate_model"] is None
    assert result["evidence"] == []
    assert worker._trainer_checkpoint_proof["state"] == "stable-file"


@pytest.mark.parametrize("binding", ["champion", "katago"])
def test_hash_drift_fails_closed_before_any_command(tmp_path, binding):
    fixture = make_fixture(tmp_path)
    runner = FakeCommandRunner(fixture["clock"])
    worker = make_worker(fixture, runner)
    if binding == "champion":
        fixture["champion"].write_bytes(b"changed champion")
    else:
        fixture["katago"].write_bytes(b"#!/bin/sh\nexit 7\n")

    with pytest.raises(FrozenInputChanged):
        worker.run()

    assert runner.stages == []
    assert not fixture["result_path"].exists()


def test_cli_requires_all_hash_budget_and_result_bindings(tmp_path):
    fixture = make_fixture(tmp_path)
    args = parse_args(
        [
            "--worker-spec",
            str(fixture["spec_path"]),
            "--worker-spec-sha256",
            fixture["spec"].spec_sha256,
            "--trial-manifest-path",
            str(fixture["manifest_path"]),
            "--trial-manifest-sha256",
            fixture["manifest"]["manifest_sha256"],
            "--work-id",
            "adaptive-work-a",
            "--round-index",
            "0",
            "--gpu-time-reservation-seconds",
            "14400",
            "--deadline-unix",
            str(fixture["deadline"]),
            "--result-path",
            str(fixture["result_path"]),
        ]
    )

    assert args.expected_spec_sha256 == fixture["spec"].spec_sha256
    assert args.trial_manifest_sha256 == fixture["manifest"]["manifest_sha256"]
    assert args.reservation_gpu_seconds == 14_400
    assert args.result == fixture["result_path"]


def test_worker_spec_is_canonical_self_hashed_and_source_clean(tmp_path):
    fixture = make_fixture(tmp_path)
    raw = json.loads(fixture["spec_path"].read_text(encoding="utf-8"))

    assert raw["contract"] == WORKER_SPEC_CONTRACT
    body = dict(raw)
    supplied = body.pop("spec_sha256")
    assert supplied == canonical_sha256(body)
    assert fixture["spec_path"].read_bytes() == canonical_json_bytes(raw) + b"\n"

    with pytest.raises(WorkerSpecError) as dirty:
        load_worker_spec(
            fixture["spec_path"],
            expected_spec_sha256=supplied,
            revision_reader=fixture["revision_reader"],
            repository_status_reader=lambda _path: "?? unexpected\n",
        )
    assert dirty.value.code == "source_checkout_dirty"


def test_noncanonical_or_rehashed_spec_drift_is_rejected(tmp_path):
    fixture = make_fixture(tmp_path)
    raw = copy.deepcopy(dict(fixture["spec"].raw))
    changed = (tmp_path / "changed-worker.json").resolve()
    changed.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    with pytest.raises(WorkerSpecError, match="cannot load"):
        load_worker_spec(
            changed,
            expected_spec_sha256=raw["spec_sha256"],
            revision_reader=fixture["revision_reader"],
            repository_status_reader=fixture["status_reader"],
        )

    raw["gpu_id"] = "6"
    raw.pop("spec_sha256")
    raw["spec_sha256"] = canonical_sha256(raw)
    write_spec(changed, raw)
    with pytest.raises(WorkerSpecError, match="GPU ID 7"):
        load_worker_spec(
            changed,
            expected_spec_sha256=raw["spec_sha256"],
            revision_reader=fixture["revision_reader"],
            repository_status_reader=fixture["status_reader"],
        )
