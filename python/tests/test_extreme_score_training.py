import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from katago.train.data_processing_pytorch import (
    pad_global_targets_nc,
    validate_extreme_score_training_rows,
)
from katago.train.extreme_score_policy import (
    load_extreme_score_training_policy,
    validate_extreme_score_curriculum_transition,
)
from katago.train.metrics_pytorch import Metrics
from katago.train.model_pytorch import EXTRA_SCORE_DISTR_RADIUS
from katago.train.objective_mode import (
    reset_swa_for_objective_migration,
    resolve_objective_mode,
)


class _PolicyHead:
    num_policy_outputs = 8


class _ValueHead:
    def __init__(self, scorebelief_len: int):
        midpoint = scorebelief_len // 2
        self.score_belief_offset_vector = (
            torch.arange(scorebelief_len, dtype=torch.float32) - midpoint + 0.5
        )


class _RawModel:
    def __init__(self, pos_len: int):
        self.pos_len = pos_len
        self.scoremean_multiplier = 20.0
        self.policy_head = _PolicyHead()
        scorebelief_len = 2 * (pos_len * pos_len + EXTRA_SCORE_DISTR_RADIUS)
        self.value_head = _ValueHead(scorebelief_len)
        self.config = {"version": 17, "predict_q_values": True}
        self.training = True

    def get_has_intermediate_head(self):
        return False


def _valid_extreme_global_targets(num_rows: int = 2) -> np.ndarray:
    assert num_rows == 2
    targets = np.zeros((num_rows, 80), dtype=np.float32)
    targets[:, 25] = np.array([0.5, 0.25], dtype=np.float32)
    targets[:, 26] = 1.0
    targets[:, 27] = 1.0
    targets[:, 28] = 0.0
    targets[:, 29] = 1.0
    targets[:, 33] = 1.0
    targets[:, 34] = 1.0
    targets[:, 68] = 1.0
    targets[:, 69] = 2.0
    targets[:, 70] = np.array([0.0, 1.0], dtype=np.float32)
    targets[:, 71] = 1.0
    targets[:, 72] = np.array([10.0, 12.0], dtype=np.float32)
    targets[:, 73] = np.array([5.0, 10.0], dtype=np.float32)
    targets[:, 74] = np.array([5.0, 2.0], dtype=np.float32)
    targets[:, 75] = np.array([1.0, 1.0], dtype=np.float32)
    targets[:, 76] = 1.0
    targets[:, 77] = targets[:, 74]
    targets[:, 78] = np.array([1234.0, 1235.0], dtype=np.float32)
    return targets


def test_extreme_score_row_contract_is_opt_in_and_fail_closed():
    targets = _valid_extreme_global_targets()
    original = targets.copy()
    validate_extreme_score_training_rows(targets, expected_group_size=2)
    np.testing.assert_array_equal(targets, original)
    with pytest.raises(ValueError, match="frozen curriculum"):
        validate_extreme_score_training_rows(targets, expected_group_size=1)

    malformed = targets.copy()
    malformed[0, 68:80] = 0.0
    with pytest.raises(ValueError, match="without supported"):
        validate_extreme_score_training_rows(malformed)

    for bad_weight in (0.0, -1.0, np.nan):
        malformed = targets.copy()
        malformed[0, 25] = bad_weight
        with pytest.raises(ValueError, match="cohort weight"):
            validate_extreme_score_training_rows(malformed)

    malformed = targets.copy()
    malformed[0, 26] = 0.0
    with pytest.raises(ValueError, match="focal policy weight"):
        validate_extreme_score_training_rows(malformed)

    malformed = targets.copy()
    malformed[0, 28] = 1.0
    with pytest.raises(ValueError, match="opponent policy weight"):
        validate_extreme_score_training_rows(malformed)

    malformed = targets.copy()
    malformed[0, 79] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_extreme_score_training_rows(malformed)

    malformed = targets.copy()
    malformed[0, 74] = 4.0
    with pytest.raises(ValueError, match="leave-one-out"):
        validate_extreme_score_training_rows(malformed)

    for channel in (27, 29, 33, 34):
        malformed = targets.copy()
        malformed[0, channel] = np.nan
        with pytest.raises(ValueError, match=f"C{channel}"):
            validate_extreme_score_training_rows(malformed)

    optional_lead = targets.copy()
    optional_lead[:, 29] = 0.0
    validate_extreme_score_training_rows(optional_lead)
    for bad_lead_weight in (-0.1, 1.1):
        malformed = targets.copy()
        malformed[0, 29] = bad_lead_weight
        with pytest.raises(ValueError, match="C29"):
            validate_extreme_score_training_rows(malformed)


def test_extreme_score_row_contract_validates_consumed_target_arrays():
    targets = _valid_extreme_global_targets()
    policies = np.ones((2, 2, 5), dtype=np.float32)
    score_distr = np.zeros((2, 8), dtype=np.float32)
    score_distr[:, 3] = 100.0
    value_targets = np.zeros((2, 5, 2, 2), dtype=np.float32)
    qvalues = np.zeros((2, 3, 5), dtype=np.float32)

    validate_extreme_score_training_rows(
        targets,
        policyTargetsNCMove=policies,
        scoreDistrN=score_distr,
        valueTargetsNCHW=value_targets,
        qValueTargetsNCMove=qvalues,
    )

    bad_policies = policies.copy()
    bad_policies[0, 0, :] = 0.0
    with pytest.raises(ValueError, match="policyTargetsNCMove"):
        validate_extreme_score_training_rows(targets, policyTargetsNCMove=bad_policies)

    bad_score_distr = score_distr.copy()
    bad_score_distr[0, 3] = np.nan
    with pytest.raises(ValueError, match="scoreDistrN"):
        validate_extreme_score_training_rows(targets, scoreDistrN=bad_score_distr)

    bad_value_targets = value_targets.copy()
    bad_value_targets[0, 4, 0, 0] = 121.0
    with pytest.raises(ValueError, match="valueTargetsNCHW"):
        validate_extreme_score_training_rows(
            targets, valueTargetsNCHW=bad_value_targets
        )

    bad_qvalues = qvalues.copy()
    bad_qvalues[0, 2, 0] = -1.0
    with pytest.raises(ValueError, match="qValueTargetsNCMove"):
        validate_extreme_score_training_rows(targets, qValueTargetsNCMove=bad_qvalues)


def test_singleton_expected_score_cohort_uses_unit_credit_for_any_margin():
    for margin in (-25.0, 0.0, 40.0):
        targets = np.zeros((1, 80), dtype=np.float32)
        targets[:, 25] = 1.0
        targets[:, 26] = 1.0
        targets[:, 27] = 1.0
        targets[:, 29] = 1.0
        targets[:, 33] = 1.0
        targets[:, 34] = 1.0
        targets[:, 68] = 1.0
        targets[:, 69] = 1.0
        targets[:, 70] = 0.0
        targets[:, 71] = -1.0
        targets[:, 72] = margin
        targets[:, 73] = 0.0
        targets[:, 74] = 1.0
        targets[:, 75] = 1.0
        targets[:, 76] = 1.0
        targets[:, 77] = 1.0
        validate_extreme_score_training_rows(targets)


def test_legacy_rows_remain_loadable_by_default_but_not_score_only():
    legacy = np.zeros((3, 64), dtype=np.float32)
    padded = pad_global_targets_nc(legacy)
    assert padded.shape == (3, 80)
    np.testing.assert_array_equal(padded[:, :64], legacy)
    np.testing.assert_array_equal(padded[:, 64:], 0.0)
    with pytest.raises(ValueError, match="without supported"):
        validate_extreme_score_training_rows(padded)


def _leaf(*shape, offset=0.0):
    return (torch.randn(*shape, dtype=torch.float32) + offset).requires_grad_()


def _positive_leaf(*shape):
    return (torch.rand(*shape, dtype=torch.float32) + 0.5).requires_grad_()


def _make_metrics_case():
    torch.manual_seed(1234)
    n = 2
    pos_len = 2
    pos_area = pos_len * pos_len
    policy_len = pos_area + 1
    scorebelief_len = 2 * (pos_area + EXTRA_SCORE_DISTR_RADIUS)

    outputs_by_name = {
        "policy": _leaf(n, 8, policy_len),
        "value": _leaf(n, 3),
        "td_value": _leaf(n, 3, 3),
        "td_score": _leaf(n, 3),
        "ownership": _leaf(n, 1, pos_len, pos_len),
        "scoring": _leaf(n, 1, pos_len, pos_len),
        "futurepos": _leaf(n, 2, pos_len, pos_len),
        "seki": _leaf(n, 4, pos_len, pos_len),
        "scoremean": _leaf(n),
        "scorestdev": _positive_leaf(n),
        "lead": _leaf(n),
        "variance_time": _leaf(n, offset=1.0),
        "shortterm_value_error": _positive_leaf(n),
        "shortterm_score_error": _positive_leaf(n),
        "scorebelief": _leaf(n, scorebelief_len),
    }
    outputs = (
        outputs_by_name["policy"],
        outputs_by_name["value"],
        outputs_by_name["td_value"],
        outputs_by_name["td_score"],
        outputs_by_name["ownership"],
        outputs_by_name["scoring"],
        outputs_by_name["futurepos"],
        outputs_by_name["seki"],
        outputs_by_name["scoremean"],
        outputs_by_name["scorestdev"],
        outputs_by_name["lead"],
        outputs_by_name["variance_time"],
        outputs_by_name["shortterm_value_error"],
        outputs_by_name["shortterm_score_error"],
        outputs_by_name["scorebelief"],
    )

    global_targets = torch.zeros(n, 80)
    global_targets[:, 0:3] = torch.tensor([0.65, 0.25, 0.10])
    for start in (4, 8, 12):
        global_targets[:, start : start + 3] = torch.tensor([0.55, 0.35, 0.10])
    global_targets[:, 3] = torch.tensor([7.5, -4.0])
    global_targets[:, 7] = torch.tensor([3.0, -2.0])
    global_targets[:, 11] = torch.tensor([5.0, -3.0])
    global_targets[:, 15] = torch.tensor([8.0, -5.0])
    global_targets[:, 21] = torch.tensor([6.0, -3.5])
    global_targets[:, 22] = 2.0
    global_targets[:, 25] = torch.tensor([0.75, 1.25])
    global_targets[:, 26] = 1.0
    global_targets[:, 27] = 1.0
    # Deliberately nonzero: the metrics layer must defensively zero it for an
    # explicitly marked focal cohort even if a caller bypasses loader validation.
    global_targets[:, 28] = 1.0
    global_targets[:, 29] = 1.0
    global_targets[:, 33] = 1.0
    global_targets[:, 34] = 1.0
    global_targets[:, 68] = 1.0
    global_targets[:, 69] = 1.0
    global_targets[:, 71] = 1.0
    global_targets[:, 72] = torch.tensor([3.0, 4.0])
    global_targets[:, 74] = global_targets[:, 72]
    global_targets[:, 75] = 1.0
    global_targets[:, 76] = 1.0
    global_targets[:, 77] = global_targets[:, 74]
    global_targets[:, 78] = torch.tensor([101.0, 102.0])

    policy_targets = torch.arange(
        1,
        n * 2 * policy_len + 1,
        dtype=torch.float32,
    ).reshape(n, 2, policy_len)
    score_distribution = torch.zeros(n, scorebelief_len)
    score_distribution[0, scorebelief_len // 2 + 7] = 100.0
    score_distribution[1, scorebelief_len // 2 - 4] = 100.0
    value_targets = torch.zeros(n, 5, pos_len, pos_len)
    value_targets[:, 0] = torch.tensor(
        [[0.75, -0.25], [0.25, -0.75]], dtype=torch.float32
    )
    value_targets[:, 1] = 0.25
    value_targets[:, 2] = 0.5
    value_targets[:, 3] = -0.5
    value_targets[:, 4] = 24.0

    qvalue_targets = torch.zeros(n, 3, policy_len)
    qvalue_targets[:, 0] = torch.tensor([[12000.0, -8000.0, 5000.0, -3000.0, 1000.0]])
    qvalue_targets[:, 1] = torch.tensor([[540.0, 300.0, -420.0, 180.0, -120.0]])
    qvalue_targets[:, 2] = torch.tensor([[16.0, 9.0, 4.0, 1.0, 25.0]])

    batch = {
        "binaryInputNCHW": torch.ones(n, 1, pos_len, pos_len),
        "globalInputNC": torch.zeros(n, 1),
        "policyTargetsNCMove": policy_targets,
        "globalTargetsNC": global_targets,
        "scoreDistrN": score_distribution,
        "valueTargetsNCHW": value_targets,
        "qValueTargetsNCMove": qvalue_targets,
    }
    raw_model = _RawModel(pos_len)
    metrics_obj = Metrics(1, raw_model)
    return raw_model, metrics_obj, outputs, outputs_by_name, batch


def _compute_metrics(raw_model, metrics_obj, outputs, batch, extreme_score_only):
    return metrics_obj.metrics_dict_batchwise(
        raw_model,
        [outputs],
        extra_outputs=None,
        batch=batch,
        is_training=False,
        soft_policy_weight_scale=8.0,
        disable_optimistic_policy=False,
        meta_kata_only_soft_policy=False,
        value_loss_scale=0.6,
        td_value_loss_scales=[0.6, 0.6, 0.6],
        seki_loss_scale=1.0,
        variance_time_loss_scale=1.0,
        main_loss_scale=1.0,
        intermediate_loss_scale=None,
        include_model_norms=False,
        extreme_score_only=extreme_score_only,
    )


def test_score_only_metrics_zero_forbidden_losses_and_backprop_allowed_losses():
    raw_model, metrics_obj, outputs, output_tensors, batch = _make_metrics_case()
    metrics = _compute_metrics(
        raw_model,
        metrics_obj,
        outputs,
        batch,
        extreme_score_only=True,
    )

    forbidden_metric_names = (
        "p1loss_sum",
        "p1softloss_sum",
        "p0lopt_sum",
        "p0loptw_sum",
        "p0sopt_sum",
        "p0soptw_sum",
        "vloss_sum",
        "tdvloss1_sum",
        "tdvloss2_sum",
        "tdvloss3_sum",
        "vtimeloss_sum",
        "evstloss_sum",
        "qwlloss_sum",
        "xsforbid_sum",
    )
    for name in forbidden_metric_names:
        assert metrics[name].item() == 0.0, name

    for name in (
        "p0loss_sum",
        "p0softloss_sum",
        "tdsloss_sum",
        "smloss_sum",
        "sbcdfloss_sum",
        "sbpdfloss_sum",
        "qscloss_sum",
        "xspolicy_sum",
        "xsscore_sum",
    ):
        assert metrics[name].item() > 0.0, name
    assert metrics["xsmode_batch"].item() == 1.0

    metrics["loss_sum"].backward()
    policy_grad = output_tensors["policy"].grad
    assert torch.count_nonzero(policy_grad[:, 0]).item() > 0
    assert torch.count_nonzero(policy_grad[:, 1]).item() == 0
    assert torch.count_nonzero(policy_grad[:, 2]).item() > 0
    assert torch.count_nonzero(policy_grad[:, 3]).item() == 0
    assert torch.count_nonzero(policy_grad[:, 4]).item() == 0
    assert torch.count_nonzero(policy_grad[:, 5]).item() == 0
    assert torch.count_nonzero(policy_grad[:, 6]).item() == 0
    assert torch.count_nonzero(policy_grad[:, 7]).item() > 0

    assert torch.count_nonzero(output_tensors["value"].grad).item() == 0
    assert torch.count_nonzero(output_tensors["td_value"].grad).item() == 0
    for allowed in (
        "td_score",
        "ownership",
        "scoring",
        "futurepos",
        "seki",
        "scoremean",
        "scorestdev",
        "lead",
        "shortterm_score_error",
        "scorebelief",
    ):
        assert torch.count_nonzero(output_tensors[allowed].grad).item() > 0, allowed
    assert torch.count_nonzero(output_tensors["variance_time"].grad).item() == 0
    assert torch.count_nonzero(output_tensors["shortterm_value_error"].grad).item() == 0


def test_default_metrics_keep_legacy_loss_behavior_and_checkpoint_shape():
    raw_model, metrics_obj, outputs, _, batch = _make_metrics_case()
    metrics = _compute_metrics(
        raw_model,
        metrics_obj,
        outputs,
        batch,
        extreme_score_only=False,
    )
    for name in (
        "p1loss_sum",
        "p1softloss_sum",
        "p0lopt_sum",
        "p0sopt_sum",
        "vloss_sum",
        "tdvloss1_sum",
        "tdvloss2_sum",
        "tdvloss3_sum",
        "qwlloss_sum",
    ):
        assert metrics[name].item() > 0.0, name
    assert "xsmode_batch" not in metrics

    state = metrics_obj.state_dict()
    assert set(state) == {
        "moving_unowned_proportion_sum",
        "moving_unowned_proportion_weight",
    }
    restored = Metrics(1, raw_model)
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_checkpoint_objective_mode_inherits_and_migrations_fail_closed():
    assert resolve_objective_mode(True, None, False) == (True, False)
    assert resolve_objective_mode(False, None, False) == (False, False)
    assert resolve_objective_mode(None, None, False) == (False, False)

    with pytest.raises(ValueError, match="allow-objective-mode-migration"):
        resolve_objective_mode(True, False, False)
    with pytest.raises(ValueError, match="allow-objective-mode-migration"):
        resolve_objective_mode(None, True, False)

    assert resolve_objective_mode(True, False, True) == (False, True)
    assert resolve_objective_mode(None, True, True) == (True, True)


def test_objective_migration_discards_prior_swa_average():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    prior = torch.optim.swa_utils.AveragedModel(model)
    prior.update_parameters(model)
    with torch.no_grad():
        model.weight.fill_(3.0)
    prior.update_parameters(model)
    assert prior.module.weight.item() == pytest.approx(2.0)

    reset = reset_swa_for_objective_migration(model, prior, 8.0)
    assert reset.n_averaged.item() == 0
    assert reset.module.weight.item() == pytest.approx(3.0)


def test_frozen_training_policy_is_hash_bound_and_caps_cohort_size(tmp_path):
    python_dir = Path(__file__).resolve().parents[1]
    policy_path = python_dir / "risk_score" / "extreme_score_training_policy_v1.json"
    binding = load_extreme_score_training_policy(policy_path, 8)
    assert binding["cohort_size"] == 8
    assert binding["maximum_supported_cohort_size"] == 8
    assert len(binding["file_sha256"]) == 64
    first_stage = load_extreme_score_training_policy(policy_path, 1)
    assert first_stage["minimum_selected_training_samples"] == 0

    with pytest.raises(ValueError, match="absent from"):
        load_extreme_score_training_policy(policy_path, 16)

    tampered = tmp_path / "tampered-policy.json"
    text = policy_path.read_text(encoding="utf-8").replace(
        '"win_loss_weight": 0', '"win_loss_weight": 1'
    )
    tampered.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="weakens"):
        load_extreme_score_training_policy(tampered, 1)


def test_curriculum_transition_requires_next_stage_and_checkpointed_samples():
    policy_path = (
        Path(__file__).resolve().parents[1]
        / "risk_score"
        / "extreme_score_training_policy_v1.json"
    )
    stage_two = load_extreme_score_training_policy(policy_path, 2)
    with pytest.raises(ValueError, match="without"):
        validate_extreme_score_curriculum_transition(
            previous_cohort_size=1,
            requested_policy=stage_two,
            selected_training_samples=2_000_000,
            allow_transition=False,
        )
    with pytest.raises(ValueError, match="premature"):
        validate_extreme_score_curriculum_transition(
            previous_cohort_size=1,
            requested_policy=stage_two,
            selected_training_samples=1_999_999,
            allow_transition=True,
        )
    assert validate_extreme_score_curriculum_transition(
        previous_cohort_size=1,
        requested_policy=stage_two,
        selected_training_samples=2_000_000,
        allow_transition=True,
    )
    stage_four = load_extreme_score_training_policy(policy_path, 4)
    with pytest.raises(ValueError, match="exactly one"):
        validate_extreme_score_curriculum_transition(
            previous_cohort_size=1,
            requested_policy=stage_four,
            selected_training_samples=4_000_000,
            allow_transition=True,
        )


def test_train_cli_exposes_extreme_score_only_mode():
    python_dir = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(python_dir / "train.py"), "--help"],
        cwd=python_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "-extreme-score-only" in completed.stdout
    assert "-standard-objective" in completed.stdout
    assert "-allow-objective-mode-migration" in completed.stdout
    assert "-extreme-score-training-policy" in completed.stdout
    assert "-extreme-score-cohort-size" in completed.stdout
    assert "-allow-extreme-score-curriculum-transition" in completed.stdout
