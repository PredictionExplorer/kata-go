from torch.optim.swa_utils import AveragedModel


def reset_swa_for_objective_migration(raw_model, swa_model, swa_scale):
    """Discard averages from the prior objective while preserving EMA policy."""
    if swa_model is None:
        return None
    new_factor = 1.0 / swa_scale

    def ema_avg(avg_param, cur_param, _num_averaged):
        return avg_param + new_factor * (cur_param - avg_param)

    return AveragedModel(raw_model, avg_fn=ema_avg)


def resolve_objective_mode(
    previous_extreme_score_only: bool | None,
    requested_extreme_score_only: bool | None,
    allow_migration: bool,
) -> tuple[bool, bool]:
    """Resolve checkpoint objective mode and whether optimizer state must reset."""
    if requested_extreme_score_only is None:
        extreme_score_only = (
            bool(previous_extreme_score_only)
            if previous_extreme_score_only is not None
            else False
        )
    else:
        extreme_score_only = bool(requested_extreme_score_only)

    changed = (
        previous_extreme_score_only is not None
        and bool(previous_extreme_score_only) != extreme_score_only
    ) or (previous_extreme_score_only is None and extreme_score_only)
    if changed and not allow_migration:
        previous_name = (
            "extreme-score-only" if bool(previous_extreme_score_only) else "standard"
        )
        requested_name = "extreme-score-only" if extreme_score_only else "standard"
        raise ValueError(
            "Refusing objective-mode migration from "
            f"{previous_name} to {requested_name}; pass "
            "-allow-objective-mode-migration to authorize the transition "
            "and reset optimizer state"
        )
    return extreme_score_only, changed
