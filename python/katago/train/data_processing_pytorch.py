import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
import torch.nn.functional

from ..train import modelconfigs

# Needs to be kept in sync with GLOBAL_TARGET_NUM_CHANNELS in trainingwrite.cpp C++ code among other places.
# Data format version 2 files (recorded in channel 63 of each row) had only 64 channels; they are zero-padded
# up to this width when loading, which correctly encodes "not reanalyzed" for the version 3 channels.
GLOBAL_TARGETS_NC_CHANNELS = 80
EXTREME_SCORE_COHORT_METADATA_START = 68
EXTREME_SCORE_COHORT_METADATA_END = 80
EXTREME_SCORE_COHORT_METADATA_VERSION = 1
EXTREME_SCORE_METADATA_VERSION_CHANNEL = 68
EXTREME_SCORE_GROUP_SIZE_CHANNEL = 69
EXTREME_SCORE_ATTEMPT_INDEX_CHANNEL = 70
EXTREME_SCORE_FOCAL_COLOR_CHANNEL = 71
EXTREME_SCORE_FOCAL_MARGIN_CHANNEL = 72
EXTREME_SCORE_LEAVE_ONE_OUT_MAX_CHANNEL = 73
EXTREME_SCORE_CREDIT_CHANNEL = 74
EXTREME_SCORE_RANK_CHANNEL = 75
EXTREME_SCORE_SELECTED_CHANNEL = 76
EXTREME_SCORE_APPLIED_WEIGHT_CHANNEL = 77
EXTREME_SCORE_COHORT_ID_LOW_CHANNEL = 78
EXTREME_SCORE_COHORT_ID_HIGH_CHANNEL = 79
GLOBAL_WEIGHT_CHANNEL = 25
PLAYER_POLICY_WEIGHT_CHANNEL = 26
OPPONENT_POLICY_WEIGHT_CHANNEL = 28

def pad_global_targets_nc(globalTargetsNC: np.ndarray) -> np.ndarray:
    """Zero-pad older-format globalTargetsNC rows up to the current channel count."""
    num_channels = globalTargetsNC.shape[1]
    if num_channels == GLOBAL_TARGETS_NC_CHANNELS:
        return globalTargetsNC
    assert num_channels < GLOBAL_TARGETS_NC_CHANNELS, f"globalTargetsNC has {num_channels} channels, more than the expected {GLOBAL_TARGETS_NC_CHANNELS}"
    padded = np.zeros((globalTargetsNC.shape[0], GLOBAL_TARGETS_NC_CHANNELS), dtype=globalTargetsNC.dtype)
    padded[:, :num_channels] = globalTargetsNC
    return padded

def validate_extreme_score_training_rows(
    globalTargetsNC: np.ndarray,
    source: str = "globalTargetsNC",
    policyTargetsNCMove: np.ndarray | None = None,
    scoreDistrN: np.ndarray | None = None,
    valueTargetsNCHW: np.ndarray | None = None,
    qValueTargetsNCMove: np.ndarray | None = None,
    expected_group_size: int | None = None,
) -> None:
    """Fail closed unless every row is an explicit, consistently weighted cohort row.

    Channels 68-79 carry the writer's v1 cohort assignment and exact
    leave-one-out credit. C25 remains the authoritative aggregate loss weight;
    C77 records the unscaled cohort credit for auditability.
    """
    if globalTargetsNC.ndim != 2:
        raise ValueError(f"{source} must be rank 2, got shape {globalTargetsNC.shape}")
    if globalTargetsNC.shape[1] < EXTREME_SCORE_COHORT_METADATA_END:
        raise ValueError(
            f"{source} has {globalTargetsNC.shape[1]} channels; "
            f"extreme-score training requires channels "
            f"{EXTREME_SCORE_COHORT_METADATA_START}-"
            f"{EXTREME_SCORE_COHORT_METADATA_END - 1}"
        )

    cohort_metadata = globalTargetsNC[
        :,
        EXTREME_SCORE_COHORT_METADATA_START:EXTREME_SCORE_COHORT_METADATA_END,
    ]
    if not np.all(np.isfinite(cohort_metadata)):
        raise ValueError(f"{source} has non-finite extreme-score cohort metadata")

    metadata_versions = globalTargetsNC[:, EXTREME_SCORE_METADATA_VERSION_CHANNEL]
    if not np.all(metadata_versions == EXTREME_SCORE_COHORT_METADATA_VERSION):
        bad_rows = np.flatnonzero(
            metadata_versions != EXTREME_SCORE_COHORT_METADATA_VERSION
        )
        raise ValueError(
            f"{source} contains {bad_rows.size} row(s) without supported "
            f"extreme-score cohort metadata v"
            f"{EXTREME_SCORE_COHORT_METADATA_VERSION}; first bad row "
            f"{int(bad_rows[0])}"
        )

    group_sizes = globalTargetsNC[:, EXTREME_SCORE_GROUP_SIZE_CHANNEL]
    attempt_indices = globalTargetsNC[:, EXTREME_SCORE_ATTEMPT_INDEX_CHANNEL]
    ranks = globalTargetsNC[:, EXTREME_SCORE_RANK_CHANNEL]
    selected = globalTargetsNC[:, EXTREME_SCORE_SELECTED_CHANNEL]
    cohort_id_low = globalTargetsNC[:, EXTREME_SCORE_COHORT_ID_LOW_CHANNEL]
    cohort_id_high = globalTargetsNC[:, EXTREME_SCORE_COHORT_ID_HIGH_CHANNEL]
    integer_fields_valid = (
        (group_sizes == np.floor(group_sizes))
        & (attempt_indices == np.floor(attempt_indices))
        & (ranks == np.floor(ranks))
        & (selected == np.floor(selected))
        & (cohort_id_low == np.floor(cohort_id_low))
        & (cohort_id_high == np.floor(cohort_id_high))
    )
    assignment_fields_valid = (
        integer_fields_valid
        & (group_sizes >= 1.0)
        & (attempt_indices >= 0.0)
        & (attempt_indices < group_sizes)
        & (ranks >= 1.0)
        & (ranks <= group_sizes)
        & ((selected == 0.0) | (selected == 1.0))
        & (cohort_id_low >= 0.0)
        & (cohort_id_low <= float(0x3FFFFF))
        & (cohort_id_high >= 0.0)
        & (cohort_id_high <= float(0x3FFFFF))
    )
    focal_colors = globalTargetsNC[:, EXTREME_SCORE_FOCAL_COLOR_CHANNEL]
    assignment_fields_valid &= (focal_colors == -1.0) | (focal_colors == 1.0)
    if not np.all(assignment_fields_valid):
        bad_rows = np.flatnonzero(~assignment_fields_valid)
        raise ValueError(
            f"{source} contains invalid extreme-score cohort assignment "
            f"fields; first bad row {int(bad_rows[0])}"
        )
    if expected_group_size is not None:
        if type(expected_group_size) is not int or expected_group_size <= 0:
            raise ValueError("expected extreme-score group size must be positive")
        if not np.all(group_sizes == float(expected_group_size)):
            bad_rows = np.flatnonzero(
                group_sizes != float(expected_group_size)
            )
            raise ValueError(
                f"{source} cohort size differs from the frozen curriculum "
                f"N={expected_group_size}; first bad row {int(bad_rows[0])}"
            )

    focal_margins = globalTargetsNC[:, EXTREME_SCORE_FOCAL_MARGIN_CHANNEL]
    leave_one_out_max = globalTargetsNC[
        :, EXTREME_SCORE_LEAVE_ONE_OUT_MAX_CHANNEL
    ]
    credits = globalTargetsNC[:, EXTREME_SCORE_CREDIT_CHANNEL]
    applied_weights = globalTargetsNC[:, EXTREME_SCORE_APPLIED_WEIGHT_CHANNEL]
    expected_credits = np.where(
        group_sizes == 1.0,
        1.0,
        np.maximum(0.0, focal_margins - leave_one_out_max),
    )
    credit_valid = (
        (credits >= 0.0)
        & np.isclose(credits, expected_credits, rtol=1e-5, atol=1e-5)
        & np.isclose(applied_weights, credits, rtol=1e-6, atol=1e-6)
        & (selected == (credits > 0.0).astype(selected.dtype))
        & ((group_sizes != 1.0) | (leave_one_out_max == 0.0))
    )
    if not np.all(credit_valid):
        bad_rows = np.flatnonzero(~credit_valid)
        raise ValueError(
            f"{source} contains inconsistent leave-one-out cohort credit "
            f"metadata; first bad row {int(bad_rows[0])}"
        )

    cohort_weights = globalTargetsNC[:, GLOBAL_WEIGHT_CHANNEL]
    valid_cohort_weights = (
        np.isfinite(cohort_weights)
        & (cohort_weights >= 0.0)
        & ((selected == 0.0) | (cohort_weights > 0.0))
        & ((selected == 1.0) | (cohort_weights == 0.0))
    )
    if not np.all(valid_cohort_weights):
        bad_rows = np.flatnonzero(~valid_cohort_weights)
        raise ValueError(
            f"{source} contains {bad_rows.size} invalid extreme-score cohort "
            f"weight(s) in C{GLOBAL_WEIGHT_CHANNEL}; first bad row "
            f"{int(bad_rows[0])}"
        )
    if not np.all(selected == 1.0):
        raise ValueError(
            f"{source} contains unselected cohort rows; the writer must omit "
            "zero-credit games before score-only training"
        )

    player_policy_weights = globalTargetsNC[:, PLAYER_POLICY_WEIGHT_CHANNEL]
    if not np.all(np.isfinite(player_policy_weights) & (player_policy_weights > 0.0)):
        raise ValueError(
            f"{source} contains an extreme-score row without positive focal "
            f"policy weight in C{PLAYER_POLICY_WEIGHT_CHANNEL}"
        )

    opponent_policy_weights = globalTargetsNC[:, OPPONENT_POLICY_WEIGHT_CHANNEL]
    if not np.all(np.isfinite(opponent_policy_weights) & (opponent_policy_weights == 0.0)):
        raise ValueError(
            f"{source} contains an extreme-score row with nonzero opponent "
            f"policy weight in C{OPPONENT_POLICY_WEIGHT_CHANNEL}"
        )

    required_weight_channels = {
        27: "ownership/score-distribution",
        33: "future-position",
        34: "scoring",
    }
    for channel, name in required_weight_channels.items():
        weights = globalTargetsNC[:, channel]
        if not np.all(
            np.isfinite(weights) & (weights > 0.0) & (weights <= 1.0)
        ):
            raise ValueError(
                f"{source} contains an invalid {name} weight in C{channel}; "
                "extreme-score rows require a finite value in (0,1]"
            )
    lead_weights = globalTargetsNC[:, 29]
    if not np.all(
        np.isfinite(lead_weights)
        & (lead_weights >= 0.0)
        & (lead_weights <= 1.0)
    ):
        raise ValueError(
            f"{source} contains an invalid optional lead weight in C29; "
            "expected a finite value in [0,1]"
        )

    # Metrics computes every head before zeroing forbidden contributions, so
    # non-finite dormant targets can still poison loss_sum through NaN * 0.
    consumed_global_channels = list(range(23)) + list(range(24, 30)) + [
        33,
        34,
        35,
    ]
    if not np.all(np.isfinite(globalTargetsNC[:, consumed_global_channels])):
        raise ValueError(f"{source} has non-finite consumed global target values")

    def _check_row_count(name: str, array: np.ndarray) -> None:
        if array.shape[0] != globalTargetsNC.shape[0]:
            raise ValueError(
                f"{source}:{name} has {array.shape[0]} rows, expected "
                f"{globalTargetsNC.shape[0]}"
            )

    if policyTargetsNCMove is not None:
        _check_row_count("policyTargetsNCMove", policyTargetsNCMove)
        if policyTargetsNCMove.ndim != 3 or policyTargetsNCMove.shape[1] < 2:
            raise ValueError(
                f"{source}:policyTargetsNCMove must have shape (N,>=2,M)"
            )
        policies = np.asarray(policyTargetsNCMove[:, :2, :])
        valid_policies = (
            np.all(np.isfinite(policies), axis=(1, 2))
            & np.all(policies >= 0.0, axis=(1, 2))
            & np.all(np.sum(policies, axis=2) > 0.0, axis=1)
        )
        if not np.all(valid_policies):
            raise ValueError(
                f"{source}:policyTargetsNCMove contains a non-finite, negative, "
                "or empty policy target"
            )

    if scoreDistrN is not None:
        _check_row_count("scoreDistrN", scoreDistrN)
        if scoreDistrN.ndim != 2:
            raise ValueError(f"{source}:scoreDistrN must have shape (N,S)")
        score_distributions = np.asarray(scoreDistrN)
        valid_score_distributions = (
            np.all(np.isfinite(score_distributions), axis=1)
            & np.all(score_distributions >= 0.0, axis=1)
            & np.isclose(
                np.sum(score_distributions, axis=1),
                100.0,
                rtol=0.0,
                atol=1e-4,
            )
        )
        if not np.all(valid_score_distributions):
            raise ValueError(
                f"{source}:scoreDistrN contains an invalid score distribution"
            )

    if valueTargetsNCHW is not None:
        _check_row_count("valueTargetsNCHW", valueTargetsNCHW)
        if valueTargetsNCHW.ndim != 4 or valueTargetsNCHW.shape[1] < 5:
            raise ValueError(
                f"{source}:valueTargetsNCHW must have shape (N,>=5,H,W)"
            )
        value_targets = np.asarray(valueTargetsNCHW[:, :5, :, :])
        valid_value_targets = (
            np.all(np.isfinite(value_targets), axis=(1, 2, 3))
            & np.all(
                (value_targets[:, :4] >= -1.0)
                & (value_targets[:, :4] <= 1.0),
                axis=(1, 2, 3),
            )
            & np.all(
                (value_targets[:, 4] >= -120.0)
                & (value_targets[:, 4] <= 120.0),
                axis=(1, 2),
            )
        )
        if not np.all(valid_value_targets):
            raise ValueError(
                f"{source}:valueTargetsNCHW contains non-finite or out-of-range "
                "ownership/scoring targets"
            )

    if qValueTargetsNCMove is not None:
        _check_row_count("qValueTargetsNCMove", qValueTargetsNCMove)
        if qValueTargetsNCMove.ndim != 3 or qValueTargetsNCMove.shape[1] < 3:
            raise ValueError(
                f"{source}:qValueTargetsNCMove must have shape (N,>=3,M)"
            )
        qvalue_targets = np.asarray(qValueTargetsNCMove[:, :3, :])
        if not np.all(np.isfinite(qvalue_targets)) or not np.all(
            qvalue_targets[:, 2, :] >= 0.0
        ):
            raise ValueError(
                f"{source}:qValueTargetsNCMove contains non-finite targets or "
                "negative visit weights"
            )

def read_npz_training_data(
    npz_files,
    batch_size: int,
    world_size: int,
    rank: int,
    pos_len: int,
    device,
    randomize_symmetries: bool,
    include_meta: bool,
    model_config: modelconfigs.ModelConfig,
    prefetch_depth: int = 1,
    extreme_score_only: bool = False,
    extreme_score_cohort_size: int | None = None,
):
    if extreme_score_only and extreme_score_cohort_size is None:
        raise ValueError(
            "score-only data loading requires a frozen cohort size"
        )
    if not extreme_score_only and extreme_score_cohort_size is not None:
        raise ValueError(
            "extreme-score cohort size requires score-only data loading"
        )
    rand = np.random.default_rng(seed=list(os.urandom(12)))
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    (h_base,h_builder) = build_history_matrices(model_config, device)

    # Version 16 always predicts q values; version 17+ does so only when configured.
    include_qvalues = model_config["version"] == 16 or (
        model_config["version"] >= 17 and bool(model_config.get("predict_q_values"))
    )

    def load_npz_file(npz_file):
        # Select only THIS rank's rows up front, while the arrays are still in
        # their compact on-disk dtypes (packed bits / int8 / int16), so the
        # expensive unpackbits + float32 expansion runs on 1/world_size of the
        # data rather than the whole shard in every rank.
        with np.load(npz_file) as npz:
            num_samples = npz["globalInputNC"].shape[0]
            num_whole_steps = num_samples // (batch_size * world_size)
            used = num_whole_steps * world_size * batch_size

            def select_rank_rows(arr):
                # Keep only the rows this rank will consume:
                # reshape the used prefix to (steps, world_size, batch, ...) and
                # take this rank's slice. For world_size>1 the trailing reshape
                # forces a compact 1/world_size-size copy and lets the full
                # decompressed array be freed immediately.
                # Drop any trailing suffix that doesn't match the overall world batch size.
                arr = arr[:used]
                rest = arr.shape[1:]
                arr = arr.reshape(num_whole_steps, world_size, batch_size, *rest)
                arr = arr[:, rank]
                return arr.reshape(num_whole_steps * batch_size, *rest)

            binaryInputNCHWPacked = select_rank_rows(npz["binaryInputNCHWPacked"])
            globalInputNC = select_rank_rows(npz["globalInputNC"])
            policyTargetsNCMove = select_rank_rows(npz["policyTargetsNCMove"]).astype(np.float32)
            if extreme_score_only:
                # Validate the whole globally used prefix before rank slicing so
                # every DDP rank fails on the same malformed shard.
                globalTargetsNCAllRanks = pad_global_targets_nc(
                    npz["globalTargetsNC"][:used]
                )
                validate_extreme_score_training_rows(
                    globalTargetsNCAllRanks,
                    source=f"{npz_file}:globalTargetsNC",
                    policyTargetsNCMove=npz["policyTargetsNCMove"][:used],
                    scoreDistrN=npz["scoreDistrN"][:used],
                    valueTargetsNCHW=npz["valueTargetsNCHW"][:used],
                    qValueTargetsNCMove=(
                        npz["qValueTargetsNCMove"][:used]
                        if include_qvalues
                        else None
                    ),
                    expected_group_size=extreme_score_cohort_size,
                )
                globalTargetsNC = select_rank_rows(globalTargetsNCAllRanks)
                del globalTargetsNCAllRanks
            else:
                globalTargetsNC = pad_global_targets_nc(
                    select_rank_rows(npz["globalTargetsNC"])
                )
            scoreDistrN = select_rank_rows(npz["scoreDistrN"]).astype(np.float32)
            valueTargetsNCHW = select_rank_rows(npz["valueTargetsNCHW"]).astype(np.float32)
            if include_meta:
                metadataInputNC = select_rank_rows(npz["metadataInputNC"]).astype(np.float32)
            else:
                metadataInputNC = None
            if include_qvalues:
                qValueTargetsNCMove = select_rank_rows(npz["qValueTargetsNCMove"]).astype(np.float32)
            else:
                qValueTargetsNCMove = None
        del npz

        binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked,axis=2)
        assert len(binaryInputNCHW.shape) == 3
        assert binaryInputNCHW.shape[2] == ((pos_len * pos_len + 7) // 8) * 8
        binaryInputNCHW = binaryInputNCHW[:,:,:pos_len*pos_len]
        binaryInputNCHW = np.reshape(binaryInputNCHW, (
            binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], pos_len, pos_len
        )).astype(np.float32)

        assert binaryInputNCHW.shape[1] == num_bin_features
        assert globalInputNC.shape[1] == num_global_features
        return (npz_file, binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW, metadataInputNC, qValueTargetsNCMove)

    if not npz_files:
        return

    # Prefetch up to prefetch_depth files *ahead* of the one currently being
    # consumed, so the GPU does not stall at a file boundary waiting on disk +
    # decompress + unpackbits for the next shard.
    # Each in-flight file holds its full expanded (float32) arrays in RAM,
    # so memory scales linearly with prefetch_depth
    # (times world_size, since every rank loads each file).
    prefetch_depth = max(1, prefetch_depth)
    with ThreadPoolExecutor(max_workers=prefetch_depth) as executor:
        # Keep a queue of in-flight loads: the head is the file being consumed,
        # and up to prefetch_depth more are loading/loaded behind it.
        pending = deque()
        next_index = 0
        while next_index < len(npz_files) and len(pending) <= prefetch_depth:
            pending.append(executor.submit(load_npz_file, npz_files[next_index]))
            next_index += 1

        while pending:
            future = pending.popleft()
            (npz_file, binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW, metadataInputNC, qValueTargetsNCMove) = future.result()

            # The arrays already hold only this rank's rows (selected in load_npz_file),
            # so the first dim is num_whole_steps * batch_size.
            num_whole_steps = binaryInputNCHW.shape[0] // batch_size

            logging.info(f"Beginning {npz_file} with {num_whole_steps * world_size} usable batches, my rank is {rank}")

            # Top the pipeline back up so prefetch_depth files stay in flight.
            if next_index < len(npz_files):
                logging.info(f"Preloading {npz_files[next_index]} while processing this file")
                pending.append(executor.submit(load_npz_file, npz_files[next_index]))
                next_index += 1

            for n in range(num_whole_steps):
                start = n * batch_size
                end = start + batch_size

                batch_binaryInputNCHW = torch.from_numpy(binaryInputNCHW[start:end]).to(device)
                batch_globalInputNC = torch.from_numpy(globalInputNC[start:end]).to(device)
                batch_policyTargetsNCMove = torch.from_numpy(policyTargetsNCMove[start:end]).to(device)
                batch_globalTargetsNC = torch.from_numpy(globalTargetsNC[start:end]).to(device)
                batch_scoreDistrN = torch.from_numpy(scoreDistrN[start:end]).to(device)
                batch_valueTargetsNCHW = torch.from_numpy(valueTargetsNCHW[start:end]).to(device)
                if include_meta:
                    batch_metadataInputNC = torch.from_numpy(metadataInputNC[start:end]).to(device)
                if include_qvalues:
                    batch_qValueTargetsNCMove = torch.from_numpy(qValueTargetsNCMove[start:end]).to(device)

                (batch_binaryInputNCHW, batch_globalInputNC) = apply_history_matrices(
                    model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder
                )

                if randomize_symmetries:
                    symm = int(rand.integers(0,8))
                    batch_binaryInputNCHW = apply_symmetry(batch_binaryInputNCHW, symm)
                    batch_policyTargetsNCMove = apply_symmetry_policy(batch_policyTargetsNCMove, symm, pos_len)
                    batch_valueTargetsNCHW = apply_symmetry(batch_valueTargetsNCHW, symm)
                    if include_qvalues:
                        batch_qValueTargetsNCMove = apply_symmetry_policy(batch_qValueTargetsNCMove, symm, pos_len)

                batch_binaryInputNCHW = batch_binaryInputNCHW.contiguous()
                batch_policyTargetsNCMove = batch_policyTargetsNCMove.contiguous()
                batch_valueTargetsNCHW = batch_valueTargetsNCHW.contiguous()
                if include_qvalues:
                    batch_qValueTargetsNCMove = batch_qValueTargetsNCMove.contiguous()

                batch = dict(
                    binaryInputNCHW = batch_binaryInputNCHW,
                    globalInputNC = batch_globalInputNC,
                    policyTargetsNCMove = batch_policyTargetsNCMove,
                    globalTargetsNC = batch_globalTargetsNC,
                    scoreDistrN = batch_scoreDistrN,
                    valueTargetsNCHW = batch_valueTargetsNCHW,
                )
                if include_meta:
                    batch["metadataInputNC"] = batch_metadataInputNC
                if include_qvalues:
                    batch["qValueTargetsNCMove"] = batch_qValueTargetsNCMove

                yield batch


def apply_symmetry_policy(tensor, symm, pos_len):
    """Same as apply_symmetry but also handles the pass index"""
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    tensor_without_pass = tensor[:,:,:-1].view((batch_size, channels, pos_len, pos_len))
    tensor_transformed = apply_symmetry(tensor_without_pass, symm)
    return torch.cat((
        tensor_transformed.reshape(batch_size, channels, pos_len*pos_len),
        tensor[:,:,-1:]
    ), dim=2)

def apply_symmetry(tensor, symm):
    """
    Apply a symmetry operation to the given tensor.

    Args:
        tensor (torch.Tensor): Tensor to be rotated. (..., W, W)
        symm (int):
            0, 1, 2, 3: Rotation by symm * pi / 2 radians.
            4, 5, 6, 7: Mirror symmetry on top of rotation.
    """
    assert tensor.shape[-1] == tensor.shape[-2]

    if symm == 0:
        return tensor
    if symm == 1:
        return tensor.transpose(-2, -1).flip(-2)
    if symm == 2:
        return tensor.flip(-1).flip(-2)
    if symm == 3:
        return tensor.transpose(-2, -1).flip(-1)
    if symm == 4:
        return tensor.transpose(-2, -1)
    if symm == 5:
        return tensor.flip(-1)
    if symm == 6:
        return tensor.transpose(-2, -1).flip(-1).flip(-2)
    if symm == 7:
        return tensor.flip(-2)


def build_history_matrices(model_config: modelconfigs.ModelConfig, device):
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    assert num_bin_features == 22, "Currently this code is hardcoded for this many features"

    h_base = torch.diag(
        torch.tensor(
            [
                1.0,  # 0
                1.0,  # 1
                1.0,  # 2
                1.0,  # 3
                1.0,  # 4
                1.0,  # 5
                1.0,  # 6
                1.0,  # 7
                1.0,  # 8
                0.0,  # 9   Location of move 1 turn ago
                0.0,  # 10  Location of move 2 turns ago
                0.0,  # 11  Location of move 3 turns ago
                0.0,  # 12  Location of move 4 turns ago
                0.0,  # 13  Location of move 5 turns ago
                1.0,  # 14  Ladder-threatened stone
                0.0,  # 15  Ladder-threatened stone, 1 turn ago
                0.0,  # 16  Ladder-threatened stone, 2 turns ago
                1.0,  # 17
                1.0,  # 18
                1.0,  # 19
                1.0,  # 20
                1.0,  # 21
            ],
            device=device,
            requires_grad=False,
        )
    )
    # Because we have ladder features that express past states rather than past diffs,
    # the most natural encoding when we have no history is that they were always the
    # same, rather than that they were all zero. So rather than zeroing them we have no
    # history, we add entries in the matrix to copy them over.
    # By default, without history, the ladder features 15 and 16 just copy over from 14.
    h_base[14, 15] = 1.0
    h_base[14, 16] = 1.0

    h0 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    # When have the prev move, we enable feature 9 and 15
    h0[9, 9] = 1.0  # Enable 9 -> 9
    h0[14, 15] = -1.0  # Stop copying 14 -> 15
    h0[14, 16] = -1.0  # Stop copying 14 -> 16
    h0[15, 15] = 1.0  # Enable 15 -> 15
    h0[15, 16] = 1.0  # Start copying 15 -> 16

    h1 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    # When have the prevprev move, we enable feature 10 and 16
    h1[10, 10] = 1.0  # Enable 10 -> 10
    h1[15, 16] = -1.0  # Stop copying 15 -> 16
    h1[16, 16] = 1.0  # Enable 16 -> 16

    h2 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h2[11, 11] = 1.0

    h3 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h3[12, 12] = 1.0

    h4 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h4[13, 13] = 1.0

    # (1, n_bin, n_bin)
    h_base = h_base.reshape((1, num_bin_features, num_bin_features))
    # (5, n_bin, n_bin)
    h_builder = torch.stack((h0, h1, h2, h3, h4), dim=0)

    return (h_base, h_builder)


def apply_history_matrices(model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder):
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    # include_history = batch_globalTargetsNC[:,36:41]
    should_stop_history = torch.rand_like(batch_globalTargetsNC[:,36:41]) >= 0.98
    include_history = (torch.cumsum(should_stop_history,axis=1,dtype=torch.float32) <= 0.1).to(torch.float32)

    # include_history: (N, 5)
    # bi * ijk -> bjk, (N, 5) * (5, n_bin, n_bin) -> (N, n_bin, n_bin)
    h_matrix = h_base + torch.einsum("bi,ijk->bjk", include_history, h_builder)


    # batch_binaryInputNCHW: (N, n_bin_in, 19, 19)
    # h_matrix: (N, n_bin_in, n_bin_out)
    # Result: (N, n_bin_out, 19, 19)
    batch_binaryInputNCHW = torch.einsum("bijk,bil->bljk", batch_binaryInputNCHW, h_matrix)

    # First 5 global input features exactly correspond to include_history, pointwise multiply to
    # enable/disable them
    batch_globalInputNC = batch_globalInputNC * torch.nn.functional.pad(
        include_history, ((0, num_global_features - include_history.shape[1])), value=1.0
    )
    return batch_binaryInputNCHW, batch_globalInputNC
