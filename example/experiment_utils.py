"""Experiment-setting selection helpers."""

import numpy as np


def _is_cmp_al_mode(exp_type):
    """Return True for cmp_al variants (for example cmp_al5), but not cmp_alpha."""
    exp_type = exp_type.lower()
    return exp_type == "cmp_al" or (
        exp_type.startswith("cmp_al") and exp_type != "cmp_alpha"
    )


def score_checkpoints(score_array, n_train_init):
    """Return score checkpoints with at least one finite value."""
    finite_mask = np.isfinite(score_array[..., 0])
    reduce_axes = tuple(range(finite_mask.ndim - 1))
    has_score = (
        np.any(finite_mask, axis=reduce_axes)
        if reduce_axes
        else finite_mask
    )
    checkpoint_indices = np.flatnonzero(has_score)
    return checkpoint_indices[checkpoint_indices >= n_train_init - 1]


def update_params(
    n_set,
    exp_type,
    AL_set,
    delta_set=(1.0,),
    alpha_set=(1.0,),
    const_set=None,
):
    """Select the active-learning method and tuning values for one setting."""
    exp_type = exp_type.lower()
    n_al_set = len(AL_set)
    delta = delta_set[0]
    alpha = alpha_set[0]

    if _is_cmp_al_mode(exp_type):
        al_method = AL_set[n_set]
    elif exp_type == "cmp_ekf":
        al_method = AL_set[0]
    elif exp_type == "cmp_delta":
        if n_set >= n_al_set - 1:
            al_method = "idw"
            delta = delta_set[n_set - n_al_set + 1]
        else:
            al_method = AL_set[n_set]
    elif exp_type == "cmp_delta_idwuy":
        if n_set >= n_al_set - 1:
            al_method = "IDWuy"
            delta = delta_set[n_set - n_al_set + 1]
        else:
            al_method = AL_set[n_set]
    elif exp_type == "cmp_alpha":
        if n_set >= n_al_set - 1:
            al_method = "idw"
            alpha = alpha_set[n_set - n_al_set + 1]
        else:
            al_method = AL_set[n_set]
    elif exp_type == "cmp_idw_grid":
        if n_set >= n_al_set - 1:
            al_method = "idw"
            grid_index = n_set - n_al_set + 1
            delta = delta_set[grid_index // len(alpha_set)]
            alpha = alpha_set[grid_index % len(alpha_set)]
        else:
            al_method = AL_set[n_set]
    elif exp_type == "const_y":
        if const_set is None or len(const_set) == 0:
            raise ValueError("const_set is required for exp_type='const_y'")
        al_method = AL_set[0]
        const_set[n_set % len(const_set)]
    elif exp_type == "cmp_const":
        if const_set is None or len(const_set) == 0:
            raise ValueError("const_set is required for exp_type='cmp_const'")
        al_method = AL_set[n_set]
        const_set[0]
    else:
        raise ValueError(f"Wrong experiment type: {exp_type!r}")

    return al_method, delta, alpha
