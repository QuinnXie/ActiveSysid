"""Input-output inverse-distance-weighting acquisition."""

import jax
import jax.numpy as jnp

from ._input_output import (
    distance_option,
    one_step_penalty,
    predicted_outputs,
    valid_component_scale,
)


def _IDWuy_fixed_jax(
    U,
    Y,
    m,
    u_set,
    x,
    fx,
    fy,
    params,
    delta=1.0,
    const=None,
    input_weight=1.0,
    output_weight=1.0,
    distance_eps=1e-8,
):
    """Select an input using IDW in input space and output uncertainty."""
    valid = jnp.arange(U.shape[0]) < m
    eps = 1e-12
    input_weight = distance_option(
        const, "input_distance_weight", input_weight
    )
    output_weight = distance_option(
        const, "output_distance_weight", output_weight
    )
    distance_eps = distance_option(const, "distance_eps", distance_eps)
    u_scale = valid_component_scale(U, valid, distance_eps)
    y_scale = valid_component_scale(Y[1:], valid, distance_eps)
    y1 = predicted_outputs(x, u_set, fx, fy, params)

    distances = input_weight * jnp.sum(
        ((U[:, None, :] - u_set[None, :, :]) / u_scale) ** 2,
        axis=2,
    )
    close = valid[:, None] & (distances < eps)
    safe_distances = jnp.where(valid[:, None], distances, 1.0)
    weights = jnp.where(
        valid[:, None],
        jnp.exp(-safe_distances) / safe_distances,
        0.0,
    )
    sum_weights = jnp.sum(weights, axis=0)
    square_y = output_weight * jnp.sum(
        ((Y[1:, None, :] - y1[None, :, :]) / y_scale) ** 2,
        axis=2,
    )
    uncertainty = jnp.sum(
        (weights / jnp.maximum(sum_weights[None, :], eps)) * square_y,
        axis=0,
    )
    representativeness = (
        jnp.arctan(1.0 / jnp.maximum(sum_weights, eps)) * 2.0 / jnp.pi
    )

    close_values = output_weight * jnp.sum(
        ((Y[jnp.argmax(close, axis=0) + 1] - y1) / y_scale) ** 2,
        axis=1,
    )
    values = jnp.where(
        jnp.any(close, axis=0),
        close_values,
        uncertainty + delta * representativeness,
    )

    if const is not None:
        values = jax.lax.cond(
            const["flag"],
            lambda scores: scores - one_step_penalty(y1, const),
            lambda scores: scores,
            values,
        )
    return u_set[jnp.argmax(values)]


IDWuy_fixed_jax = jax.jit(_IDWuy_fixed_jax, static_argnums=(5, 6))
