"""Input-output greedy sampling acquisition."""

import jax
import jax.numpy as jnp

from ._input_output import (
    distance_option,
    one_step_penalty,
    predicted_outputs,
    valid_component_scale,
)


def _GSuy_fixed_jax(
    U,
    Y,
    m,
    u_set,
    x,
    fx,
    fy,
    params,
    const=None,
    input_weight=1.0,
    output_weight=1.0,
    distance_eps=1e-8,
):
    """Input-output greedy sampling with normalized fixed-size histories."""
    valid = jnp.arange(U.shape[0]) < m
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

    input_distances = input_weight * jnp.sum(
        ((U[:, None, :] - u_set[None, :, :]) / u_scale) ** 2,
        axis=2,
    )
    output_distances = output_weight * jnp.sum(
        ((Y[1:, None, :] - y1[None, :, :]) / y_scale) ** 2,
        axis=2,
    )
    du = jnp.min(
        jnp.where(valid[:, None], input_distances, jnp.inf), axis=0
    )
    dy = jnp.min(
        jnp.where(valid[:, None], output_distances, jnp.inf), axis=0
    )
    values = du * dy

    if const is not None:
        values = jax.lax.cond(
            const["flag"],
            lambda scores: scores - one_step_penalty(y1, const),
            lambda scores: scores,
            values,
        )
    return u_set[jnp.argmax(values)]


GSuy_fixed_jax = jax.jit(_GSuy_fixed_jax, static_argnums=(5, 6))
