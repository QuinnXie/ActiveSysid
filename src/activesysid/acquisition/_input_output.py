"""Shared helpers for input-output acquisition methods."""

import jax
import jax.numpy as jnp


def valid_component_scale(values, valid, eps):
    valid_count = jnp.maximum(jnp.sum(valid), 1)
    masked_values = jnp.where(valid[:, None], values, 0.0)
    mean = jnp.sum(masked_values, axis=0) / valid_count
    variance = (
        jnp.sum(
            jnp.where(valid[:, None], (values - mean) ** 2, 0.0),
            axis=0,
        )
        / valid_count
    )
    return jnp.sqrt(jnp.maximum(variance, eps**2))


def distance_option(const, name, default):
    if const is None:
        return default
    return const.get(name, default)


def predicted_outputs(x, u_set, fx, fy, params):
    x1 = jax.vmap(lambda u: fx(x, u, params))(u_set)
    return jax.vmap(lambda x_next, u: fy(x_next, u, params))(x1, u_set)


def one_step_penalty(y1, const):
    p_min = jnp.sum(jnp.maximum(-y1 + const["y_min"], 0), axis=1)
    p_max = jnp.sum(jnp.maximum(y1 - const["y_max"], 0), axis=1)
    return const.get("penalty_rho", 1e12) * (p_min + p_max)
