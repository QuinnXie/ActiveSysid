"""State and output trajectory prediction."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint


@partial(jax.jit, static_argnames=("state_fcn", "output_fcn"))
def _predict_fixed_discrete(
        x0, U, valid_length, state_fcn, output_fcn, params):
    """Run a deterministic discrete model without changing the input shape."""
    x0 = jnp.asarray(x0).reshape(-1)
    indices = jnp.arange(U.shape[0])

    def model_step(current_x, inputs):
        index, u = inputs

        def valid_step(x):
            y = jnp.asarray(output_fcn(x, u, params)).reshape(-1)
            next_x = jnp.asarray(state_fcn(x, u, params)).reshape(x.shape)
            return next_x, (y, x)

        def padded_step(x):
            y = jnp.zeros_like(
                jnp.asarray(output_fcn(x, u, params)).reshape(-1)
            )
            return x, (y, x)

        return jax.lax.cond(
            index < valid_length, valid_step, padded_step, current_x
        )

    _, (Y, X) = jax.lax.scan(model_step, x0, (indices, U))
    return Y, X


def predict_fixed(
        x0, U, valid_length, state_fcn, output_fcn, params,
        return_X=False):
    """Predict into fixed-size buffers while using only a valid prefix.

    Unlike :func:`predict`, the returned arrays retain ``U.shape[0]`` rows.
    Rows at and after ``valid_length`` are padding.  Keeping the outer shape
    fixed lets repeated online/checkpoint evaluations reuse one JAX
    compilation; callers should slice the valid prefix after transferring the
    result to NumPy.

    This helper is for learned discrete-time models.  Physical continuous-time
    simulation remains available through :func:`predict`.
    """
    U = jnp.asarray(U)
    if U.ndim != 2:
        raise ValueError("U must have shape (sample_count, input_dimension)")
    if U.shape[0] == 0:
        raise ValueError("U must contain at least one sample")
    valid_length = int(valid_length)
    if not 0 < valid_length <= U.shape[0]:
        raise ValueError("valid_length must be between 1 and U.shape[0]")

    Y, X = _predict_fixed_discrete(
        x0, U, jnp.asarray(valid_length), state_fcn, output_fcn, params
    )
    if return_X:
        return Y, X
    return Y


def predict(
        x0, U, state_fcn, output_fcn, params, qx=0.0, qy=0.0,
        return_X=False, temporality="discrete", Ts=1.0,
        noise_mode="additive", key=None):
    """Predict an output signal and its state trajectory.

    The returned arrays use the convention

    ``X[k] = x_k``, ``Y[k] = h(x_k, u_k)``,
    and ``x_{k+1} = f(x_k, u_k)``.

    ``qx`` and ``qy`` are interpreted as noise scales: additive noise uses
    them as standard deviations, while multiplicative noise uses them as
    relative standard-deviation factors. Pass a JAX PRNG key to ``key`` to
    control the random realization. If omitted, a fixed key is used for
    backward reproducibility.
    """
    if temporality not in ("discrete", "continuous"):
        raise ValueError(
            "temporality must be either 'discrete' or 'continuous'"
        )
    if noise_mode not in ("additive", "multiplicative"):
        raise ValueError(
            "noise_mode must be either 'additive' or 'multiplicative'"
        )

    U = jnp.asarray(U)
    if U.ndim != 2:
        raise ValueError("U must have shape (sample_count, input_dimension)")
    if U.shape[0] == 0:
        raise ValueError("U must contain at least one sample")

    x = jnp.asarray(x0).reshape(-1)
    nx = x.shape[0]
    sample_count = U.shape[0]
    initial_output = jnp.asarray(output_fcn(x, U[0], params)).reshape(-1)
    ny = initial_output.shape[0]

    if key is None:
        key = jax.random.PRNGKey(0)
    process_keys, measurement_keys = jax.random.split(key, 2)
    process_keys = jax.random.split(process_keys, sample_count)
    measurement_keys = jax.random.split(measurement_keys, sample_count)

    def add_noise(value, scale, random_key):
        noise = scale * jax.random.normal(random_key, value.shape)
        if noise_mode == "additive":
            return value + noise
        return value * (1 + noise)

    if temporality == "continuous":
        integration_grid = jnp.asarray([0.0, Ts])

        def model_step_ode(current_x, inputs):
            u, process_key, measurement_key = inputs
            y = jnp.asarray(
                output_fcn(current_x, u, params)
            ).reshape(-1)
            y = add_noise(y, qy, measurement_key)
            next_x = odeint(
                state_fcn, current_x, integration_grid, u
            )[-1]
            next_x = add_noise(next_x.reshape(nx), qx, process_key)
            return next_x, (y, current_x)

        _, (Y, X) = jax.lax.scan(
            model_step_ode, x, (U, process_keys, measurement_keys)
        )

    elif temporality == "discrete":
        def model_step(current_x, inputs):
            u, process_key, measurement_key = inputs
            y = jnp.asarray(
                output_fcn(current_x, u, params)
            ).reshape(ny)
            y = add_noise(y, qy, measurement_key)
            next_x = jnp.asarray(
                state_fcn(current_x, u, params)
            ).reshape(nx)
            next_x = add_noise(next_x, qx, process_key)
            return next_x, (y, current_x)

        _, (Y, X) = jax.lax.scan(
            model_step, x, (U, process_keys, measurement_keys)
        )

    if return_X:
        return Y, X
    return Y
