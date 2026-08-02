"""Augmented-state extended Kalman filter for states and model parameters."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import cho_factor, cho_solve

from .utils import hstack_arrays, params_to_abcd


def _covariance_matrix(covariance, dimension, name):
    """Return a full covariance matrix from scalar, diagonal, or matrix input."""
    covariance = jnp.asarray(covariance, dtype=jnp.float64)
    if covariance.ndim == 0:
        return covariance * jnp.eye(dimension)
    if covariance.ndim == 1:
        if covariance.shape[0] != dimension:
            raise ValueError(
                f"{name} diagonal must have length {dimension}, "
                f"got {covariance.shape[0]}"
            )
        return jnp.diag(covariance)
    if covariance.ndim == 2:
        if covariance.shape != (dimension, dimension):
            raise ValueError(
                f"{name} covariance must have shape "
                f"{(dimension, dimension)}, got {covariance.shape}"
            )
        return covariance
    raise ValueError(f"{name} must be a scalar, diagonal vector, or matrix")


@partial(jax.jit, static_argnums=(0,))
def _output_jacobian_x(output_fcn, x, u, params):
    """Jacobian of the output with respect to the state."""
    return jax.jacrev(output_fcn)(x, u=u, params=params)


@partial(jax.jit, static_argnums=(0,))
def _output_jacobian_params(output_fcn, x, u, params):
    """Jacobian pytree of the output with respect to model parameters."""
    return jax.jacrev(lambda p: output_fcn(x, u, p))(params)


@partial(jax.jit, static_argnums=(0,))
def _state_jacobian_x(state_fcn, x, u, params):
    """Jacobian of the state update with respect to the state."""
    return jax.jacrev(state_fcn)(x, u=u, params=params)


@partial(jax.jit, static_argnums=(0,))
def _state_jacobian_params(state_fcn, x, u, params):
    """Jacobian pytree of the state update with respect to model parameters."""
    return jax.jacrev(lambda p: state_fcn(x, u, p))(params)


def EKF_measurement_update(
        x_10, u, y, output_fcn, p_10, P=None,
        rho_x=1e-3, rho_th=1e-3, Qy_cov=1, isLinear=False,
        return_diagnostics=False, nis_gate=None):
    """Assimilate ``y(k)`` and return the posterior state, parameters, and P.

    ``Qy_cov`` is a measurement-noise covariance in the coordinates of ``y``:
    pass a scalar variance, a diagonal variance vector, or a full covariance
    matrix.
    """
    x_10 = jnp.asarray(x_10, dtype=jnp.float64)
    nx = x_10.shape[0]
    ny = jnp.shape(y)[0]

    th_10, unravel_th = ravel_pytree(p_10)
    nth = th_10.shape[0]
    R = _covariance_matrix(Qy_cov, ny, "Qy_cov")

    if P is None:
        P = jnp.block([
            [rho_x * jnp.eye(nx), jnp.zeros((nx, nth))],
            [jnp.zeros((nth, nx)), rho_th * jnp.eye(nth)],
        ])
    else:
        P = jnp.asarray(P)

    if isLinear:
        _, _, Cx, _ = params_to_abcd(p_10)
    else:
        Cx = _output_jacobian_x(output_fcn, x_10, u, p_10)

    Cth = hstack_arrays(
        _output_jacobian_params(output_fcn, x_10, u, p_10)
    )
    x_th_10 = jnp.concatenate([x_10, th_10], axis=0)
    y_10 = output_fcn(x_10, u, p_10)
    C = jnp.hstack([Cx, Cth])

    PC = P @ C.T
    innovation_cov = R + C @ PC
    innovation_cov = 0.5 * (innovation_cov + innovation_cov.T)
    innovation_cov += 1e-9 * jnp.eye(ny)
    factor, lower = cho_factor(innovation_cov)
    gain = cho_solve((factor, lower), PC.T).T

    innovation = y - y_10
    nis = innovation @ cho_solve((factor, lower), innovation)
    if nis_gate is None:
        innovation_scale = jnp.asarray(1.0, dtype=innovation.dtype)
    else:
        soft_threshold, hard_threshold = nis_gate
        safe_nis = jnp.maximum(nis, jnp.finfo(innovation.dtype).tiny)
        innovation_scale = jnp.where(
            nis <= soft_threshold,
            1.0,
            jnp.sqrt(soft_threshold / safe_nis),
        )
        innovation_scale = jnp.where(
            nis > hard_threshold, 0.0, innovation_scale
        )
    robust_gain = innovation_scale * gain
    x_th_11 = x_th_10 + robust_gain @ innovation

    identity_minus_gain_jacobian = jnp.eye(nx + nth) - robust_gain @ C
    P_11 = (
        identity_minus_gain_jacobian
        @ P
        @ identity_minus_gain_jacobian.T
        + robust_gain @ R @ robust_gain.T
    )
    P_11 = 0.5 * (P_11 + P_11.T)

    p_11 = unravel_th(x_th_11[nx:])
    if return_diagnostics:
        diagnostics = {
            "innovation": innovation,
            "innovation_covariance": innovation_cov,
            "nis": nis,
            "innovation_scale": innovation_scale,
            "parameter_update_norm": jnp.linalg.norm(x_th_11[nx:] - th_10),
            "posterior_parameter_covariance_trace": jnp.trace(P_11[nx:, nx:]),
            "posterior_covariance_min_eig": jnp.min(jnp.linalg.eigvalsh(P_11)),
        }
        return p_11, x_th_11[:nx], P_11, diagnostics
    return p_11, x_th_11[:nx], P_11


def EKF_time_update(
        x_11, u, state_fcn, p_11, P_11,
        Qx_cov=1e-10, Qth_cov=1e-10, isLinear=False):
    """Propagate the posterior at ``k`` to the prior at ``k+1``.

    ``Qx_cov`` and ``Qth_cov`` are process covariance terms in the coordinates
    of the EKF state and flattened parameters.  Each accepts a scalar variance,
    a diagonal variance vector, or a full covariance matrix.
    """
    x_11 = jnp.asarray(x_11, dtype=jnp.float64)
    P_11 = jnp.asarray(P_11)
    nx = x_11.shape[0]
    th_11, _ = ravel_pytree(p_11)
    nth = th_11.shape[0]

    Qx = _covariance_matrix(Qx_cov, nx, "Qx_cov")
    Qth = _covariance_matrix(Qth_cov, nth, "Qth_cov")
    x_21 = state_fcn(x_11, u, p_11)

    if isLinear:
        Ax, _, _, _ = params_to_abcd(p_11)
    else:
        Ax = _state_jacobian_x(state_fcn, x_11, u, p_11)
    Ath = hstack_arrays(
        _state_jacobian_params(state_fcn, x_11, u, p_11)
    )

    A = jnp.eye(nx + nth)
    A = A.at[:nx, :nx].set(Ax)
    A = A.at[:nx, nx:].set(Ath)

    process_cov = jnp.block([
        [Qx, jnp.zeros((nx, nth))],
        [jnp.zeros((nth, nx)), Qth],
    ])
    P_21 = A @ P_11 @ A.T + process_cov
    P_21 = 0.5 * (P_21 + P_21.T)
    P_21 += 1e-9 * jnp.eye(nx + nth)
    return x_21, P_21


def EKF_step(
        x_10, u, y, state_fcn, output_fcn, p_10,
        P=None, rho_x=1e-3, rho_th=1e-3, Qx_cov=1e-10,
        Qy_cov=1, Qth_cov=1e-10, isLinear=False):
    """Assimilate ``y(k)`` and then propagate with ``u(k)``."""
    p_11, x_11, P_11 = EKF_measurement_update(
        x_10, u, y, output_fcn, p_10, P=P,
        rho_x=rho_x, rho_th=rho_th, Qy_cov=Qy_cov,
        isLinear=isLinear,
    )
    x_21, P_21 = EKF_time_update(
        x_11, u, state_fcn, p_11, P_11,
        Qx_cov=Qx_cov, Qth_cov=Qth_cov, isLinear=isLinear,
    )
    return p_11, x_21, P_21


EKF_measurement_update = jax.jit(
    EKF_measurement_update,
    static_argnames=("output_fcn", "isLinear", "return_diagnostics", "nis_gate"),
)
EKF_time_update = jax.jit(
    EKF_time_update,
    static_argnames=("state_fcn", "isLinear"),
)
EKF_step = jax.jit(
    EKF_step,
    static_argnames=("state_fcn", "output_fcn", "isLinear"),
)


def EKF_set(
        x0, U, Y, state_fcn, output_fcn, params,
        P=None, rho_x=1e-4, rho_th=1e-4,
        Qx_cov=1e-8, Qy_cov=1, Qth_cov=1e-8,
        isLinear=False, N_epoch=1, verbosity=True):
    """Replay a data set through the augmented-state EKF.

    Each epoch replays the same trajectory from ``x0`` while carrying the
    current parameter estimate and covariance into the next epoch.  The
    returned state trajectory from the final epoch follows
    ``X[0] = x0`` and ``X[k + 1]`` equal to the prior state after assimilating
    ``Y[k]`` and propagating with ``U[k]``.
    """
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    params = jax.tree_util.tree_map(
        lambda array: jnp.asarray(array, dtype=jnp.float64), params
    )
    nx = x0.shape[0]
    sample_count = U.shape[0]
    if sample_count == 0:
        raise ValueError("U and Y must contain at least one sample")
    if Y.shape[0] != sample_count:
        raise ValueError("U and Y must contain the same number of samples")
    if not isinstance(N_epoch, int) or N_epoch < 1:
        raise ValueError("N_epoch must be a positive integer")

    theta, _ = ravel_pytree(params)
    nth = theta.shape[0]

    if P is None:
        P = jnp.block([
            [
                1 / rho_x / sample_count * jnp.eye(nx),
                jnp.zeros((nx, nth)),
            ],
            [
                jnp.zeros((nth, nx)),
                1 / rho_th / sample_count * jnp.eye(nth),
            ],
        ])

    @jax.jit
    def ekf_step_fn(carry, inputs):
        current_params, current_x, current_P = carry
        u, y = inputs
        current_params, current_x, current_P = EKF_step(
            current_x,
            u,
            y,
            state_fcn,
            output_fcn,
            current_params,
            current_P,
            Qx_cov=Qx_cov,
            Qy_cov=Qy_cov,
            Qth_cov=Qth_cov,
            isLinear=isLinear,
        )
        return (current_params, current_x, current_P), current_x

    for epoch in range(N_epoch):
        if verbosity:
            print(f"Epoch {epoch + 1}/{N_epoch}")
        (params, x, P), X = jax.lax.scan(
            ekf_step_fn, (params, x0, P), (U, Y)
        )

    return params, x, P, jnp.vstack((x0, X))
