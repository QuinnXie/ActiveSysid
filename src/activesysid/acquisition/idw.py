import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import timeit


def _valid_component_scale(values, valid, eps):
    valid_count = jnp.maximum(jnp.sum(valid), 1)
    masked_values = jnp.where(valid[:, None], values, 0.0)
    mean = jnp.sum(masked_values, axis=0) / valid_count
    variance = jnp.sum(jnp.where(valid[:, None], (values - mean) ** 2, 0.0), axis=0) / valid_count
    return jnp.sqrt(jnp.maximum(variance, eps ** 2))


def _distance_option(const, name, default):
    if const is None:
        return default
    return const.get(name, default)


# Original NumPy version
def idw(U, Xhat, Y, u_set, fx, fy, params, delta = 1, alpha = 1, const = None):
    aa = np.array([acqidw_numpy(u, U, Xhat, Y, fx, fy, params, delta, alpha, const) for u in u_set])
    u = u_set[np.argmax(aa)]
    return u

def acqidw_numpy(u, U, Xhat, Y, fx, fy, params, delta = 1, alpha = 1, const=None):
    x = Xhat[-1]
    N = U.shape[0]
    x1 = fx(x, u, params)
    y1 = fy(x1, u, params)

    q = np.hstack([x, u])
    Q = np.hstack([Xhat[:N], U])
    d = np.sum((Q - q) ** 2, axis=1)

    # Evaluate uncertainty measure by inverse distance weighting
    ii = np.where(d < 1e-12)[0]  # Find the indices where d < 1e-12
    r = 0.0

    if ii.size > 0:
        ii = ii[0]  # Take the first index
        s = np.square(Y[ii] - Y[-1])
    else:
        w = np.exp(-d) / d
        sw = np.sum(w)
        s = (w / sw) @ (np.sum(np.square(Y[1:N+1] - y1), axis = 1) + alpha * np.sum(np.square(Xhat[1:N+1] - x1), axis = 1))
        
    r = np.arctan(1 / sw) * 2 / np.pi

    a = s + delta * r

    if const is not None:
        if const['flag']:
            p_min_y1 = np.sum(np.maximum(-y1 + const['y_min'], 0))
            p_max_y1 = np.sum(np.maximum(y1 - const['y_max'], 0))

            a = a - 1e6 * p_max_y1 - 1e6 * p_min_y1
            a = a.item()

    return a

# Optimized JAX version
def idw_jax(U, Xhat, Y, u_set, fx, fy, params, delta=1, alpha=1, const=None):
    def acqidw_vmap(u):
        return acqidw_jax(u, U, Xhat, Y, fx, fy, params, delta, alpha, const)
    
    aa = jax.vmap(acqidw_vmap)(u_set)
    u = u_set[jnp.argmax(aa)]
    return u

def acqidw_jax(u, U, Xhat, Y, fx, fy, params, delta=1, alpha=1, const=None):
    x = Xhat[-1]
    N = U.shape[0]
    x1 = fx(x, u, params)
    y1 = fy(x1, u, params)

    q = jnp.hstack([x, u])
    Q = jnp.hstack([Xhat[:N], U])
    d = jnp.sum((Q - q) ** 2, axis=1)

    # Compute a boolean mask instead of relying on `jnp.where`
    close_mask = d < 1e-12

    def if_close():
        # Take the first match if the mask is true anywhere
        first_match = jnp.argmax(close_mask)
        s = jnp.sum(jnp.square(Y[first_match] - Y[-1]))
        r = 0.0
        return s, r

    def if_not_close():
        # Compute weights and weighted sum for cases where d >= 1e-12
        weights = jnp.exp(-d) / d
        sum_weights = jnp.sum(weights)
        sum_squares_y = jnp.sum(jnp.square(Y[1:N+1] - y1), axis=1)
        sum_squares_x = jnp.sum(jnp.square(Xhat[1:N+1] - x1), axis=1)
        s = (weights / sum_weights) @ (sum_squares_y + alpha * sum_squares_x)
        r = jnp.arctan(1 / sum_weights) * 2 / jnp.pi

        return s, r

    # Use jax.lax.cond to choose between the two cases
    s, r = jax.lax.cond(jnp.any(close_mask), if_close, if_not_close)

    a = s + delta * r

    def apply_constraints(a):
        p_min_y1 = jnp.sum(jnp.maximum(-y1 + const['y_min'], 0))
        p_max_y1 = jnp.sum(jnp.maximum(y1 - const['y_max'], 0))
        return a - 1e12 * (p_max_y1 + p_min_y1)

    if const is not None:
        a = jax.lax.cond(const['flag'], apply_constraints, lambda a: a, a)
    return a


def _idw_fixed_jax_values(U, Xhat, Y, m, u_set, fx, fy, params, delta=1,
                            alpha=1, const=None, prediction_horizon=1,
                            input_weight=1.0, state_weight=1.0,
                            output_weight=1.0, next_state_weight=1.0,
                            distance_eps=1e-8, uncertainty_weight=1.0):
    """IDW acquisition values and confidence margins on a fixed-size history."""
    valid = jnp.arange(U.shape[0]) < m
    x = Xhat[m]
    eps = 1e-12
    input_weight = _distance_option(const, "input_distance_weight", input_weight)
    state_weight = _distance_option(const, "state_distance_weight", state_weight)
    output_weight = _distance_option(const, "output_distance_weight", output_weight)
    next_state_weight = _distance_option(const, "next_state_distance_weight", next_state_weight)
    distance_eps = _distance_option(const, "distance_eps", distance_eps)
    u_scale = _valid_component_scale(U, valid, distance_eps)
    x_scale = _valid_component_scale(Xhat[:-1], valid, distance_eps)
    y_scale = _valid_component_scale(Y[1:], valid, distance_eps)
    x_next_scale = _valid_component_scale(Xhat[1:], valid, distance_eps)

    def query_distances(query_x, query_u):
        return (
            state_weight * jnp.sum(((Xhat[:-1] - query_x) / x_scale) ** 2, axis=1)
            + input_weight * jnp.sum(((U - query_u) / u_scale) ** 2, axis=1)
        )

    def one_step_output(x0, u0):
        x1 = fx(x0, u0, params)
        return fy(x1, u0, params)

    if const is not None:
        y_hat_hist = jax.vmap(one_step_output)(Xhat[:-1], U)
        y_obs_hist = Y[1:]
        residual_hist = y_obs_hist - y_hat_hist

        hist_distances = (
            state_weight * jnp.sum(((Xhat[:-1, None, :] - Xhat[None, :-1, :]) / x_scale) ** 2, axis=2)
            + input_weight * jnp.sum(((U[:, None, :] - U[None, :, :]) / u_scale) ** 2, axis=2)
        )
        not_self = ~jnp.eye(U.shape[0], dtype=bool)
        pair_valid = valid[:, None] & valid[None, :] & not_self & (hist_distances >= eps)
        safe_hist_distances = jnp.where(pair_valid, hist_distances, 1.0)
        hist_weights = jnp.where(
            pair_valid, jnp.exp(-safe_hist_distances) / safe_hist_distances, 0.0
        )
        sum_hist_weights = jnp.sum(hist_weights, axis=1)
        loo_var = (
            hist_weights / jnp.maximum(sum_hist_weights[:, None], eps)
        ) @ jnp.sum(jnp.square(residual_hist), axis=1)
        loo_std = jnp.sqrt(jnp.maximum(loo_var, eps))

        cv = jnp.abs(residual_hist) / loo_std[:, None]
        cv = jnp.where(
            valid[:, None] & (sum_hist_weights[:, None] > eps), cv, -jnp.inf
        )
        cv_sorted = jnp.sort(cv, axis=0)
        valid_count = jnp.sum(valid & (sum_hist_weights > eps))
        confidence_alpha = const.get('confidence_alpha', 0.9)
        quantile_idx = jnp.maximum(
            jnp.ceil(confidence_alpha * valid_count).astype(jnp.int32) - 1, 0
        )
        padded_offset = U.shape[0] - valid_count
        sorted_idx = jnp.minimum(padded_offset + quantile_idx, U.shape[0] - 1)
        kappa_alpha = jnp.where(
            valid_count > 0, cv_sorted[sorted_idx], jnp.zeros(Y.shape[1])
        )
        valid_residual = jnp.where(valid[:, None], residual_hist, 0.0)
        lower_residual = jnp.maximum(-valid_residual, 0.0)
        upper_residual = jnp.maximum(valid_residual, 0.0)
        residual_sorted_lower = jnp.sort(
            jnp.where(valid[:, None], lower_residual, -jnp.inf), axis=0
        )
        residual_sorted_upper = jnp.sort(
            jnp.where(valid[:, None], upper_residual, -jnp.inf), axis=0
        )
        residual_count = jnp.sum(valid)
        residual_quantile_idx = jnp.maximum(
            jnp.ceil(confidence_alpha * residual_count).astype(jnp.int32) - 1, 0
        )
        residual_padded_offset = U.shape[0] - residual_count
        residual_sorted_idx = jnp.minimum(
            residual_padded_offset + residual_quantile_idx, U.shape[0] - 1
        )
        empirical_lower_margin = jnp.where(
            residual_count > 0,
            residual_sorted_lower[residual_sorted_idx],
            jnp.zeros(Y.shape[1]),
        )
        empirical_upper_margin = jnp.where(
            residual_count > 0,
            residual_sorted_upper[residual_sorted_idx],
            jnp.zeros(Y.shape[1]),
        )

    def output_idw_std(q, y1):
        """IDW output standard deviation at q over acquired transitions."""
        distances = query_distances(q[:Xhat.shape[1]], q[Xhat.shape[1]:])
        close = valid & (distances < eps)
        safe_distances = jnp.where(valid, distances, 1.0)

        def if_close(_):
            first_match = jnp.argmax(close)
            return jnp.abs(Y[first_match + 1] - y1)

        def if_not_close(_):
            weights = jnp.where(
                valid, jnp.exp(-safe_distances) / safe_distances, 0.0
            )
            sum_weights = jnp.sum(weights)
            output_sq_error = jnp.square(Y[1:] - y1)
            return jnp.sqrt((weights / sum_weights) @ output_sq_error)

        return jax.lax.cond(jnp.any(close), if_close, if_not_close, operand=None)

    def confidence_margin(q, y1):
        """Joseph-style IDW confidence margin for shrinking constraints."""
        beta = const.get('uncertainty_beta', 1.0 / 3.0)
        cap = beta * (const['y_max'] - const['y_min'])
        return jnp.minimum(kappa_alpha * output_idw_std(q, y1), cap)

    def asymmetric_idw_margin(q, y1):
        """Directional IDW margins from positive/negative residual energy."""
        beta = const.get('uncertainty_beta', 1.0 / 3.0)
        cap = beta * (const['y_max'] - const['y_min'])
        distances = query_distances(q[:Xhat.shape[1]], q[Xhat.shape[1]:])
        close = valid & (distances < eps)
        safe_distances = jnp.where(valid, distances, 1.0)

        def if_close(_):
            first_match = jnp.argmax(close)
            residual = Y[first_match + 1] - y1
            lower = jnp.maximum(-residual, 0.0)
            upper = jnp.maximum(residual, 0.0)
            return lower, upper

        def if_not_close(_):
            weights = jnp.where(
                valid, jnp.exp(-safe_distances) / safe_distances, 0.0
            )
            sum_weights = jnp.sum(weights)
            lower_sq = jnp.square(jnp.maximum(-(Y[1:] - y1), 0.0))
            upper_sq = jnp.square(jnp.maximum(Y[1:] - y1, 0.0))
            lower_std = jnp.sqrt((weights / sum_weights) @ lower_sq)
            upper_std = jnp.sqrt((weights / sum_weights) @ upper_sq)
            return (
                jnp.minimum(kappa_alpha * lower_std, cap),
                jnp.minimum(kappa_alpha * upper_std, cap),
            )

        return jax.lax.cond(jnp.any(close), if_close, if_not_close, operand=None)

    def cbf_violation(y_next, lower_margin, upper_margin):
        """Lightweight discrete CBF violation for the next predicted output."""
        if const is None:
            return 0.0

        gamma = const.get('cbf_gamma', 1.0)
        buffer = const.get('cbf_buffer', 0.0)
        y_now = Y[m]
        lower = const['y_min'] + lower_margin + buffer
        upper = const['y_max'] - upper_margin - buffer

        lower_barrier_now = jnp.maximum(y_now - lower, 0.0)
        upper_barrier_now = jnp.maximum(upper - y_now, 0.0)
        lower_required = lower + (1.0 - gamma) * lower_barrier_now
        upper_required = upper - (1.0 - gamma) * upper_barrier_now

        lower_violation = jnp.maximum(lower_required - y_next, 0.0)
        upper_violation = jnp.maximum(y_next - upper_required, 0.0)
        return jnp.sum(lower_violation + upper_violation)

    def acquisition(u):
        x1 = fx(x, u, params)
        y1 = fy(x1, u, params)

        distances = query_distances(x, u)
        close = valid & (distances < 1e-12)
        safe_distances = jnp.where(valid, distances, 1.0)

        def if_close(_):
            first_match = jnp.argmax(close)
            return output_weight * jnp.sum(((Y[first_match] - Y[m]) / y_scale) ** 2), 0.0

        def if_not_close(_):
            weights = jnp.where(
                valid, jnp.exp(-safe_distances) / safe_distances, 0.0
            )
            sum_weights = jnp.sum(weights)
            square_y = output_weight * jnp.sum(((Y[1:] - y1) / y_scale) ** 2, axis=1)
            square_x = next_state_weight * jnp.sum(((Xhat[1:] - x1) / x_next_scale) ** 2, axis=1)
            uncertainty = (
                (weights / sum_weights) @ (square_y + alpha * square_x)
            )
            representativeness = (
                jnp.arctan(1.0 / sum_weights) * 2.0 / jnp.pi
            )
            return uncertainty, representativeness

        uncertainty, representativeness = jax.lax.cond(
            jnp.any(close), if_close, if_not_close, operand=None
        )
        a = uncertainty_weight * uncertainty + delta * representativeness
        zero_margin = jnp.zeros(Y.shape[1])
        margin = zero_margin
        lower_margin = zero_margin
        upper_margin = zero_margin

        cbf = 0.0

        if const is not None:
            rho = const.get('penalty_rho', 1e12)
            cbf_rho = const.get('cbf_rho', 1e30)

            def apply_plain_constraints(value):
                p_min = jnp.sum(jnp.maximum(-y1 + const['y_min'], 0))
                p_max = jnp.sum(jnp.maximum(y1 - const['y_max'], 0))
                return value - rho * (p_min + p_max), zero_margin

            def apply_shrunk_constraints(value):
                q = jnp.hstack([x, u])
                shrunk_margin = confidence_margin(q, y1)
                p_min = jnp.sum(jnp.maximum(-y1 + const['y_min'] + shrunk_margin, 0))
                p_max = jnp.sum(jnp.maximum(y1 - const['y_max'] + shrunk_margin, 0))
                return (
                    value - rho * (p_min + p_max),
                    shrunk_margin,
                    shrunk_margin,
                    shrunk_margin,
                )

            def apply_asymmetric_idw_constraints(value):
                q = jnp.hstack([x, u])
                lower_idw_margin, upper_idw_margin = asymmetric_idw_margin(q, y1)
                p_min = jnp.sum(
                    jnp.maximum(-y1 + const['y_min'] + lower_idw_margin, 0)
                )
                p_max = jnp.sum(
                    jnp.maximum(y1 - const['y_max'] + upper_idw_margin, 0)
                )
                combined_margin = jnp.maximum(lower_idw_margin, upper_idw_margin)
                return (
                    value - rho * (p_min + p_max),
                    combined_margin,
                    lower_idw_margin,
                    upper_idw_margin,
                )

            def apply_empirical_constraints(value):
                p_min = jnp.sum(
                    jnp.maximum(-y1 + const['y_min'] + empirical_lower_margin, 0)
                )
                p_max = jnp.sum(
                    jnp.maximum(y1 - const['y_max'] + empirical_upper_margin, 0)
                )
                combined_margin = jnp.maximum(
                    empirical_lower_margin, empirical_upper_margin
                )
                return (
                    value - rho * (p_min + p_max),
                    combined_margin,
                    empirical_lower_margin,
                    empirical_upper_margin,
                )

            def apply_horizon_constraints(value):
                first_step_buffer = const.get('first_step_buffer', 0.0)
                first_step_weight = const.get('first_step_weight', 1.0)
                first_step_violation = (
                    jnp.sum(jnp.maximum(-y1 + const['y_min'] + first_step_buffer, 0))
                    + jnp.sum(jnp.maximum(y1 - const['y_max'] + first_step_buffer, 0))
                )

                def penalty_step(carry, _):
                    x_pred, step = carry
                    y_pred = fy(x_pred, u, params)
                    p_min = jnp.sum(jnp.maximum(-y_pred + const['y_min'], 0))
                    p_max = jnp.sum(jnp.maximum(y_pred - const['y_max'], 0))
                    weight = jnp.where(step == 0, first_step_weight, 1.0)
                    x_next = fx(x_pred, u, params)
                    return (x_next, step + 1), weight * (p_min + p_max)

                _, penalties = jax.lax.scan(
                    penalty_step, (x1, 0), xs=None, length=prediction_horizon
                )
                hard_gate = 1e30 * first_step_violation
                return (
                    value - hard_gate - rho * jnp.sum(penalties),
                    zero_margin,
                    zero_margin,
                    zero_margin,
                )

            def apply_non_horizon_constraints(value):
                return jax.lax.cond(
                    const['flag'] == 5,
                    apply_empirical_constraints,
                    lambda inner_value: jax.lax.cond(
                        const['flag'] == 4,
                        apply_asymmetric_idw_constraints,
                        lambda asym_value: jax.lax.cond(
                            const['flag'] == 2,
                            apply_shrunk_constraints,
                            lambda plain_value: jax.lax.cond(
                                const['flag'] == 1,
                                lambda v: (
                                    apply_plain_constraints(v)[0],
                                    zero_margin,
                                    zero_margin,
                                    zero_margin,
                                ),
                                lambda v: (v, zero_margin, zero_margin, zero_margin),
                                plain_value,
                            ),
                            asym_value,
                        ),
                        inner_value,
                    ),
                    value,
                )

            a, margin, lower_margin, upper_margin = jax.lax.cond(
                const['flag'] == 3,
                apply_horizon_constraints,
                apply_non_horizon_constraints,
                a,
            )
            cbf_lower_margin = jax.lax.cond(
                const.get('cbf_use_margin', False),
                lambda _: margin,
                lambda _: zero_margin,
                operand=None,
            )
            cbf_upper_margin = jax.lax.cond(
                const.get('cbf_use_margin', False),
                lambda _: margin,
                lambda _: zero_margin,
                operand=None,
            )
            cbf = cbf_violation(y1, cbf_lower_margin, cbf_upper_margin)
            a = jax.lax.cond(
                const.get('safety_filter', False),
                lambda value: value - cbf_rho * cbf,
                lambda value: value,
                a,
            )
        return a, margin, lower_margin, upper_margin, cbf

    return jax.vmap(acquisition)(u_set)


def _idw_fixed_jax(U, Xhat, Y, m, u_set, fx, fy, params, delta=1,
                     alpha=1, const=None, prediction_horizon=1,
                     input_weight=1.0, state_weight=1.0,
                     output_weight=1.0, next_state_weight=1.0,
                     distance_eps=1e-8, uncertainty_weight=1.0):
    """IDW with fixed-size history arrays and a dynamic valid length."""
    values, _, _, _, _ = _idw_fixed_jax_values(
        U, Xhat, Y, m, u_set, fx, fy, params, delta, alpha, const,
        prediction_horizon, input_weight, state_weight, output_weight,
        next_state_weight, distance_eps, uncertainty_weight,
    )
    return u_set[jnp.argmax(values)]


def _idw_fixed_jax_plain_constraints(
        U, Xhat, Y, m, u_set, fx, fy, params, delta=1, alpha=1,
        y_min=0.0, y_max=1.0, penalty_rho=1e12,
        input_weight=1.0, state_weight=1.0,
        output_weight=1.0, next_state_weight=1.0,
        distance_eps=1e-8):
    """IDW with only the deterministic one-step output penalty.

    This flag-1 path intentionally avoids confidence-margin history matrices,
    which are only needed by the uncertainty-aware constraint variants.
    """
    valid = jnp.arange(U.shape[0]) < m
    x = Xhat[m]
    u_scale = _valid_component_scale(U, valid, distance_eps)
    x_scale = _valid_component_scale(Xhat[:-1], valid, distance_eps)
    y_scale = _valid_component_scale(Y[1:], valid, distance_eps)
    x_next_scale = _valid_component_scale(Xhat[1:], valid, distance_eps)

    def acquisition(u):
        x1 = fx(x, u, params)
        y1 = fy(x1, u, params)

        distances = (
            state_weight * jnp.sum(((Xhat[:-1] - x) / x_scale) ** 2, axis=1)
            + input_weight * jnp.sum(((U - u) / u_scale) ** 2, axis=1)
        )
        close = valid & (distances < 1e-12)
        safe_distances = jnp.where(valid, distances, 1.0)

        def if_close(_):
            first_match = jnp.argmax(close)
            return output_weight * jnp.sum(((Y[first_match] - Y[m]) / y_scale) ** 2), 0.0

        def if_not_close(_):
            weights = jnp.where(
                valid, jnp.exp(-safe_distances) / safe_distances, 0.0
            )
            sum_weights = jnp.sum(weights)
            square_y = output_weight * jnp.sum(((Y[1:] - y1) / y_scale) ** 2, axis=1)
            square_x = next_state_weight * jnp.sum(((Xhat[1:] - x1) / x_next_scale) ** 2, axis=1)
            uncertainty = (
                (weights / sum_weights) @ (square_y + alpha * square_x)
            )
            representativeness = (
                jnp.arctan(1.0 / sum_weights) * 2.0 / jnp.pi
            )
            return uncertainty, representativeness

        uncertainty, representativeness = jax.lax.cond(
            jnp.any(close), if_close, if_not_close, operand=None
        )
        value = uncertainty + delta * representativeness
        p_min = jnp.sum(jnp.maximum(-y1 + y_min, 0))
        p_max = jnp.sum(jnp.maximum(y1 - y_max, 0))
        return value - penalty_rho * (p_min + p_max)

    values = jax.vmap(acquisition)(u_set)
    return u_set[jnp.argmax(values)]


def _idw_fixed_jax_with_margin(U, Xhat, Y, m, u_set, fx, fy, params, delta=1,
                                 alpha=1, const=None, prediction_horizon=1,
                                 input_weight=1.0, state_weight=1.0,
                                 output_weight=1.0, next_state_weight=1.0,
                                 distance_eps=1e-8):
    """Return the selected IDW input and its realized confidence margin."""
    values, margins, _, _, _ = _idw_fixed_jax_values(
        U, Xhat, Y, m, u_set, fx, fy, params, delta, alpha, const,
        prediction_horizon, input_weight, state_weight, output_weight,
        next_state_weight, distance_eps,
    )
    best = jnp.argmax(values)
    return u_set[best], margins[best]


def _idw_fixed_jax_asymmetric_margin(U, Xhat, Y, m, u_set, fx, fy, params,
                                       delta=1, alpha=1, const=None,
                                       prediction_horizon=1,
                                       input_weight=1.0, state_weight=1.0,
                                       output_weight=1.0, next_state_weight=1.0,
                                       distance_eps=1e-8):
    """Return selected IDW input and empirical lower/upper residual margins."""
    values, _, lower_margins, upper_margins, _ = _idw_fixed_jax_values(
        U, Xhat, Y, m, u_set, fx, fy, params, delta, alpha, const,
        prediction_horizon, input_weight, state_weight, output_weight,
        next_state_weight, distance_eps,
    )
    best = jnp.argmax(values)
    return u_set[best], jnp.hstack([lower_margins[best], upper_margins[best]])


def _idw_fixed_jax_filtered_with_margin(U, Xhat, Y, m, u_set, fx, fy, params,
                                          delta=1, alpha=1, const=None,
                                          prediction_horizon=1,
                                          input_weight=1.0, state_weight=1.0,
                                          output_weight=1.0, next_state_weight=1.0,
                                          distance_eps=1e-8):
    """IDW over safe inputs first; fall back to minimum CBF violation."""
    values, margins, _, _, cbf_violations = _idw_fixed_jax_values(
        U, Xhat, Y, m, u_set, fx, fy, params, delta, alpha, const,
        prediction_horizon, input_weight, state_weight, output_weight,
        next_state_weight, distance_eps,
    )
    eps = const.get('cbf_eps', 1e-12)
    safe = cbf_violations <= eps
    safe_values = jnp.where(safe, values, -jnp.inf)
    fallback_values = -cbf_violations
    filtered_values = jnp.where(jnp.any(safe), safe_values, fallback_values)
    best = jnp.argmax(filtered_values)
    return u_set[best], margins[best]


def _idw_fixed_jax_filtered_with_asymmetric_margin(
    U, Xhat, Y, m, u_set, fx, fy, params, delta=1, alpha=1, const=None,
    prediction_horizon=1, input_weight=1.0, state_weight=1.0,
    output_weight=1.0, next_state_weight=1.0, distance_eps=1e-8,
):
    """Filtered IDW returning lower/upper margins for asymmetric constraints."""
    values, _, lower_margins, upper_margins, cbf_violations = _idw_fixed_jax_values(
        U, Xhat, Y, m, u_set, fx, fy, params, delta, alpha, const,
        prediction_horizon, input_weight, state_weight, output_weight,
        next_state_weight, distance_eps,
    )
    eps = const.get('cbf_eps', 1e-12)
    safe = cbf_violations <= eps
    safe_values = jnp.where(safe, values, -jnp.inf)
    fallback_values = -cbf_violations
    filtered_values = jnp.where(jnp.any(safe), safe_values, fallback_values)
    best = jnp.argmax(filtered_values)
    return u_set[best], jnp.hstack([lower_margins[best], upper_margins[best]])


def _idw_fixed_jax_sequence(U, Xhat, Y, m, u_set, u_future_set, fx, fy,
                              params, delta=1, alpha=1, const=None,
                              prediction_horizon=3, input_weight=1.0,
                              state_weight=1.0, output_weight=1.0,
                              next_state_weight=1.0, distance_eps=1e-8):
    """IDW with exhaustive finite-horizon input-sequence enumeration."""
    values, _, _, _, _ = _idw_fixed_jax_values(
        U, Xhat, Y, m, u_set, fx, fy, params, delta, alpha, const=None,
        prediction_horizon=1, input_weight=input_weight,
        state_weight=state_weight, output_weight=output_weight,
        next_state_weight=next_state_weight, distance_eps=distance_eps,
    )
    x = Xhat[m]
    rho = const.get('penalty_rho', 1e12)
    cbf_rho = const.get('cbf_rho', 1e30)
    cbf_gamma = const.get('cbf_gamma', 1.0)
    cbf_buffer = const.get('cbf_buffer', 0.0)
    first_step_buffer = const.get('first_step_buffer', 0.0)
    first_step_weight = const.get('first_step_weight', 1.0)
    n_first = u_set.shape[0]
    n_future = u_future_set.shape[0]
    n_seq = n_first * (n_future ** (prediction_horizon - 1))

    def sequence_from_index(index):
        first_index = index % n_first
        remaining = index // n_first
        future_digits = []
        for _ in range(prediction_horizon - 1):
            future_digits.append(remaining % n_future)
            remaining = remaining // n_future
        future_indices = jnp.stack(future_digits)
        return first_index, future_indices

    def sequence_objective(index):
        first_index, future_indices = sequence_from_index(index)
        u_seq = jnp.vstack([u_set[first_index][None, :], u_future_set[future_indices]])
        base_value = values[first_index]

        def penalty_step(carry, u_step):
            x_pred, step = carry
            x_next = fx(x_pred, u_step, params)
            y_pred = fy(x_next, u_step, params)
            p_min = jnp.sum(jnp.maximum(-y_pred + const['y_min'], 0))
            p_max = jnp.sum(jnp.maximum(y_pred - const['y_max'], 0))
            weight = jnp.where(step == 0, first_step_weight, 1.0)
            return (x_next, step + 1), weight * (p_min + p_max)

        _, penalties = jax.lax.scan(penalty_step, (x, 0), u_seq)
        x_first = fx(x, u_seq[0], params)
        y_first = fy(x_first, u_seq[0], params)
        first_step_violation = (
            jnp.sum(jnp.maximum(-y_first + const['y_min'] + first_step_buffer, 0))
            + jnp.sum(jnp.maximum(y_first - const['y_max'] + first_step_buffer, 0))
        )
        lower = const['y_min'] + cbf_buffer
        upper = const['y_max'] - cbf_buffer
        lower_barrier_now = jnp.maximum(Y[m] - lower, 0.0)
        upper_barrier_now = jnp.maximum(upper - Y[m], 0.0)
        lower_required = lower + (1.0 - cbf_gamma) * lower_barrier_now
        upper_required = upper - (1.0 - cbf_gamma) * upper_barrier_now
        cbf_violation = (
            jnp.sum(jnp.maximum(lower_required - y_first, 0.0))
            + jnp.sum(jnp.maximum(y_first - upper_required, 0.0))
        )
        hard_gate = 1e30 * first_step_violation
        cbf_gate = jnp.where(const.get('safety_filter', False), cbf_rho * cbf_violation, 0.0)
        return base_value - hard_gate - cbf_gate - rho * jnp.sum(penalties)

    sequence_indices = jnp.arange(n_seq)
    sequence_values = jax.vmap(sequence_objective)(sequence_indices)
    best_first_index, _ = sequence_from_index(jnp.argmax(sequence_values))
    return u_set[best_first_index]


# Apply JIT with static_argnums for fx and fy functions
idw_jax = jax.jit(idw_jax, static_argnums=(4, 5))
acqidw_jax = jax.jit(acqidw_jax, static_argnums=(4, 5))
idw_fixed_jax = jax.jit(_idw_fixed_jax, static_argnums=(5, 6, 11))
idw_fixed_jax_plain_constraints = jax.jit(
    _idw_fixed_jax_plain_constraints, static_argnums=(5, 6)
)
idw_fixed_jax_with_margin = jax.jit(
    _idw_fixed_jax_with_margin, static_argnums=(5, 6, 11)
)
idw_fixed_jax_asymmetric_margin = jax.jit(
    _idw_fixed_jax_asymmetric_margin, static_argnums=(5, 6, 11)
)
idw_fixed_jax_filtered_with_margin = jax.jit(
    _idw_fixed_jax_filtered_with_margin, static_argnums=(5, 6, 11)
)
idw_fixed_jax_filtered_with_asymmetric_margin = jax.jit(
    _idw_fixed_jax_filtered_with_asymmetric_margin, static_argnums=(5, 6, 11)
)
idw_fixed_jax_sequence = jax.jit(
    _idw_fixed_jax_sequence, static_argnums=(6, 7, 12)
)

if __name__ == "__main__":
    # Generate sample data
    ny = 10
    U = np.random.rand(100, 10)
    Xhat = np.random.rand(101, 10)
    Y = np.random.rand(101, ny)
    u_set = np.random.rand(500, 10)
    U_jax = jnp.array(U)
    Xhat_jax = jnp.array(Xhat)
    Y_jax = jnp.array(Y)
    u_set_jax = jnp.array(u_set)

    # Dummy functions for fx and fy using JAX operations
    def fx(x, u, params):
        return x + u

    def fy(x1, u, params):
        return x1 * u

    const = {}
    const['flag'] = 1
    const['y_min'] = jnp.zeros(ny) 
    const['y_max'] = jnp.ones(ny)*10 

    # Timing the NumPy version
    time_numpy = timeit.timeit(lambda: idw(U, Xhat, Y, u_set, fx, fy, None, const = const), number=10)

    # Warm-up JAX to compile the function
    key = random.PRNGKey(0)
    idw_jax(U_jax, Xhat_jax, Y_jax, u_set_jax, fx, fy, None, const = const).block_until_ready()

    # Timing the JAX version
    time_jax = timeit.timeit(lambda: idw_jax(U_jax, Xhat_jax, Y_jax, u_set_jax, fx, fy,None, const = const).block_until_ready(), number=10)

    print(f"NumPy version time: {time_numpy:.6f} seconds")
    print(f"JAX version time: {time_jax:.6f} seconds")
    
    # timing results: JAX version is much faster
    # NumPy version time: 0.178313 seconds
    # JAX version time: 0.011530 seconds
