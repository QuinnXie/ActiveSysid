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
def iGS(U, Xhat, Y, u_set, fx, fy, params, const=None):
    aa = np.array([acqiGS_numpy(u, U, Xhat, Y, fx, fy, params, const) for u in u_set])
    u = u_set[np.argmax(aa)]
    return u

def acqiGS_numpy(u, U, Xhat, Y, fx, fy, params, const=None):
    x = Xhat[-1]
    N = U.shape[0]
    x1 = fx(x, u, params)
    y1 = fy(x1, u, params)

    q = np.hstack([x, u])
    Q = np.hstack([Xhat[:N, :], U])
    ddx = np.sum((Q - q) ** 2, axis=1)
    dx = np.min(ddx)

    r = np.hstack([y1, x1])
    R = np.hstack([Y[1:N+1, :], Xhat[1:N+1, :]])
    ddy = np.sum((R - r) ** 2, axis=1)
    dy = np.min(ddy)
    
    a = dx * dy
    if const is not None:
        if const['flag']:
            p_min_y1 = np.sum(np.maximum(-y1 + const['y_min'], 0))
            p_max_y1 = np.sum(np.maximum(y1 - const['y_max'], 0))
            a = a - 1e6 * p_max_y1 - 1e6 * p_min_y1
            a = a.item()
            
    return a

# Optimized JAX version
def _iGS_jax(U, Xhat, Y, u_set, fx, fy, params, const=None):
    def acqiGS_vmap(u):
        return acqiGS_jax(u, U, Xhat, Y, fx, fy, params, const)
    aa = jax.vmap(acqiGS_vmap)(u_set)
    u = u_set[jnp.argmax(aa)]

    return u

def _acqiGS_jax(u, U, Xhat, Y, fx, fy, params, const=None):
    x = Xhat[-1]
    N = U.shape[0]
    x1 = fx(x, u, params)
    y1 = fy(x1, u, params)

    q = jnp.hstack([x, u])
    Q = jnp.hstack([Xhat[:N], U])
    ddx = jnp.sum((Q - q) ** 2, axis=1)
    dx = jnp.min(ddx)

    r = jnp.hstack([y1, x1])
    R = jnp.hstack([Y[1:N+1], Xhat[1:N+1]])
    ddy = jnp.sum((R - r) ** 2, axis=1)
    dy = jnp.min(ddy)

    a = dx * dy

    # Use jax.lax.cond for const['flag']
    def apply_constraints(a):
        p_min_y1 = jnp.sum(jnp.maximum(-y1 + const['y_min'], 0))
        p_max_y1 = jnp.sum(jnp.maximum(y1 - const['y_max'], 0))
        return a - 1e12 * (p_max_y1 + p_min_y1)

    if const is not None:
        a = jax.lax.cond(const['flag'], apply_constraints, lambda a: a, a)

    return a


def _iGS_fixed_jax(U, Xhat, Y, m, u_set, fx, fy, params, const=None,
                   input_weight=1.0, state_weight=1.0,
                   output_weight=1.0, next_state_weight=1.0,
                   distance_eps=1e-8):
    """iGS with fixed-size history arrays, normalized distances, and a dynamic valid length."""
    valid = jnp.arange(U.shape[0]) < m
    x = Xhat[m]
    input_weight = _distance_option(const, "input_distance_weight", input_weight)
    state_weight = _distance_option(const, "state_distance_weight", state_weight)
    output_weight = _distance_option(const, "output_distance_weight", output_weight)
    next_state_weight = _distance_option(const, "next_state_distance_weight", next_state_weight)
    distance_eps = _distance_option(const, "distance_eps", distance_eps)
    u_scale = _valid_component_scale(U, valid, distance_eps)
    x_scale = _valid_component_scale(Xhat[:-1], valid, distance_eps)
    y_scale = _valid_component_scale(Y[1:], valid, distance_eps)
    x_next_scale = _valid_component_scale(Xhat[1:], valid, distance_eps)

    x1 = jax.vmap(lambda u: fx(x, u, params))(u_set)
    y1 = jax.vmap(lambda x_next, u: fy(x_next, u, params))(x1, u_set)

    state_distances = state_weight * jnp.sum(((Xhat[:-1] - x) / x_scale) ** 2, axis=1)
    input_distances = input_weight * jnp.sum(((U[:, None, :] - u_set[None, :, :]) / u_scale) ** 2, axis=2)
    dx_all = state_distances[:, None] + input_distances
    dx = jnp.min(jnp.where(valid[:, None], dx_all, jnp.inf), axis=0)

    output_distances = output_weight * jnp.sum(((Y[1:, None, :] - y1[None, :, :]) / y_scale) ** 2, axis=2)
    next_state_distances = next_state_weight * jnp.sum(((Xhat[1:, None, :] - x1[None, :, :]) / x_next_scale) ** 2, axis=2)
    dy_all = output_distances + next_state_distances
    dy = jnp.min(jnp.where(valid[:, None], dy_all, jnp.inf), axis=0)
    values = dx * dy

    if const is not None:
        def apply_constraints(scores):
            p_min = jnp.sum(jnp.maximum(-y1 + const['y_min'], 0), axis=1)
            p_max = jnp.sum(jnp.maximum(y1 - const['y_max'], 0), axis=1)
            return scores - 1e12 * (p_min + p_max)

        values = jax.lax.cond(const['flag'], apply_constraints, lambda scores: scores, values)
    return u_set[jnp.argmax(values)]


# Apply JIT with static arguments functionally
iGS_jax = jax.jit(_iGS_jax, static_argnums=(4, 5))
acqiGS_jax = jax.jit(_acqiGS_jax, static_argnums=(4, 5))
iGS_fixed_jax = jax.jit(_iGS_fixed_jax, static_argnums=(5, 6))


if __name__ == "__main__":
    # Generate sample data
    ny = 10
    U = np.random.rand(100, 10)
    Xhat = np.random.rand(101, 10)
    Y = np.random.rand(101, ny)
    u_set = np.random.rand(50, 10)

    U_jax = jnp.array(U)
    Xhat_jax = jnp.array(Xhat)
    Y_jax = jnp.array(Y)
    u_set_jax = jnp.array(u_set)

    # Dummy functions for fx and fy
    def fx(x, u, params):
        return x + u

    def fy(x1, u, params):
        return x1 - u

    const = {}
    const['flag'] = 1
    const['y_min'] = jnp.zeros(ny) 
    const['y_max'] = jnp.ones(ny)*10 

    # Warm-up JAX to compile the function
    key = random.PRNGKey(0)
    iGS_jax(U_jax, Xhat_jax, Y_jax, u_set_jax, fx, fy, None, const = const).block_until_ready()

    # Timing the NumPy version
    time_numpy = timeit.timeit(lambda: iGS(U, Xhat, Y, u_set, fx, fy, None, const = const), number=30)

    # Timing the JAX version
    time_jax = timeit.timeit(lambda: iGS_jax(U_jax, Xhat_jax, Y_jax, u_set_jax, fx, fy, None, const = const).block_until_ready(), number=30)

    print(f"NumPy version time: {time_numpy:.6f} seconds")
    print(f"JAX version time: {time_jax:.6f} seconds")

    # timing results: JAX version is much faster
    # NumPy version time: 0.178313 seconds
    # JAX version time: 0.011530 seconds
