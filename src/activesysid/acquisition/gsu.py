import numpy as np
import jax
import jax.numpy as jnp

def GSu(U, u_set, x = None, fx=None, fy=None, params=None, const=None):
    # Vectorized computation of acquisition values
    aa = np.array([acqGSu(u, U, x, fx, fy, params, const) for u in u_set])

    u = u_set[np.argmax(aa)]
    return u

def acqGSu(u, U, x = None, fx=None, fy=None, params=None, const=None):
    ddz = np.sum((U - u) ** 2, axis=1)

    a = np.min(ddz)
    if const is not None:
        if const['flag']:
            x1 = fx(x, u, params)
            y1 = fy(x1, u, params)
            p_min_y1 = np.sum(np.maximum(-y1 + const['y_min'], 0))
            p_max_y1 = np.sum(np.maximum(y1 - const['y_max'], 0))
            a = a - 1e6 * p_max_y1 - 1e6 * p_min_y1
            a = a.item()

    return a

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

def _GSu_jax(U, u_set, x = None, fx=None, fy=None, params=None, const=None):
    def acqGSu_vmap(u):
        return acqGSu_jax(u, U, x, fx, fy, params, const)
    aa = jax.vmap(acqGSu_vmap)(u_set)
    u = u_set[jnp.argmax(aa)]
    return u

def acqGSu_jax(u, U, x = None, fx=None, fy=None, params=None, const=None):

    ddz = jnp.sum((U - u) ** 2, axis=1)

    a = jnp.min(ddz)

    # Use jax.lax.cond for const['flag']
    def apply_constraints(a):
        x1 = fx(x, u, params)
        y1 = fy(x1, u, params)

        p_min_y1 = jnp.sum(jnp.maximum(-y1 + const['y_min'], 0))
        p_max_y1 = jnp.sum(jnp.maximum(y1 - const['y_max'], 0))

        return a - 1e6 * p_max_y1 - 1e6 * p_min_y1

    if const is not None:
        a = jax.lax.cond(const['flag'], apply_constraints, lambda a: a, a)

    return a


def _GSu_fixed_jax(U, m, u_set, x=None, fx=None, fy=None, params=None,
                   const=None, input_weight=1.0, distance_eps=1e-8):
    """GSu with fixed-size history arrays, normalized distances, and a dynamic valid length."""
    valid = jnp.arange(U.shape[0]) < m
    input_weight = _distance_option(const, "input_distance_weight", input_weight)
    distance_eps = _distance_option(const, "distance_eps", distance_eps)
    u_scale = _valid_component_scale(U, valid, distance_eps)
    distances = input_weight * jnp.sum(((U[:, None, :] - u_set[None, :, :]) / u_scale) ** 2, axis=2)
    values = jnp.min(jnp.where(valid[:, None], distances, jnp.inf), axis=0)

    if const is not None:
        def apply_constraints(scores):
            x1 = jax.vmap(lambda u: fx(x, u, params))(u_set)
            y1 = jax.vmap(lambda x_next, u: fy(x_next, u, params))(x1, u_set)
            p_min = jnp.sum(jnp.maximum(-y1 + const['y_min'], 0), axis=1)
            p_max = jnp.sum(jnp.maximum(y1 - const['y_max'], 0), axis=1)
            return scores - 1e12 * (p_min + p_max)

        values = jax.lax.cond(const['flag'], apply_constraints, lambda scores: scores, values)
    return u_set[jnp.argmax(values)]


GSu_jax = jax.jit(_GSu_jax, static_argnums=(3, 4))
acqiGS_jitted = jax.jit(acqGSu_jax, static_argnums=(3, 4))
GSu_fixed_jax = jax.jit(_GSu_fixed_jax, static_argnums=(4, 5))

if __name__ == "__main__":
    from jax import random
    import timeit
    # Generate sample data
    ny = 10
    U = np.random.rand(100, 10)
    Xhat = np.random.rand(101, 10)
    u_set = np.random.rand(500, 10)
    U_jax = jnp.array(U)
    Xhat_jax = jnp.array(Xhat)
    u_set_jax = jnp.array(u_set)

    params = 0.0
    def fx(x, u, params):
        return x + u

    def fy(x1, u, params):
        return x1 - u
    
    const = {}
    const['flag'] = 1
    const['y_min'] = jnp.zeros(ny) 
    const['y_max'] = jnp.ones(ny)*10 

    # Timing the NumPy version
    time_numpy = timeit.timeit(lambda: GSu(U, u_set, Xhat[-1], fx, fy, params = params, const = const), number=10)

    # Warm-up JAX to compile the function
    key = random.PRNGKey(0)
    GSu_jax(U_jax, u_set_jax, Xhat_jax[-1], fx, fy, params, const).block_until_ready()

    # Timing the JAX version
    time_jax = timeit.timeit(lambda: GSu_jax(U_jax, u_set_jax, Xhat_jax[-1], fx, fy, params = params, const = const).block_until_ready(), number=10)

    print(f"NumPy version time: {time_numpy:.6f} seconds")
    print(f"JAX version time: {time_jax:.6f} seconds")

    # Timing results: JAX version is much faster
    # NumPy version time: 32.417515 seconds
    # JAX version time: 0.422861 seconds
