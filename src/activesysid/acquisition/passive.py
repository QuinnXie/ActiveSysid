import numpy as np


def passive(u_set, x=None, fx=None, fy=None, params=None, const=None):
    """Select a candidate uniformly at random.

    When constraints and a model state are supplied, candidates whose
    one-step predicted output violates the output bounds are discarded first.
    If no candidate is feasible, sampling falls back to the complete set.
    """
    candidate_indices = np.arange(u_set.shape[0])

    if const is not None and const.get('flag', 0) and x is not None:
        feasible = []
        y_min = np.asarray(const['y_min'])
        y_max = np.asarray(const['y_max'])

        for index, candidate in enumerate(u_set):
            x1 = fx(x, candidate, params)
            y1 = np.asarray(fy(x1, candidate, params))
            if (
                np.all(np.isfinite(y1))
                and np.all(y1 >= y_min)
                and np.all(y1 <= y_max)
            ):
                feasible.append(index)

        if feasible:
            candidate_indices = np.asarray(feasible)

    index = candidate_indices[np.random.randint(0, candidate_indices.size)]
    return u_set[index]


import jax
import jax.numpy as jnp
from jax import random


@jax.jit
def passive_jax(u_set, key):
    Nu_set = u_set.shape[0]
    idx = random.randint(key, (1,), 0, Nu_set)[0]
    u = u_set[idx, :]
    return u


def _passive_constrained_jax(
        u_set, key, x, fx, fy, params, y_min, y_max):
    """JAX implementation of constrained random sampling."""
    def predict_output(u):
        x1 = fx(x, u, params)
        return fy(x1, u, params)

    predicted_outputs = jax.vmap(predict_output)(u_set)
    output_axes = tuple(range(1, predicted_outputs.ndim))
    feasible = (
        jnp.all(jnp.isfinite(predicted_outputs), axis=output_axes)
        & jnp.all(predicted_outputs >= y_min, axis=output_axes)
        & jnp.all(predicted_outputs <= y_max, axis=output_axes)
    )

    # If every candidate violates the constraints, all candidates receive
    # equal probability, reproducing unconstrained passive sampling.
    weights = feasible.astype(jnp.float32)
    weights = jnp.where(jnp.any(feasible), weights, jnp.ones_like(weights))
    index = random.choice(key, u_set.shape[0], p=weights)
    return u_set[index]


passive_constrained_jax = jax.jit(
    _passive_constrained_jax, static_argnums=(3, 4)
)

if __name__ == "__main__":
    # from jax import random
    import timeit
    # Generate a sample u_set
    u_set_np = np.random.rand(1000, 10)
    u_set_jax = jnp.array(u_set_np)

    # Timing the NumPy version
    time_numpy = timeit.timeit(lambda: passive(u_set_np), number=1000)

    # Timing the JAX version
    key = random.PRNGKey(0)
    time_jax = timeit.timeit(lambda: passive_jax(u_set_jax, key).block_until_ready(), number=1000)

    print(f"NumPy version time: {time_numpy:.6f} seconds")
    print(f"JAX version time: {time_jax:.6f} seconds")

    # result: NumPy version is 120 times faster
    # NumPy version time: 0.005635 seconds
    # JAX version time: 0.644868 seconds
