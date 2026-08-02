"""Shared utilities for the core system-identification modules."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax_sysid.utils import compute_scores, standard_scale, unscale


__all__ = [
    "arrays_to_1d",
    "compute_constraint_violation",
    "compute_gradient",
    "compute_scores",
    "hstack_arrays",
    "init_results",
    "one_d_to_arrays",
    "params2ABCD",
    "params_to_abcd",
    "permute_matrix",
    "replace_nan_with_previous",
    "save_results",
    "scale",
    "sigmoid",
    "softplus",
    "standard_scale",
    "transform_matrix",
    "unscale",
]


def scale(X, offset, gain):
    """Scale a signal using ``Xs = (X - offset) * gain``."""
    return (X - offset) * gain


def hstack_arrays(array_list):
    """Flatten array leaves and concatenate them along the feature axis."""
    if not array_list:
        raise ValueError("array_list must contain at least one array")
    row_count = array_list[0].shape[0]
    return jnp.hstack([
        jnp.asarray(array).reshape(row_count, -1)
        for array in array_list
    ])


def params_to_abcd(params):
    """Extract A, B, C and a zero-feedthrough D from linear parameters."""
    if len(params) < 3:
        raise ValueError("linear params must contain at least A, B, and C")
    A, B, C = (jnp.asarray(value) for value in params[:3])
    D = jnp.zeros((C.shape[0], B.shape[1]), dtype=C.dtype)
    return A, B, C, D


params2ABCD = params_to_abcd


def permute_matrix(matrix):
    rows, cols = matrix.shape
    block_size = jnp.count_nonzero(matrix[0])

    def transform_with_block_size(matrix, block_size):
        transformed = jnp.zeros((rows, cols), dtype=matrix.dtype)

        def update_row(i, current):
            block = lax.dynamic_slice(matrix[i], (i * block_size,), (block_size,))
            for j in range(block_size):
                current = current.at[i, i + j * rows].set(block[j])
            return current

        return lax.fori_loop(0, rows, update_row, transformed)

    return transform_with_block_size(matrix, block_size)


def transform_matrix(matrix):
    rows, cols = matrix.shape
    transformed = np.zeros((rows, cols), dtype=matrix.dtype)
    block_size = np.count_nonzero(matrix[0])
    for i in range(rows):
        block = matrix[i, i * block_size : (i + 1) * block_size]
        for j in range(block_size):
            transformed[i, i + j * rows] = block[j]
    return transformed


@jax.jit
def arrays_to_1d(arrays):
    return jnp.concatenate([array.flatten() for array in arrays])


def one_d_to_arrays(one_d_array, arrays):
    result = []
    index = 0
    for array in arrays:
        size = array.size
        result.append(one_d_array[index : index + size].reshape(array.shape))
        index += size
    return result


def init_results(N_set, N_exp, nu, nx, ny, N_train_max, N_test):
    """Allocate storage for samples and scores from repeated experiments."""
    del nx  # Retained in the signature for compatibility.
    samples = {
        "U_train": np.zeros((N_set, N_exp, N_train_max, nu)),
        "U_test": np.zeros((N_set, N_exp, N_test, nu)),
        "Y_train": np.zeros((N_set, N_exp, N_train_max, ny)),
        "Y_test": np.zeros((N_set, N_exp, N_test, ny)),
        "Yhat_train": np.zeros((N_set, N_exp, N_train_max, ny)),
        "Yhat_test": np.zeros((N_set, N_exp, N_test, ny)),
    }
    scores = {
        "R2_train": np.zeros((N_set, N_exp, N_train_max, ny)),
        "R2_test": np.zeros((N_set, N_exp, N_train_max, ny)),
        "BFR_train": np.zeros((N_set, N_exp, N_train_max, ny)),
        "BFR_test": np.zeros((N_set, N_exp, N_train_max, ny)),
        "rmse_train": np.zeros((N_set, N_exp, N_train_max, ny)),
        "rmse_test": np.zeros((N_set, N_exp, N_train_max, ny)),
        "timings": np.empty((N_set, N_exp), dtype=object),
    }
    return samples, scores


def save_results(samples_sim, scores_sim, samples, scores, n_exp, n_set):
    """Copy one experiment's results into aggregate storage."""
    for key in (
        "U_train", "U_test", "Y_train", "Y_test", "Yhat_train", "Yhat_test"
    ):
        samples_sim[key][n_set, n_exp] = samples[key]
    for key in (
        "R2_train", "R2_test", "BFR_train", "BFR_test", "rmse_train", "rmse_test"
    ):
        scores_sim[key][n_set, n_exp] = scores[key]
    scores_sim["timings"][n_set, n_exp] = scores.get("timings")
    return samples_sim, scores_sim


def softplus(x, beta):
    return np.logaddexp(0, beta * x) / beta


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_gradient(A, beta, epsilon=0.01):
    norm_A = np.linalg.norm(A, ord=2)
    if norm_A == 0:
        raise ValueError("The norm of matrix A is zero, gradient is undefined.")
    z = norm_A - (1 - epsilon)
    return 2 * softplus(z, beta) * sigmoid(beta * z) * (A / norm_A)


def replace_nan_with_previous(arr):
    arr = np.asarray(arr)
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            arr[i] = arr[i - 1]
    return arr


def compute_constraint_violation(Y, y_min, y_max):
    lower_violation = np.maximum(y_min - Y, 0)
    upper_violation = np.maximum(Y - y_max, 0)
    violation = lower_violation + upper_violation
    return (
        np.mean(violation, axis=(2, 3)),
        np.sum(violation > 0, axis=(2, 3)),
    )
