"""Timing helpers that synchronize asynchronous JAX computations."""

import time

import jax
import numpy as np


def block_until_ready(value):
    """Wait for every JAX array leaf in a returned value."""
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def time_call(func):
    """Execute ``func`` and return its synchronized value and elapsed time."""
    start = time.perf_counter()
    value = func()
    block_until_ready(value)
    return value, time.perf_counter() - start


def timing_stats(times, warmup=1):
    """Summarize timing samples after optionally excluding warm-up calls."""
    samples = np.asarray(times, dtype=float)
    requested_warmup = min(int(warmup), samples.size)
    excluded_warmup = (
        requested_warmup if samples.size > requested_warmup else 0
    )
    measured = samples[excluded_warmup:]

    if measured.size == 0:
        return {
            "samples": samples,
            "warmup_samples": excluded_warmup,
            "n": 0,
            "min": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "max": np.nan,
            "std": np.nan,
        }

    return {
        "samples": samples,
        "warmup_samples": excluded_warmup,
        "n": measured.size,
        "min": float(np.min(measured)),
        "mean": float(np.mean(measured)),
        "median": float(np.median(measured)),
        "p25": float(np.percentile(measured, 25)),
        "p75": float(np.percentile(measured, 75)),
        "max": float(np.max(measured)),
        "std": (
            float(np.std(measured, ddof=1))
            if measured.size > 1 else 0.0
        ),
    }
