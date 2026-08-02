"""Common imports for directly executed experiment scripts."""

from pathlib import Path
import sys

RUN_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = RUN_DIR / "src"
for path in (RUN_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from jax_sysid.models import RNN
from example.experiment_runner import (
    EKFDefaults,
    IDWDefaults,
    add_common_arguments,
    add_standard_example_arguments,
    run_common_experiment,
    run_standard_example,
    selected_al_methods,
    setup_runtime,
)
from activesysid.system import System

import argparse
import numpy as np
from flax import linen as nn
import jax.numpy as jnp

__all__ = [
    "RUN_DIR",
    "EKFDefaults",
    "IDWDefaults",
    "RNN",
    "System",
    "add_common_arguments",
    "add_standard_example_arguments",
    "argparse",
    "jnp",
    "nn",
    "np",
    "run_common_experiment",
    "run_standard_example",
    "selected_al_methods",
    "setup_runtime",
]
