<img src="https://raw.githubusercontent.com/QuinnXie/ActiveSysID/main/docs/assets/activesysid-logo.svg" alt="ActiveSysID" width="40%"/>

# ActiveSysID - Active Learning for System Identification

`activesysid` is a JAX-based research package for learning nonlinear dynamical models while actively selecting the system inputs used to collect training data. It combines `jax-sysid` models with online state and parameter estimation, several acquisition strategies, optional output constraints, and reproducible benchmark experiments.

The package provides:

- passive, GSx, iGS, and IDW input selection;
- online extended Kalman filter (EKF) state and parameter updates;
- initial-state estimation with EKF filtering and RTS smoothing;
- discrete- and continuous-time trajectory simulation;
- optional input/output scaling and constrained acquisition;
- R², best-fit-rate (BFR), RMSE, and timing;
- oxidation and unbalanced-disk examples.

The physical system and learned `jax-sysid` model remain user-defined. For the methods and experimental results, see the [paper](https://doi.org/10.48550/arXiv.2506.21754).

## Installation

Python 3.10 or newer is required.

### Install a released version

Install the latest tagged release directly from GitHub:

```bash
python -m pip install \
  "git+https://github.com/QuinnXie/ActiveSysID.git@v0.1.0"
```

Verify the installed package and version:

```bash
python -c "import importlib.metadata as m; import activesysid; print(m.version('activesysid'))"
```

### Development installation

To modify the source code or run the repository experiments, clone the
repository and install the package in editable mode:

```bash
git clone https://github.com/QuinnXie/ActiveSysID.git
cd ActiveSysID
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
```

`requirements.txt` is the exact environment used by the reference
implementation. The final editable install makes changes under
`src/activesysid` immediately available without reinstalling the package.

Verify that Python imports this checkout rather than a previously installed
release:

```bash
.venv/bin/python -c "import activesysid; print(activesysid.__file__)"
```

The printed path must end in `ActiveSysID/src/activesysid/__init__.py`.

## Overview

The physical system is represented by

$$
\begin{equation}
    \mathcal{P}: 
    \begin{cases}
    \begin{aligned}
        \xi_{k+1} &= f_{\mathcal{P}}(\xi_k,u_k),\\
        y_k &= g_{\mathcal{P}}(\xi_k) + \eta_k,
    \end{aligned}
    \end{cases}
\end{equation}
$$

and the learned model by

$$
\begin{aligned}
    x_{k+1} &= f_x(x_k,u_k,\theta_x),\\
    \hat y_k &= f_y(x_k,\theta_y),
\end{aligned}
$$

Starting from an initial dataset, active system identification repeatedly:

1. assimilates the newest measurement;
2. updates the estimated state and model parameters;
3. scores candidate inputs with the selected acquisition method;
4. applies the selected input to the physical system; and
5. records prediction accuracy, constraints, and runtime statistics.

The online EKF workflow assumes no direct input-to-output feedthrough: $\hat y_k = f_y(x_k,\theta_y)$. The output callback still accepts `(x, u, params)` for a common interface, but models used with the online workflow should not use `u` directly in their output equation.

## Define a System

Use `System` to collect the physical dynamics, dimensions, simulation settings, candidate inputs, noise levels, and optional constraints. The ethylene-oxidation benchmark provides a representative nonlinear continuous-time example:

```python
import jax.numpy as jnp

from activesysid import System
from example.systems.oxidation import (
    oxidation_output_fcn,
    oxidation_state_fcn,
)

nx = 4
nu = 1
ny = 1

x0 = jnp.array([0.9981, 0.4291, 0.0303, 1.0019])
u_set = jnp.arange(0.0704, 0.7042, 0.002).reshape(-1, nu)

constraints = {
    "flag": 1,
    "y_min": 0.02,
    "y_max": 0.05,
}

measurement_noise_fraction = 0.05
qx = 0.0
qy = measurement_noise_fraction * (
    constraints["y_max"] - constraints["y_min"]
)


system = System(
    nx=nx,
    nu=nu,
    ny=ny,
    state_fcn=oxidation_state_fcn,
    output_fcn=oxidation_output_fcn,
    params={},
    x0=x0,
    u_set=u_set,
    Ts=5.0,
    qx=qx,
    qy=qy,
    const=constraints,
    temporality="continuous",
)
```

Here, `oxidation_state_fcn(x, t, u)` is the four-state nonlinear continuous-time plant model and `oxidation_output_fcn(x, u, params)` returns the third state, the measured product concentration. The active learner chooses one input from `u_set` at every sampling instant. Because `temporality="continuous"`, the state equation is numerically integrated over the sampling period `Ts=5.0`.

Simulation noise is additive by default: `qx` and `qy` are standard deviations. This example has no simulated process noise and sets the measurement-noise standard deviation to 5% of the allowed output range. With `noise_mode="multiplicative"`, `qx` and `qy` are instead relative standard-deviation factors.

## Run Active System Identification

Create a compatible `jax-sysid` model, configure its loss and optimizer, and pass it to `active_learning_sysid`:

```python
from flax import linen as nn
from jax_sysid.models import RNN

from activesysid.active_sysid import active_learning_sysid


class StateNetwork(nn.Module):
    @nn.compact
    def __call__(self, xu):
        xu = nn.tanh(nn.Dense(8)(xu))
        xu = nn.tanh(nn.Dense(6)(xu))
        return nn.Dense(nx)(xu)


class OutputNetwork(nn.Module):
    @nn.compact
    def __call__(self, xu):
        # The oxidation measurement has no direct input feedthrough.
        x = xu[:nx]
        x = nn.tanh(nn.Dense(5)(x))
        return nn.Dense(ny)(x)


model = RNN(
    nx=nx,
    ny=ny,
    nu=nu,
    FX=StateNetwork,
    FY=OutputNetwork,
    x_scaling=0.1,
)
model.loss(rho_x0=1e-4, rho_th=1e-4)
model.optimization(adam_epochs=1000, lbfgs_epochs=2000, iprint=-1)

model, samples, scores = active_learning_sysid(
    system=system,
    model=model,
    u_set=system.u_set,
    N_train_init=60,
    N_train_max=500,
    AL_method="idw",
    delta=3e4,
    alpha=3e2,
    pred="EKF_step",
    isScale=True,
    isConst=True,
    qx=system.qx,
    qy=system.qy,
    Ts=system.Ts,
    temporality=system.temporality,
    seed=3,
)
```

The RNN learns a discrete-time surrogate of the sampled oxidation process. Its state network predicts the next latent state from the current latent state and plant input. Its output network deliberately ignores the appended input because the oxidation measurement depends only on the state. The initial 60 samples fit the model; IDW then chooses the remaining inputs while the `EKF_step` path updates the latent state and model parameters online.

The `constraints` dictionary attached to `System` defines the permitted output
range. Setting its `flag` and passing `isConst=True` enables the output penalty
during acquisition. Here the constrained Oxidation tuning uses
`delta=3e4` and `alpha=3e2`; the unconstrained tuning is `delta=1e4` and
`alpha=1e2`.

The main options are:

- `AL_method`: `passive`, `GSx`, `iGS`, or `IDW` (method names are case-insensitive).
- `pred`: `EKF_step` for online joint state/parameter updates, `EKF_set` for repeated full-dataset EKF updates, or `jax-sysid` for repeated model fits.
- `N_train_init`, `N_train_max`: initial and final training-set sizes.
- `delta`, `alpha`: IDW acquisition weights.
- `train_interval`: retraining interval for fitting-based workflows.
- `isScale`: perform identification in scaled input/output coordinates.
- `rho_x`, `rho_th`: inverse-covariance scales used to initialize the EKF state and parameter blocks when `P` is not supplied.
- `Qx_cov`, `Qy_cov`, `Qth_cov`: EKF process, measurement, and parameter covariance values. When scaling is enabled these use scaled coordinates.
- `initial_ekf_epochs`: EKF replay passes over the initial dataset.
- `score_interval`: interval between full prediction evaluations.
- `history_refresh_interval`: number of online samples between full latent state-history reconstructions.

`samples` contains the measured and predicted training/test trajectories and their inputs. `scores` contains R², BFR, RMSE, optional confidence margins, stage timings, and failure metadata.

## Acquisition Methods

The available methods trade off input-space coverage, predicted behavior, and model uncertainty:

- **Passive** randomly samples the candidate input set.
- **GSx** selects inputs whose predicted states are far from the acquired state history.
- **iGS** combines state-space and output-space diversity.
- **IDW** combines inverse-distance exploration, prediction behavior, and configurable uncertainty weighting.

Candidate inputs must have shape `(number_of_candidates, nu)`. Distance
components can be reweighted through `distance_weights`, and `distance_eps`
controls numerical regularization.

## Examples

Run the benchmark entry points from the repository root:

```bash
python example/experiments/EXP_CMP_AL_sysid_Oxidation.py
python example/experiments/EXP_CMP_AL_sysid_Unbalanced_Disk.py
```

Use `--help` for available options. Published comparison commands, tuned
weights, sensitivity grids, and safe output labels are documented in the
[benchmark reproduction guide](https://github.com/QuinnXie/ActiveSysID/blob/main/docs/benchmark_reproduction.md).

## Save and Load Results

Pickle helpers are available in `activesysid.data_save_load`:

```python
from activesysid.data_save_load.pickle_io import load_data_pkl, save_data_pkl
```

The experiment workflow manages naming, loading, saving, plotting, and timing summaries. Default data paths resolve to `example/experiments/artifacts/data`.

## Project Layout

```text
active_sysid_python/
├── src/activesysid/     # reusable identification library
│   ├── acquisition/     # active input-selection methods
│   └── data_save_load/  # Pickle persistence
├── example/
│   ├── systems/         # benchmark physical systems
│   ├── experiments/     # runnable examples and comparison scripts
│   └── analysis/        # plotting and saved-result analysis
└── docs/                # setup notes and project assets
```

## References

> [1] K. Xie and A. Bemporad, "[Online design of experiments by active learning for nonlinear system identification](https://doi.org/10.48550/arXiv.2506.21754)," arXiv:2506.21754, 2025. Code available at [https://github.com/QuinnXie/ActiveSysID.git](https://github.com/QuinnXie/ActiveSysID.git).

> [2] A. Bemporad, "[An L-BFGS-B approach for linear and nonlinear system identification under $\ell_1$ and group-Lasso regularization](https://doi.org/10.1109/TAC.2024.3406595)," *IEEE Transactions on Automatic Control*, vol. 70, no. 7, pp. 4857–4864, 2025. (**jax-sysid**)

## Citation

If you use this package in your research, please cite:

```bibtex
@misc{XB26,
  author = {K. Xie and A. Bemporad},
  title = {Online design of experiments by active learning for nonlinear system identification},
  howpublished = {arXiv:2506.21754},
  note = {Code available at \url{https://github.com/QuinnXie/ActiveSysID.git}},
  year = {2025}
}
```

## Copyright

(C) 2025 K. Xie

## Acknowledgement

This work was funded by the European Union (ERC Advanced Research Grant COMPACT, No. 101141351). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

<p align="center">
<img src="https://raw.githubusercontent.com/QuinnXie/ActiveSysID/main/docs/assets/erc-logo.png" alt="ERC" width="400"/>
</p>
