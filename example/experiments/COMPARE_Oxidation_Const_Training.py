"""Configure the shared constraint experiment and plotting for oxidation."""

from __future__ import annotations

import jax.numpy as jnp

from example.experiment_plotting import ConstraintPlotConfig
from example.experiments import constraint_training_driver as driver
from example.systems.oxidation import oxidation_output_fcn, oxidation_state_fcn


OXIDATION_PLOT_CONFIG = ConstraintPlotConfig(
    y_min=0.02,
    y_max=0.05,
    artifact_prefix="oxidation",
    system_plot_name="Oxidation",
    system_display_name="Ethylene oxidation",
    training_ylim=(0.013, 0.063),
    training_yticks=(0.02, 0.03, 0.04, 0.05, 0.06),
    multiseed_ylim=(0.013, 0.063),
    multiseed_yticks=(0.02, 0.03, 0.04, 0.05, 0.06),
    dense_training_plot=True,
    sample_time_seconds=5.0,
    smoothing_window=5,
)


def configure_oxidation() -> None:
    """Configure the shared constraint-study driver for oxidation."""
    driver.NX = 4
    driver.NY = 1
    driver.NU = 1
    driver.TS = 5.0
    driver.QX = 0.0
    driver.QY = 0.05 * (0.05 - 0.02)
    driver.Y_MIN = OXIDATION_PLOT_CONFIG.y_min
    driver.Y_MAX = OXIDATION_PLOT_CONFIG.y_max
    driver.STATE_FCN = oxidation_state_fcn
    driver.OUTPUT_FCN = oxidation_output_fcn
    driver.INITIAL_STATE = jnp.array([0.9981, 0.4291, 0.0303, 1.0019])
    driver.INPUT_START = 0.0704
    driver.INPUT_STOP = 0.7042
    driver.INPUT_STEP = 0.002

    # Match EXP_CMP_AL_sysid_Oxidation.py.
    driver.FX_HIDDEN = (8, 6)
    driver.FY_HIDDEN = 5
    driver.FY_STATE_ONLY = True
    driver.X_SCALING = 0.1
    driver.LOSS_RHO_X0 = 1e-4
    driver.LOSS_RHO_TH = 1e-4
    driver.ADAM_EPOCHS = 1000
    driver.LBFGS_EPOCHS = 2000

    driver.DEFAULT_N_TRAIN_INIT = 60
    driver.DEFAULT_N_TRAIN_MAX = 500
    driver.DEFAULT_N_TEST = 2000
    driver.IDW_DELTA = 1e4
    driver.IDW_ALPHA = 1e2
    driver.EKF_RHO_X = 0.5
    driver.EKF_RHO_TH = 0.5
    driver.EKF_QX_COV = 1e-10
    driver.EKF_QY_COV = 0.2 ** 2
    driver.EKF_QTH_COV = 1e-10


def main() -> None:
    configure_oxidation()
    driver.main(OXIDATION_PLOT_CONFIG)


if __name__ == "__main__":
    main()
