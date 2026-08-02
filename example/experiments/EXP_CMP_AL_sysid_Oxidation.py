"""Compare active-learning methods on the ethylene-oxidation benchmark.

Run the standard ten-repetition benchmark without the output-constraint
penalty with::

    python example/experiments/EXP_CMP_AL_sysid_Oxidation.py \
        --exp-type cmp_al --save-exp-type no_penalty --const 0

Use ``--help`` to list acquisition, EKF, saving, and plotting options.
"""

from __future__ import annotations

try:
    from ._common_imports import (
        EKFDefaults,
        IDWDefaults,
        RUN_DIR,
        RNN,
        System,
        add_standard_example_arguments,
        argparse,
        jnp,
        nn,
        run_standard_example,
        setup_runtime,
    )
except ImportError:  # Support direct execution from the repository root.
    from _common_imports import (
        EKFDefaults,
        IDWDefaults,
        RUN_DIR,
        RNN,
        System,
        add_standard_example_arguments,
        argparse,
        jnp,
        nn,
        run_standard_example,
        setup_runtime,
    )

from example.systems.oxidation import (
    oxidation_output_fcn,
    oxidation_state_fcn,
)


SYSTEM_NAME = "oxidation"
MODEL_NAME = "RNN"
NX = 4
NX_MODEL = 4
NY = 1
NU = 1
TEMPORALITY = "continuous"
SAMPLE_TIME = 5.0
OUTPUT_MIN = 0.02
OUTPUT_MAX = 0.05
MEASUREMENT_NOISE_STD = 0.05 * (OUTPUT_MAX - OUTPUT_MIN)
EKF_DEFAULTS = EKFDefaults(
    rho_x=0.5,
    rho_th=0.5,
    qy_cov=0.2**2,
    qth_cov=1e-10,
)
IDW_DEFAULTS = {
    False: IDWDefaults(delta=1e4, alpha=1e2),
    True: IDWDefaults(delta=3e4, alpha=3e2),
}


class FX(nn.Module):
    """Latent-state transition network used by the RNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(features=8)(x))
        x = nn.tanh(nn.Dense(features=6)(x))
        return nn.Dense(features=NX_MODEL)(x)


class FY(nn.Module):
    """Output network without direct input feedthrough."""

    @nn.compact
    def __call__(self, xu):
        # The RNN appends the plant input to the latent state before calling FY.
        # Oxidation output depends only on the state, so ignore that final input.
        x = xu[:NX_MODEL]
        x = nn.tanh(nn.Dense(features=5)(x))
        return nn.Dense(features=NY)(x)


def build_parser():
    """Create the command-line parser for this example."""
    parser = argparse.ArgumentParser(
        description="Compare active-learning methods on the oxidation benchmark."
    )
    add_standard_example_arguments(
        parser,
        n_exp=10,
        n_train_init=60,
        n_train_max=500,
        n_test=2000,
        score_interval=10,
        adam_epochs=1000,
        lbfgs_epochs=2000,
        exp_type="cmp_al",
        delta_set=None,
        alpha_set=None,
    )
    return parser


def build_system(args):
    """Construct the physical oxidation system."""
    input_candidates = jnp.arange(0.0704, 0.7042, 0.002).reshape(-1, NU)
    constraints = {
        "flag": args.const,
        "y_min": OUTPUT_MIN,
        "y_max": OUTPUT_MAX,
    }
    process_noise_std = 0.0

    return System(
        NX,
        NY,
        NU,
        state_fcn=oxidation_state_fcn,
        output_fcn=oxidation_output_fcn,
        params={},
        x0=jnp.array([0.9981, 0.4291, 0.0303, 1.0019]),
        u_set=input_candidates,
        Ts=SAMPLE_TIME,
        qx=process_noise_std,
        qy=MEASUREMENT_NOISE_STD,
        const=constraints,
        temporality=TEMPORALITY,
    )


def build_model(args):
    """Construct and configure the learned RNN surrogate."""
    model = RNN(
        NX_MODEL,
        NY,
        NU,
        FX=FX,
        FY=FY,
        x_scaling=0.1,
    )
    model.loss(rho_x0=1e-4, rho_th=1e-4)
    model.optimization(
        adam_epochs=args.adam_epochs,
        lbfgs_epochs=args.lbfgs_epochs,
        iprint=-1,
    )
    return model


def run(args):
    """Build and run the configured oxidation comparison."""
    system = build_system(args)
    model = build_model(args)

    return run_standard_example(
        args=args,
        system=system,
        model=model,
        system_name=SYSTEM_NAME,
        model_name=MODEL_NAME,
        ekf_defaults=EKF_DEFAULTS,
        idw_defaults=IDW_DEFAULTS[bool(args.const)],
    )


def main(argv=None):
    """Parse command-line arguments and run the example."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_runtime(args, run_dir=RUN_DIR, system_name=SYSTEM_NAME)
    return run(args)


if __name__ == "__main__":
    main()
