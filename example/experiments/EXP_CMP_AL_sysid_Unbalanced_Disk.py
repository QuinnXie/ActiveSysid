"""Compare active-learning methods on the unbalanced-disk benchmark.

Run the standard ten-repetition benchmark with the default symmetric output
constraint with::

    python example/experiments/EXP_CMP_AL_sysid_Unbalanced_Disk.py \
        --exp-type cmp_al --save-exp-type bound4p5 --const 1 \
        --constraint-bound 4.5

Use ``--constraint-bound`` to set a positive symmetric output bound, and
``--exp-type cmp_delta`` or ``--exp-type cmp_alpha`` for IDW sweeps.
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

from example.systems.unbalanceddisk import (
    unbalanceddisk_output_fcn,
    unbalanceddisk_state_fcn,
)


SYSTEM_NAME = "unbalanced_disk"
MODEL_NAME = "RNN"
NX = 2
NX_MODEL = 2
NY = 1
NU = 1
TEMPORALITY = "continuous"
SAMPLE_TIME = 0.025
HISTORICAL_NOISE_BOUND = 4.5
MEASUREMENT_NOISE_STD = 0.01 * HISTORICAL_NOISE_BOUND
EKF_DEFAULTS = EKFDefaults(
    rho_x=1.0,
    rho_th=40.0,
    qy_cov=0.1,
    qth_cov=1e-8,
)
IDW_DEFAULTS = {
    False: IDWDefaults(delta=1e5, alpha=1e-3),
    True: IDWDefaults(delta=1e5, alpha=1.0),
}


class FX(nn.Module):
    """Latent-state transition network used by the RNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(features=8)(x))
        x = nn.tanh(nn.Dense(features=4)(x))
        return nn.Dense(features=NX_MODEL)(x)


class FY(nn.Module):
    """Output network without direct input feedthrough."""

    @nn.compact
    def __call__(self, xu):
        # The RNN appends the plant input to the latent state before calling FY.
        # Disk output depends only on the state, so ignore that final input.
        x = xu[:NX_MODEL]
        x = nn.tanh(nn.Dense(features=5)(x))
        return nn.Dense(features=NY)(x)


def build_parser():
    """Create the command-line parser for this example."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare active-learning methods on the unbalanced-disk benchmark."
        )
    )
    add_standard_example_arguments(
        parser,
        n_exp=10,
        n_train_init=60,
        n_train_max=2000,
        n_test=2000,
        score_interval=50,
        adam_epochs=1000,
        lbfgs_epochs=2000,
        exp_type="cmp_al",
        delta_set=None,
        alpha_set=None,
    )
    parser.add_argument(
        "--constraint-bound",
        type=float,
        default=4.5,
        help=(
            "Symmetric physical-output bound used when --const=1 "
            "(y in [-bound, bound])."
        ),
    )
    return parser


def build_system(args):
    """Construct the physical unbalanced-disk system."""
    if args.constraint_bound <= 0:
        raise ValueError("--constraint-bound must be positive")

    input_candidates = jnp.arange(-3.0, 3.0, 0.05).reshape(-1, NU)
    constraints = {
        "flag": args.const,
        "y_min": -args.constraint_bound,
        "y_max": args.constraint_bound,
    }
    process_noise_std = 0.0
    # Keep plant noise fixed when changing the output-constraint bound. The
    # historical +/-4.5 setup gives qy = 0.01 * 4.5 = 0.045.

    return System(
        NX,
        NY,
        NU,
        state_fcn=unbalanceddisk_state_fcn,
        output_fcn=unbalanceddisk_output_fcn,
        params={},
        x0=jnp.zeros((NX, 1)),
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
    model.loss(rho_x0=1e-3, rho_th=1e-3)
    model.optimization(
        adam_epochs=args.adam_epochs,
        lbfgs_epochs=args.lbfgs_epochs,
        iprint=-1,
    )
    return model


def run(args):
    """Build and run the configured unbalanced-disk comparison."""
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
        fixed_test_set=True,
    )


def main(argv=None):
    """Parse command-line arguments and run the example."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_runtime(args, run_dir=RUN_DIR, system_name=SYSTEM_NAME)
    return run(args)


if __name__ == "__main__":
    main()
