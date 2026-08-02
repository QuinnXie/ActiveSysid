"""System-independent driver for constant-constraint training experiments.

System entry points configure the module-level model and simulation settings,
then call :func:`main` with their own :class:`ConstraintPlotConfig`.
"""

import argparse
from copy import deepcopy
from functools import partial
import importlib
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax_sysid.models import RNN

from activesysid.system import System
import activesysid.acquisition.idw as idw_module
import activesysid.active_sysid as active_learning_sysid_module
from example.experiment_plotting import (
    ConstraintPlotConfig,
    plot_constraint_case_comparisons,
    plot_constraint_training_sets,
    plot_saved_constraint_case_comparison,
    plot_saved_constraint_multiseed_data,
    plot_saved_constraint_training_data,
    save_constraint_multiseed_comparison,
)
from activesysid.predict import predict

idw_module = importlib.reload(idw_module)
active_learning_sysid_module = importlib.reload(active_learning_sysid_module)
active_learning_sysid = active_learning_sysid_module.active_learning_sysid

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64: # type: ignore
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations
os.environ.setdefault("JAX_PLATFORMS", "cpu")



# These settings are supplied by a concrete system entry point before main().
NX: int
NY: int
NU: int
TS: float
QX: float
QY: float
Y_MIN: float
Y_MAX: float
STATE_FCN: object
OUTPUT_FCN: object
INITIAL_STATE: object
INPUT_START: float
INPUT_STOP: float
INPUT_STEP: float
FX_HIDDEN: tuple[int, ...]
FY_HIDDEN: int
FY_STATE_ONLY: bool
X_SCALING: float
LOSS_RHO_X0: float
LOSS_RHO_TH: float
ADAM_EPOCHS: int
LBFGS_EPOCHS: int
DEFAULT_N_TRAIN_INIT: int
DEFAULT_N_TRAIN_MAX: int
DEFAULT_N_TEST: int
IDW_DELTA: float
IDW_ALPHA: float
EKF_RHO_X: float
EKF_RHO_TH: float
EKF_QX_COV: float
EKF_QY_COV: float
EKF_QTH_COV: float


class FX(nn.Module):
    hidden_layers: tuple[int, ...]
    output_dim: int

    @nn.compact
    def __call__(self, x):
        for features in self.hidden_layers:
            x = nn.Dense(features=features)(x)
            x = nn.tanh(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x


class FY(nn.Module):
    hidden_features: int
    output_dim: int
    state_dim: int
    state_only: bool

    @nn.compact
    def __call__(self, x):
        if self.state_only:
            x = x[:self.state_dim]
        x = nn.Dense(features=self.hidden_features)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x


def build_model():
    configured_fx = partial(
        FX, hidden_layers=tuple(FX_HIDDEN), output_dim=NX,
    )
    configured_fy = partial(
        FY,
        hidden_features=FY_HIDDEN,
        output_dim=NY,
        state_dim=NX,
        state_only=FY_STATE_ONLY,
    )
    model = RNN(
        NX, NY, NU, FX=configured_fx, FY=configured_fy,
        x_scaling=X_SCALING,
    )
    model.loss(rho_x0=LOSS_RHO_X0, rho_th=LOSS_RHO_TH)
    model.optimization(adam_epochs=ADAM_EPOCHS, lbfgs_epochs=LBFGS_EPOCHS, iprint=-1)
    return model


def build_system(
    flag, u_set, confidence_alpha, uncertainty_beta, prediction_horizon,
    sequence_stride, first_step_buffer, first_step_weight,
    safety_filter, cbf_gamma, cbf_buffer, cbf_rho,
):
    effective_confidence_alpha = 0.99 if flag == 5 else confidence_alpha
    const = {
        "flag": flag,
        "y_min": Y_MIN,
        "y_max": Y_MAX,
    }
    if flag in (2, 5):
        const["confidence_alpha"] = effective_confidence_alpha
        const["uncertainty_beta"] = uncertainty_beta
    if flag == 4:
        const["confidence_alpha"] = effective_confidence_alpha
        const["uncertainty_beta"] = uncertainty_beta
        const["safety_filter"] = safety_filter
        const["cbf_gamma"] = cbf_gamma
        const["cbf_buffer"] = cbf_buffer
        const["cbf_rho"] = cbf_rho
        const["cbf_use_margin"] = True
        const["cbf_eps"] = 1e-12
    if flag == 3:
        const["prediction_horizon"] = prediction_horizon
        const["sequence_stride"] = sequence_stride
        const["first_step_buffer"] = first_step_buffer
        const["first_step_weight"] = first_step_weight
        const["safety_filter"] = safety_filter
        const["cbf_gamma"] = cbf_gamma
        const["cbf_buffer"] = cbf_buffer
        const["cbf_rho"] = cbf_rho
        const["cbf_use_margin"] = False

    return System(
        NX,
        NU,
        NY,
        state_fcn=STATE_FCN,
        output_fcn=OUTPUT_FCN,
        params={},
        x0=INITIAL_STATE,
        u_set=u_set,
        Ts=TS,
        qx=QX,
        qy=QY,
        const=const,
        temporality="continuous",
    )


def generate_shared_data(seed, u_set, n_train_init, n_test):
    rng = np.random.RandomState(seed)
    init_set = {}
    init_set["U"] = rng.choice(np.asarray(u_set).reshape(-1), n_train_init).reshape(-1, 1)
    init_set["Y"], init_set["X"] = predict(
        INITIAL_STATE,
        init_set["U"],
        STATE_FCN,
        OUTPUT_FCN,
        {},
        qx=QX,
        qy=QY,
        return_X=True,
        temporality="continuous",
        Ts=TS,
    )

    test_set = {}
    test_set["U"] = rng.choice(np.asarray(u_set).reshape(-1), n_test).reshape(-1, 1)
    test_set["Y"] = predict(
        INITIAL_STATE,
        test_set["U"],
        STATE_FCN,
        OUTPUT_FCN,
        {},
        qx=QX,
        qy=QY,
        temporality="continuous",
        Ts=TS,
    )
    return init_set, test_set


def violation_stats(y, n_train_init):
    y_eval = np.asarray(y[n_train_init:, 0], dtype=float)
    finite = np.isfinite(y_eval)
    y_valid = y_eval[finite]
    lower = np.maximum(Y_MIN - y_valid, 0.0)
    upper = np.maximum(y_valid - Y_MAX, 0.0)
    violation = lower + upper
    return {
        "count": int(np.sum(violation > 0.0)),
        "mean": float(np.mean(violation)) if violation.size else np.nan,
        "max": float(np.max(violation)) if violation.size else np.nan,
        "valid_samples": int(finite.sum()),
        "nonfinite_samples": int((~finite).sum()),
        "failed": bool(np.any(~finite)),
    }


def print_margin_report(results):
    margin_flag = 5 if 5 in results else 2
    if "confidence_margin" not in results[margin_flag]["scores"]:
        print(f"\nNo confidence_margin values found for flag={margin_flag}.")
        return

    raw_margin = np.asarray(
        results[margin_flag]["scores"]["confidence_margin"], dtype=float
    )
    selected_margin = raw_margin[:, 0]
    selected_steps = np.flatnonzero(np.isfinite(selected_margin))

    output_margin = np.full_like(raw_margin, np.nan)
    output_margin[1:] = raw_margin[:-1]
    output_steps = np.flatnonzero(np.any(np.isfinite(output_margin), axis=1))

    if selected_steps.size == 0:
        print("\nActual confidence margin: no finite values recorded.")
        return

    finite_selected = selected_margin[selected_steps]
    print(f"\nActual confidence margin for flag={margin_flag}, unscaled output units")
    print("-" * 72)
    print(
        "selected-input margin: "
        f"count={selected_steps.size}, "
        f"min={np.min(finite_selected):.8g}, "
        f"mean={np.mean(finite_selected):.8g}, "
        f"max={np.max(finite_selected):.8g}"
    )
    print(
        "The plotted bounds use the previous selected margin: "
        "lower = y_min + margin, upper = y_max - margin."
    )
    print("\nstep, selected_margin, plotted_output_margin")
    all_steps = np.union1d(selected_steps, output_steps)
    for step in all_steps:
        selected_value = selected_margin[step]
        output_value = output_margin[step, 0]
        selected_text = f"{selected_value:.10g}" if np.isfinite(selected_value) else "nan"
        output_text = f"{output_value:.10g}" if np.isfinite(output_value) else "nan"
        print(f"{step:d}, {selected_text}, {output_text}")


def save_runtime_report(results, output_path, system_display_name):
    """Write steady-state IDW runtimes with JAX warm-up excluded."""
    labels = {
        0: "without penalty",
        1: "with penalty p",
        4: "with penalty p^idw + p^cbf",
        5: "with empirical residual margin",
    }
    lines = [
        f"{system_display_name} constant-constraint IDW runtime",
        "=" * 72,
        "The first IDW query of each case is excluded as JAX compilation warm-up.",
        "",
    ]
    total_measured_seconds = 0.0
    total_queries = 0
    for flag, data in results.items():
        timing = data["scores"]["timings"]["active_learning"]
        measured_seconds = timing["mean"] * timing["n"]
        total_measured_seconds += measured_seconds
        total_queries += timing["n"]
        lines.extend([
            f"flag={flag} ({labels[flag]})",
            f"  measured queries: {timing['n']}",
            f"  excluded warm-up queries: {timing['warmup_samples']}",
            f"  average IDW time: {timing['mean']:.8f} s/query",
            f"  median IDW time: {timing['median']:.8f} s/query",
            f"  measured IDW total: {measured_seconds:.6f} s",
            "",
        ])
    overall_average = (
        total_measured_seconds / total_queries if total_queries else np.nan
    )
    lines.extend([
        "-" * 72,
        f"Overall measured queries: {total_queries}",
        f"Overall average IDW time: {overall_average:.8f} s/query",
        f"Total measured IDW time: {total_measured_seconds:.6f} s",
    ])
    output_path.with_suffix(".txt").write_text("\n".join(lines) + "\n")


def run_seed(args, seed, flags=(0, 1, 4, 5)):
    """Run selected comparison cases for one random seed."""
    np.random.seed(seed)
    u_set = jnp.arange(INPUT_START, INPUT_STOP, INPUT_STEP).reshape(-1, 1)
    init_set, test_set = generate_shared_data(
        seed, u_set, args.n_train_init, args.n_test
    )
    common = dict(
        u_set=u_set,
        N_train_init=args.n_train_init,
        N_train_max=args.n_train_max,
        init_set=init_set,
        test_set=test_set,
        delta=IDW_DELTA,
        alpha=IDW_ALPHA,
        qx=QX,
        qy=QY,
        AL_method="idw",
        verbose=1,
        pred="EKF_step",
        isScale=1,
        init_set_trainning="jax-sysid",
        Ts=TS,
        rho_x=EKF_RHO_X,
        rho_th=EKF_RHO_TH,
        Qx_cov=EKF_QX_COV,
        Qy_cov=EKF_QY_COV,
        Qth_cov=EKF_QTH_COV,
        temporality="continuous",
        seed=seed,
        score_interval=args.score_interval,
        return_init_cache=True,
        initial_ekf_epochs=1,
    )
    results = {}
    shared_init_cache = None
    for flag in flags:
        print("\n" + "=" * 72)
        print(f"Running seed={seed}, const.flag={flag}")
        print("=" * 72)
        system = build_system(
            flag, u_set, args.confidence_alpha, args.uncertainty_beta,
            args.prediction_horizon, args.sequence_stride,
            args.first_step_buffer, args.first_step_weight,
            not args.disable_safety_filter, args.cbf_gamma,
            args.cbf_buffer, args.cbf_rho,
        )
        model = build_model()
        model, samples, scores, shared_init_cache = active_learning_sysid(
            system, model, isConst=flag, init_cache=shared_init_cache, **common
        )
        results[flag] = {
            "system": system,
            "model": model,
            "samples": deepcopy(samples),
            "scores": deepcopy(scores),
            "violations": violation_stats(samples["Y_train"], args.n_train_init),
        }
        timing = scores["timings"]["active_learning"]
        print(
            f"Average IDW time: {timing['mean']:.8f} s/query "
            f"({timing['warmup_samples']} warm-up query excluded)"
        )
    return results


def main(plot_config: ConstraintPlotConfig):
    """Run the shared experiment using an explicit system plotting config."""
    parser = argparse.ArgumentParser(
        description=(
            f"Run the {plot_config.system_display_name} IDW comparison for no penalty, deterministic "
            "penalty, IDW-margin plus CBF penalty, and empirical margin."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help=(
            "Run no-penalty, nominal-penalty, and IDW-CBF cases across multiple seeds, "
            "e.g. --seeds 0 1 2 4 5."
        ),
    )
    parser.add_argument(
        "--flags",
        type=int,
        nargs="+",
        choices=(0, 1, 4),
        default=(0, 1, 4),
        help="Constraint cases to run with --seeds; use --flags 4 for IDW--CBF only.",
    )
    parser.add_argument("--n-train-init", type=int, default=DEFAULT_N_TRAIN_INIT)
    parser.add_argument("--n-train-max", type=int, default=DEFAULT_N_TRAIN_MAX)
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    parser.add_argument("--score-interval", type=int, default=10)
    parser.add_argument(
        "--confidence-alpha",
        type=float,
        default=0.9,
        help=(
            "Sample quantile for kappa_alpha. 0.8 matches the MATLAB "
            "descending-sort length/5 choice; use 0.9 for the paper's 90%%."
        ),
    )
    parser.add_argument(
        "--uncertainty-beta",
        type=float,
        default=1.0 / 3.0,
        help="Cap for the confidence margin as beta * (y_max - y_min).",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=3,
        help="Prediction horizon h used by deterministic sequence penalty flag=3.",
    )
    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=10,
        help=(
            "Use every N-th input-pool value when enumerating flag=3 input "
            "sequences. Increase this for large input pools."
        ),
    )
    parser.add_argument(
        "--first-step-buffer",
        type=float,
        default=0.003,
        help=(
            "Deterministic buffer for flag=3 first-step predicted output. "
            "Candidates with y_{k+1|k} outside [y_min+buffer, y_max-buffer] "
            "receive a huge negative score."
        ),
    )
    parser.add_argument(
        "--first-step-weight",
        type=float,
        default=10.0,
        help="Extra soft-penalty weight for the first predicted step in flag=3.",
    )
    parser.add_argument(
        "--disable-safety-filter",
        action="store_true",
        help="Disable the lightweight CBF-style safety filter for constrained IDW.",
    )
    parser.add_argument(
        "--cbf-gamma",
        type=float,
        default=1.0,
        help=(
            "Discrete CBF relaxation in [0, 1]. 1.0 filters only predicted "
            "one-step bound violations; smaller values are more conservative."
        ),
    )
    parser.add_argument(
        "--cbf-buffer",
        type=float,
        default=0.0,
        help="Extra physical-unit buffer added inside the CBF safe output bounds.",
    )
    parser.add_argument(
        "--cbf-rho",
        type=float,
        default=1e30,
        help="Penalty multiplier for CBF filter violations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "example/experiments/artifacts/figures/"
            f"{plot_config.artifact_prefix}/"
            f"{plot_config.artifact_prefix}_cmp_const.pdf"
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the PDF from the saved pickle data without training.",
    )
    parser.add_argument(
        "--multiseed-plot-only",
        action="store_true",
        help="Regenerate the multi-seed PDF from its pickle without training.",
    )
    args = parser.parse_args()

    if args.multiseed_plot_only:
        data_path = args.output.with_name(
            f"{plot_config.artifact_prefix}_cmp_const_multiseed.pkl"
        )
        if not data_path.exists():
            parser.error(f"saved multi-seed data not found: {data_path}")
        output_path = data_path.with_suffix(".pdf")
        plot_saved_constraint_multiseed_data(
            data_path, output_path, plot_config,
        )
        print(f"Regenerated multi-seed plot from: {data_path.resolve()}")
        print(f"Saved multi-seed plot to: {output_path.resolve()}")
        return

    if args.plot_only:
        data_path = args.output.with_name(
            f"{plot_config.artifact_prefix}_cmp_const.pkl"
        )
        if not data_path.exists():
            parser.error(f"saved plotting data not found: {data_path}")
        comparison_output = plot_saved_constraint_training_data(
            data_path, args.output, plot_config,
        )
        print(f"Regenerated comparison plot from: {data_path.resolve()}")
        print(f"Saved comparison plot to: {comparison_output.resolve()}")
        for flag in (1, 4, 5):
            case_data_path = args.output.with_name(
                f"{plot_config.artifact_prefix}_cmp_const_"
                f"flag{flag}_vs_flag0.pkl"
            )
            if not case_data_path.exists():
                continue
            case_output = plot_saved_constraint_case_comparison(
                case_data_path, case_data_path.with_suffix(".pdf"), plot_config,
            )
            print(
                f"Regenerated flag {flag} vs flag 0 plot from: "
                f"{case_data_path.resolve()}"
            )
            print(f"Saved pairwise plot to: {case_output.resolve()}")
        return

    if args.seeds:
        seeds = list(dict.fromkeys(args.seeds))
        flags = tuple(dict.fromkeys(args.flags))
        all_results = {
            seed: run_seed(args, seed, flags=flags) for seed in seeds
        }
        output_path = save_constraint_multiseed_comparison(
            all_results,
            seeds,
            args.n_train_init,
            args.output.parent,
            plot_config,
            flags=flags,
        )
        print(f"\nSaved multi-seed plot to: {output_path.resolve()}")
        print(f"Saved multi-seed data to: {output_path.with_suffix('.pkl').resolve()}")
        print(f"Saved multi-seed report to: {output_path.with_suffix('.txt').resolve()}")
        return

    results = run_seed(args, args.seed)

    comparison_output = plot_constraint_training_sets(
        results, args.n_train_init, args.output, plot_config,
    )
    case_outputs = plot_constraint_case_comparisons(
        results, args.n_train_init, args.output, plot_config,
    )
    save_runtime_report(
        results, comparison_output, plot_config.system_display_name,
    )

    print("\nConstraint violation summary after the initial set")
    print("-" * 72)
    for flag, data in results.items():
        stats = data["violations"]
        print(
            f"flag={flag}: count={stats['count']}, "
            f"mean={stats['mean']:.6g}, max={stats['max']:.6g}"
        )
    # print_margin_report(results)
    print(f"\nSaved comparison plot to: {comparison_output.resolve()}")
    for flag, case_output in case_outputs.items():
        print(
            f"Saved flag {flag} vs flag 0 plot to: {case_output.resolve()}"
        )
    print(f"Saved reusable plot data to: {comparison_output.with_suffix('.pkl').resolve()}")
    print(f"Saved runtime report to: {comparison_output.with_suffix('.txt').resolve()}")
