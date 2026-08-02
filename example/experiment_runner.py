"""Common command-line and reporting utilities for experiment scripts."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import jax # type: ignore
import numpy as np # type: ignore

from example.experiment_utils import score_checkpoints
from example.experiment_workflow import (
    plot_ekf_diagnostics,
    run_or_load_experiment,
    summarize_ekf_diagnostics,
)
from example.experiment_plotting import score_series_labels


ALLOWED_AL_METHODS = ("passive", "IDWuy", "GSx", "iGS", "IDW")
AL_METHOD_SETS = {
    1: ("passive",),
    2: ("IDW",),
    3: ("passive", "IDW"),
    4: ("passive", "GSx", "iGS", "IDW"),
    5: ("passive", "IDWuy", "GSx", "iGS", "IDW"),
}


def float_list(value):
    """Parse comma-separated float lists for command-line sweeps."""
    return [float(item) for item in value.split(",") if item.strip()]


def validate_al_methods(methods):
    """Validate method names and return their canonical spellings."""
    requested = [str(method).strip() for method in methods if str(method).strip()]
    if not requested:
        raise ValueError("at least one acquisition method is required")

    canonical_names = {
        method.lower(): method for method in ALLOWED_AL_METHODS
    }
    invalid = [
        method for method in requested if method.lower() not in canonical_names
    ]
    if invalid:
        allowed = ", ".join(ALLOWED_AL_METHODS)
        raise ValueError(
            f"unsupported acquisition method(s): {', '.join(invalid)}; "
            f"choose from {allowed}"
        )
    return [canonical_names[method.lower()] for method in requested]


def al_method_list(value):
    """Parse and validate a comma-separated acquisition-method list."""
    try:
        return validate_al_methods(value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def add_common_arguments(
    parser,
    *,
    const_default=0,
    n_exp=10,
    n_train_init=60,
    n_train_max=500,
    n_test=2000,
    score_interval=10,
    adam_epochs=1000,
    lbfgs_epochs=2000,
    exp_type="cmp_al",
    exp_type_choices=None,
    delta_set=(1e3,),
    alpha_set=(1e1,),
    noise_mode="additive",
    diagnostic_plot=False,
):
    """Add CLI options shared by the system-identification experiments."""
    parser.add_argument("--const", type=int, choices=(0, 1), default=const_default, help="Apply output constraints: 1=yes, 0=no.")
    parser.add_argument(
        "--method-set",
        type=int,
        choices=tuple(AL_METHOD_SETS),
        default=4,
        help=(
            "Predefined acquisition-method set: 1=passive, "
            "2=IDW, 3=passive+IDW, "
            "4=passive+GSx+iGS+IDW, 5=all five methods."
        ),
    )
    parser.add_argument(
        "--methods",
        "--method",
        type=al_method_list,
        default=None,
        metavar="METHOD[,METHOD...]",
        help=(
            "Custom acquisition methods, overriding --method-set. "
            "Choices: passive, IDWuy, GSx, iGS, IDW."
        ),
    )
    parser.add_argument("--save-exp-type", default=None, help="Optional experiment label used for saved data/figures without changing the simulation mode.")
    parser.add_argument("--n-exp", type=int, default=n_exp, help="Number of experiment repetitions.")
    parser.add_argument("--n-train-init", type=int, default=n_train_init, help="Initial training samples.")
    parser.add_argument("--n-train-max", type=int, default=n_train_max, help="Maximum training samples.")
    parser.add_argument("--n-test", type=int, default=n_test, help="Test samples.")
    parser.add_argument("--score-interval", type=int, default=score_interval, help="R2/RMSE scoring interval.")
    parser.add_argument("--adam-epochs", type=int, default=adam_epochs, help="Initial Adam epochs.")
    parser.add_argument("--lbfgs-epochs", type=int, default=lbfgs_epochs, help="Initial L-BFGS-B epochs.")
    if exp_type_choices is None:
        exp_type_choices = (
            "cmp_al",
            "cmp_ekf",
            "cmp_delta",
            "cmp_delta_idwuy",
            "cmp_alpha",
            "cmp_idw_grid",
        )
    parser.add_argument(
        "--exp-type",
        choices=exp_type_choices,
        default=exp_type,
        help=(
            "Comparison mode. cmp_al compares acquisition methods; "
            "cmp_delta and cmp_alpha scan IDW weights; cmp_ekf scans EKF "
            "settings; cmp_delta_idwuy scans delta for IDWuy-AL; "
            "cmp_idw_grid scans every delta/alpha pair for IDW-AL."
        ),
    )
    parser.add_argument(
        "--delta-set",
        type=float_list,
        default=None if delta_set is None else list(delta_set),
        help=(
            "Comma-separated delta values, for example 1e2,1e3,1e4. "
            "Defaults to the benchmark configuration."
        ),
    )
    parser.add_argument(
        "--alpha-set",
        type=float_list,
        default=None if alpha_set is None else list(alpha_set),
        help=(
            "Comma-separated alpha values, for example 1,10,100. "
            "Defaults to the benchmark configuration."
        ),
    )
    parser.add_argument("--uncertainty-weight", type=float, default=1.0, help="Multiplier on the IDW-AL uncertainty term; use 0 for z-only acquisition.")
    parser.add_argument("--history-refresh-interval", type=int, default=1, help="Recompute the full latent-state history every N new samples for history-based acquisition methods; 1 preserves the original behavior.")
    parser.add_argument("--init-set-training", choices=("jax-sysid", "ekf"), default=None, help="Initial estimator training mode. Defaults to the experiment script setting.")
    parser.add_argument("--initial-ekf-epochs", type=int, default=None, help="Number of EKF replay epochs after initial fitting. Defaults to the experiment script setting.")
    parser.add_argument("--rho-x", type=float, default=None, help="Initial EKF state inverse-covariance scale. Initial state variance is 1 / rho_x when no covariance P is supplied.")
    parser.add_argument("--rho-th", type=float, default=None, help="Initial EKF parameter inverse-covariance scale. Initial parameter variance is 1 / rho_th when no covariance P is supplied.")
    parser.add_argument("--Qx-cov", type=float, default=None, help="EKF state process-noise variance in scaled coordinates when scaling is enabled.")
    parser.add_argument("--Qy-cov", type=float, default=None, help="EKF measurement-noise variance in scaled coordinates when scaling is enabled.")
    parser.add_argument("--Qth-cov", type=float, default=None, help="EKF parameter process-noise variance.")
    parser.add_argument("--noise-mode", choices=("additive", "multiplicative"), default=noise_mode, help="Simulation noise model for qx/qy.")
    parser.add_argument("--no-save", action="store_true", help="Do not save data or figures.")
    parser.add_argument(
        "--save3-plot",
        action="store_true",
        help=(
            "Save score plots as PDF, PNG, and SVG. Without this option, "
            "saved score plots use PDF only."
        ),
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip score plotting.")
    parser.add_argument("--load", action="store_true", help="Load the matching saved experiment and skip simulation.")
    return parser


def setup_runtime(args, *, run_dir, system_name):
    """Configure the JAX runtime used by the public examples."""
    jax.config.update("jax_platform_name", "cpu")
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


def selected_al_methods(args):
    """Return a validated custom method list or a predefined method set."""
    custom_methods = getattr(args, "methods", None)
    if custom_methods is not None:
        return validate_al_methods(custom_methods)
    try:
        return list(AL_METHOD_SETS[args.method_set])
    except KeyError as error:
        raise ValueError(
            f"Unknown acquisition method set: {args.method_set}"
        ) from error


def resolve_run_exp_type(args, exp_type=None):
    """Return the effective experiment label used for saved results."""
    resolved_exp_type = args.exp_type if exp_type is None else exp_type
    save_exp_type = args.save_exp_type
    if (
        save_exp_type is None
        and resolved_exp_type.lower() == "cmp_al"
    ):
        custom_methods = getattr(args, "methods", None)
        if custom_methods is not None:
            method_label = "_".join(
                method.lower() for method in custom_methods
            )
            save_exp_type = f"cmp_al_{method_label}"
        elif args.method_set != 4:
            method_set_save_labels = {5: "cmp_al5"}
            save_exp_type = method_set_save_labels.get(
                args.method_set, f"cmp_al{args.method_set}"
            )
    if save_exp_type is not None:
        return save_exp_type
    return resolved_exp_type


def configuration_labels(n_configurations, exp_type, al_methods, delta_set, alpha_set, ekf_config_set=None):
    """Build readable labels for result rows."""
    return score_series_labels(
        n_configurations,
        exp_type,
        al_methods,
        delta_set,
        alpha_set,
        ekf_config_set=ekf_config_set,
    )


def r2_summary(scores, labels, n_train_init):
    """Print and return the standard R2 summary."""
    r2_train = np.asarray(scores["R2_train"], dtype=float)
    r2_test = np.asarray(scores["R2_test"], dtype=float)

    lines = ["R2 SUMMARY", "=" * 64]
    for config_index, label in enumerate(labels):
        config_train = r2_train[config_index]
        config_test = r2_test[config_index]
        checkpoint_indices = score_checkpoints(config_test[None, ...], n_train_init)
        if checkpoint_indices.size == 0:
            lines.append(f"{label}: no finite R2 checkpoints")
            continue

        final_index = int(checkpoint_indices[-1])
        final_train = config_train[:, final_index, 0]
        final_test = config_test[:, final_index, 0]
        best_test = np.nanmax(config_test[:, checkpoint_indices, 0], axis=1)
        lines.append(
            f"{label}: final checkpoint k={final_index}, "
            f"train R2 mean={np.nanmean(final_train):.4f}, "
            f"test R2 mean={np.nanmean(final_test):.4f}, "
            f"best test R2 mean={np.nanmean(best_test):.4f}"
        )
        lines.append(
            f"  final test R2 per run: {np.array2string(final_test, precision=4)}"
        )

    summary = "\n".join(lines)
    print("\n" + summary)
    return summary


def performance_summary(scores, labels, n_train_init, burn_in=300):
    """Print and return a compact performance recommendation summary."""
    rmse_test = np.asarray(scores["rmse_test"], dtype=float)
    r2_test = np.asarray(scores["R2_test"], dtype=float)
    checkpoint_indices = score_checkpoints(rmse_test, n_train_init)
    if checkpoint_indices.size == 0:
        return "No finite RMSE checkpoints available."

    final_index = int(checkpoint_indices[-1])
    burn_in = max(int(burn_in), int(n_train_init))
    burn_indices = checkpoint_indices[checkpoint_indices >= burn_in]
    if burn_indices.size == 0:
        burn_indices = checkpoint_indices

    rows = []
    for config_index, label in enumerate(labels):
        final_rmse = rmse_test[config_index, :, final_index, 0]
        final_r2 = r2_test[config_index, :, final_index, 0]
        post_burn_rmse = rmse_test[config_index, :, burn_indices, 0]
        rows.append(
            {
                "label": label,
                "final_rmse_mean": float(np.nanmean(final_rmse)),
                "final_rmse_std": float(np.nanstd(final_rmse)),
                "final_r2_mean": float(np.nanmean(final_r2)),
                "post_burn_rmse_mean": float(np.nanmean(post_burn_rmse)),
            }
        )

    best_overall = min(rows, key=lambda row: row["final_rmse_mean"])
    idw_rows = [
        row
        for row in rows
        if row["label"].lower().startswith(("idw", "idw-al"))
    ]
    best_idw = min(idw_rows, key=lambda row: row["final_rmse_mean"]) if idw_rows else None
    passive = next((row for row in rows if row["label"].lower() == "passive"), None)
    igs = next((row for row in rows if row["label"].lower() == "igs"), None)

    lines = [
        "PERFORMANCE SUMMARY",
        "=" * 64,
        f"Final checkpoint: k={final_index}",
        f"Post-burn-in RMSE window: k >= {int(burn_indices[0])}",
        "",
        (
            "Recommended overall: "
            f"{best_overall['label']} "
            f"(final RMSE mean={best_overall['final_rmse_mean']:.7g}, "
            f"final R2 mean={best_overall['final_r2_mean']:.4f})"
        ),
    ]
    if best_idw is not None:
        lines.append(
            "Recommended IDW-AL setting: "
            f"{best_idw['label']} "
            f"(final RMSE mean={best_idw['final_rmse_mean']:.7g}, "
            f"post-burn RMSE mean={best_idw['post_burn_rmse_mean']:.7g})"
        )
        if passive is not None:
            lines.append(
                "  vs passive final RMSE: "
                f"{best_idw['final_rmse_mean'] - passive['final_rmse_mean']:+.7g}"
            )
        if igs is not None:
            lines.append(
                "  vs iGS final RMSE: "
                f"{best_idw['final_rmse_mean'] - igs['final_rmse_mean']:+.7g}"
            )

    lines.extend(["", "All configurations:"])
    for row in rows:
        lines.append(
            f"{row['label']}: final RMSE={row['final_rmse_mean']:.7g} "
            f"+/- {row['final_rmse_std']:.2g}, "
            f"final R2={row['final_r2_mean']:.4f}, "
            f"post-burn RMSE={row['post_burn_rmse_mean']:.7g}"
        )

    summary = "\n".join(lines)
    print("\n" + summary)
    return summary


@dataclass
class CommonRunOutput:
    result: object
    labels: list
    r2_summary: str
    performance_summary: str
    ekf_summary: dict
    score_figures: tuple
    ekf_figure: object = None


@dataclass(frozen=True)
class EKFDefaults:
    """System-specific EKF defaults used by the public examples."""

    rho_x: float
    rho_th: float
    qy_cov: float
    qth_cov: float
    qx_cov_floor: float = 1e-10
    initial_training: str = "jax-sysid"
    initial_ekf_epochs: int = 1


@dataclass(frozen=True)
class IDWDefaults:
    """Default IDW acquisition weights for one benchmark configuration."""

    delta: float
    alpha: float


def add_standard_example_arguments(parser, **common_defaults):
    """Add CLI arguments shared by the public benchmark examples."""
    add_common_arguments(
        parser,
        exp_type_choices=("cmp_al", "cmp_delta", "cmp_alpha"),
        **common_defaults,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="First experiment seed; repetitions use consecutive seeds.",
    )
    parser.set_defaults(email=False)
    return parser


def experiment_seeds(args):
    """Return deterministic consecutive seeds for an experiment."""
    return list(range(args.seed, args.seed + args.n_exp))


def run_common_experiment(
    *,
    args,
    system,
    model,
    u_set,
    system_name,
    model_name,
    n_train_init,
    n_train_max,
    n_test,
    al_methods,
    exp_type,
    delta_set,
    alpha_set,
    qx,
    qy,
    pred,
    is_scale,
    is_const,
    Ts,
    temporality,
    rho_x,
    rho_th,
    Qx_cov,
    Qy_cov,
    Qth_cov,
    al_method="passive",
    init_set_trainning="jax-sysid",
    initial_ekf_epochs=1,
    ekf_config_set=None,
    acquisition_config_set=None,
    experiment_seeds=None,
    fixed_test_set=False,
    evaluation_max=None,
    noise_mode="additive",
    burn_in=300,
):
    """Run the shared experiment workflow and emit summaries."""
    save_outputs = not args.no_save
    plot_scores = not args.no_plot
    figure_formats = ("pdf", "png", "svg") if args.save3_plot else ("pdf",)
    run_exp_type = resolve_run_exp_type(args, exp_type)
    save_exp_type = run_exp_type

    result = run_or_load_experiment(
        run_simulation=not args.load,
        save_data=save_outputs,
        system=system,
        model=model,
        u_set=u_set,
        system_name=system_name,
        model_name=model_name,
        N_train_init=n_train_init,
        N_train_max=n_train_max,
        N_test=n_test,
        N_exp=args.n_exp if experiment_seeds is None else len(experiment_seeds),
        AL_method_set=al_methods,
        exp_type=exp_type,
        save_exp_type=save_exp_type,
        qx=qx,
        qy=qy,
        AL_method=al_method,
        delta_set=delta_set,
        alpha_set=alpha_set,
        uncertainty_weight=args.uncertainty_weight,
        history_refresh_interval=args.history_refresh_interval,
        pred=pred,
        init_set_trainning=init_set_trainning,
        isScale=is_scale,
        isConst=is_const,
        Ts=Ts,
        temporality=temporality,
        rho_x=rho_x,
        rho_th=rho_th,
        Qx_cov=Qx_cov,
        Qy_cov=Qy_cov,
        Qth_cov=Qth_cov,
        noise_mode=noise_mode,
        initial_ekf_epochs=initial_ekf_epochs,
        ekf_config_set=ekf_config_set,
        acquisition_config_set=acquisition_config_set,
        score_interval=args.score_interval,
        experiment_seeds=experiment_seeds,
        fixed_test_set=fixed_test_set,
        plot_scores=plot_scores,
        save_figures=save_outputs,
        figure_formats=figure_formats,
        diagnostic_plot=False,
        diagnostic_save=False,
        diagnostic_show=False,
        evaluation_max=evaluation_max,
    )

    labels = ([config["label"] for config in acquisition_config_set]
              if acquisition_config_set is not None else configuration_labels(
        result.scores["R2_test"].shape[0],
        result.exp_type,
        result.AL_method_set,
        result.delta_set,
        result.alpha_set,
        ekf_config_set=ekf_config_set,
    ))
    r2_text = r2_summary(result.scores, labels, n_train_init)
    perf_text = performance_summary(result.scores, labels, n_train_init, burn_in=burn_in)

    ekf_summary = {}
    ekf_figure = None


    return CommonRunOutput(
        result=result,
        labels=labels,
        r2_summary=r2_text,
        performance_summary=perf_text,
        ekf_summary=ekf_summary,
        score_figures=result.score_figures if result.score_figures else (None, None),
        ekf_figure=ekf_figure,
    )


def run_standard_example(
    *,
    args,
    system,
    model,
    system_name,
    model_name,
    ekf_defaults,
    idw_defaults,
    fixed_test_set=False,
):
    """Run the workflow shared by the public nonlinear benchmark examples."""
    al_methods = selected_al_methods(args)
    if args.exp_type.lower() in {"cmp_alpha", "cmp_delta"}:
        al_methods = ["passive", "IDW"]

    delta_set = (
        list(args.delta_set)
        if args.delta_set is not None
        else [idw_defaults.delta]
    )
    alpha_set = (
        list(args.alpha_set)
        if args.alpha_set is not None
        else [idw_defaults.alpha]
    )

    rho_x = args.rho_x if args.rho_x is not None else ekf_defaults.rho_x
    rho_th = args.rho_th if args.rho_th is not None else ekf_defaults.rho_th
    Qx_cov = (
        args.Qx_cov
        if args.Qx_cov is not None
        else max(system.qx**2, ekf_defaults.qx_cov_floor)
    )
    Qy_cov = (
        args.Qy_cov if args.Qy_cov is not None else ekf_defaults.qy_cov
    )
    Qth_cov = (
        args.Qth_cov if args.Qth_cov is not None else ekf_defaults.qth_cov
    )
    initial_training = (
        args.init_set_training or ekf_defaults.initial_training
    )
    initial_ekf_epochs = (
        args.initial_ekf_epochs
        if args.initial_ekf_epochs is not None
        else ekf_defaults.initial_ekf_epochs
    )

    return run_common_experiment(
        args=args,
        system=system,
        model=model,
        u_set=system.u_set,
        system_name=system_name,
        model_name=model_name,
        n_train_init=args.n_train_init,
        n_train_max=args.n_train_max,
        n_test=args.n_test,
        al_methods=al_methods,
        exp_type=args.exp_type,
        delta_set=delta_set,
        alpha_set=alpha_set,
        qx=system.qx,
        qy=system.qy,
        pred="EKF_step",
        is_scale=True,
        is_const=bool(system.const["flag"]),
        Ts=system.Ts,
        temporality=system.temporality,
        rho_x=rho_x,
        rho_th=rho_th,
        Qx_cov=Qx_cov,
        Qy_cov=Qy_cov,
        Qth_cov=Qth_cov,
        init_set_trainning=initial_training,
        initial_ekf_epochs=initial_ekf_epochs,
        experiment_seeds=experiment_seeds(args),
        fixed_test_set=fixed_test_set,
    )
