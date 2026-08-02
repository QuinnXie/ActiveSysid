"""Shared run/load/save/plot workflow for system-identification experiments."""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from example.analysis.plotting import save_plot
from example.experiment_plotting import create_standard_mean_sem_score_figures
from example.simulation import save_constraint_violation_summary, simulation
from activesysid.data_save_load import load_data_pkl, save_data_pkl


@dataclass
class ExperimentResult:
    samples: dict
    scores: dict
    system: object
    model: object
    pred: str
    exp_type: str
    AL_method_set: list
    delta_set: list
    alpha_set: list
    N_exp: int
    N_set: int
    N_train_init: int
    N_train_max: int
    N_test: int
    rho_x: float
    rho_th: float
    Qx_cov: float
    Qy_cov: float
    Qth_cov: float
    isScale: int
    isConst: int
    initial_ekf_epochs: int = 1
    init_set_trainning: str = "jax-sysid"
    evaluation_max: int | None = None
    score_figures: tuple = ()
    diagnostic_figures: tuple = ()


def run_or_load_experiment(
    *,
    run_simulation,
    save_data,
    system,
    model,
    u_set,
    system_name,
    model_name,
    N_train_init,
    N_train_max,
    N_test,
    N_exp,
    AL_method_set,
    exp_type,
    qx,
    qy,
    AL_method,
    delta_set,
    alpha_set,
    pred,
    init_set_trainning,
    isScale,
    isConst,
    Ts,
    temporality,
    rho_x,
    rho_th,
    Qx_cov,
    Qy_cov,
    Qth_cov,
    uncertainty_weight=1.0,
    history_refresh_interval=1,
    initial_ekf_epochs=1,
    score_interval=10,
    experiment_seeds=None,
    fixed_test_set=False,
    verbose=1,
    plot_scores=True,
    save_figures=True,
    figure_formats=("pdf",),
    diagnostic_plot=False,
    diagnostic_save=False,
    diagnostic_show=False,
    diagnostic_configuration=0,
    diagnostic_experiment=0,
    evaluation_max=None,
    record_ekf_diagnostics=False,
    save_exp_type=None,
    ekf_config_set=None,
    acquisition_config_set=None,
    noise_mode="additive",
    ekf_nis_gate=None,
):
    """Run/load an experiment and apply the shared plotting workflow."""
    output_exp_type = exp_type if save_exp_type is None else save_exp_type
    if run_simulation:
        samples, scores = simulation(
            system,
            model,
            u_set,
            N_train_init,
            N_train_max,
            N_test=N_test,
            AL_method_set=AL_method_set,
            N_exp=N_exp,
            exp_type=exp_type,
            qx=qx,
            qy=qy,
            AL_method=AL_method,
            delta_set=delta_set,
            alpha_set=alpha_set,
            uncertainty_weight=uncertainty_weight,
            history_refresh_interval=history_refresh_interval,
            verbose=verbose,
            pred=pred,
            init_set_trainning=init_set_trainning,
            isScale=isScale,
            isConst=isConst,
            Ts=Ts,
            temporality=temporality,
            rho_x=rho_x,
            rho_th=rho_th,
            Qx_cov=Qx_cov,
            Qy_cov=Qy_cov,
            Qth_cov=Qth_cov,
            initial_ekf_epochs=initial_ekf_epochs,
            score_interval=score_interval,
            experiment_seeds=experiment_seeds,
            fixed_test_set=fixed_test_set,
            system_name=system_name,
            evaluation_max=evaluation_max,
            record_ekf_diagnostics=record_ekf_diagnostics,
            ekf_config_set=ekf_config_set,
            acquisition_config_set=acquisition_config_set,
            noise_mode=noise_mode,
            ekf_nis_gate=ekf_nis_gate,
        )
        N_set = scores["R2_test"].shape[0]
        saved_AL_method_set = (
            [config["label"] for config in acquisition_config_set]
            if acquisition_config_set is not None
            else
            [config.get("label", f"EKF {index}") for index, config in enumerate(ekf_config_set)]
            if exp_type.lower() == "cmp_ekf" and ekf_config_set is not None
            else AL_method_set
        )
        save_data_pkl(
            save_data,
            samples,
            scores,
            system,
            model,
            pred,
            output_exp_type,
            N_train_init,
            N_train_max,
            N_test,
            N_exp,
            N_set,
            saved_AL_method_set,
            delta_set,
            alpha_set,
            rho_x,
            rho_th,
            Qx_cov,
            Qy_cov,
            Qth_cov,
            system_name,
            model_name,
            isScale,
            isConst,
        )
        values = locals()
        # Keep the simulation/plotting mode separate from the optional label
        # used to name the saved artifact.
        values["exp_type"] = exp_type
        values["initial_ekf_epochs"] = initial_ekf_epochs
        values["init_set_trainning"] = init_set_trainning
        values["AL_method_set"] = saved_AL_method_set
    else:
        loaded = load_data_pkl(
            True,
            output_exp_type,
            int(qx != 0.0 or qy != 0.0),
            system_name=system_name,
            model_name=model_name,
            isScale=isScale,
            isConst=isConst,
        )
        # Older cmp_delta/cmp_alpha files may contain a stale N_set value
        # based on len(AL_method_set), while the score array has more rows.
        loaded["N_set"] = loaded["scores"]["R2_test"].shape[0]
        loaded["N_exp"] = loaded["scores"]["R2_test"].shape[1]
        # Older files store save_exp_type in this field.  Plotting needs the
        # canonical mode supplied by the caller (for example ``cmp_delta``).
        loaded["exp_type"] = exp_type
        values = loaded

    result = ExperimentResult(
        **{
            field: values[field]
            for field in ExperimentResult.__dataclass_fields__
            if field in values
        }
    )

    saved_const = (
        result.system.get("const")
        if isinstance(result.system, dict)
        else result.system.const
    )
    if not run_simulation and saved_const is not None:
        save_constraint_violation_summary(
            result.samples,
            result.scores,
            result.system,
            system_name=system_name,
            exp_type=result.exp_type,
            AL_method_set=result.AL_method_set,
            delta_set=result.delta_set,
            alpha_set=result.alpha_set,
            N_train_init=result.N_train_init,
            exclude_nonfinite_runs=True,
            output_suffix="" if result.isConst else "_no_const",
            evaluation_max=evaluation_max,
        )

    if plot_scores:
        result.score_figures = plot_experiment_scores(
            result,
            system_name=system_name,
            model_name=model_name,
            save=save_figures,
            formats=figure_formats,
            evaluation_max=evaluation_max,
            artifact_exp_type=output_exp_type,
        )
    result.diagnostic_figures = plot_experiment_diagnostics(
        result,
        configuration=diagnostic_configuration,
        experiment=diagnostic_experiment,
        plot=diagnostic_plot,
        save=diagnostic_save,
        show=diagnostic_show,
        fig_dir=system_name,
        file_prefix=f"{system_name}_sysid",
        evaluation_max=evaluation_max,
    )
    return result


def _run_timing(result, configuration=0, experiment=0):
    """Return timing/diagnostic metadata for one aggregate experiment run."""
    timings = result.scores.get("timings")
    if timings is None:
        return None
    if isinstance(timings, dict):
        return timings
    return timings[configuration, experiment]


def summarize_ekf_diagnostics(
    result,
    *,
    configuration=0,
    experiment=0,
    evaluation_max=None,
    print_summary=True,
):
    """Summarize EKF health diagnostics for one active-learning run."""
    timing = _run_timing(result, configuration, experiment)
    if not timing or "ekf_diagnostics" not in timing:
        return {}

    diagnostics = timing["ekf_diagnostics"]
    nis = np.asarray(diagnostics["nis"], dtype=float)
    parameter_update_norm = np.asarray(
        diagnostics["parameter_update_norm"], dtype=float
    )
    parameter_covariance_trace = np.asarray(
        diagnostics["parameter_covariance_trace"], dtype=float
    )
    covariance_min_eig = np.asarray(
        diagnostics["covariance_min_eig"], dtype=float
    )

    if evaluation_max is None:
        evaluation_max = nis.shape[0]
    evaluation_max = min(int(evaluation_max), nis.shape[0])
    slc = slice(0, evaluation_max)

    finite_trace = parameter_covariance_trace[
        np.isfinite(parameter_covariance_trace[slc])
    ]
    summary = {
        "failed": bool(timing.get("failed", False)),
        "failure_index": timing.get("failure_index"),
        "failure_reason": timing.get("failure_reason"),
        "max_nis": float(np.nanmax(nis[slc])) if np.any(np.isfinite(nis[slc])) else np.nan,
        "median_nis": float(np.nanmedian(nis[slc])) if np.any(np.isfinite(nis[slc])) else np.nan,
        "max_parameter_update_norm": (
            float(np.nanmax(parameter_update_norm[slc]))
            if np.any(np.isfinite(parameter_update_norm[slc]))
            else np.nan
        ),
        "min_covariance_eig": (
            float(np.nanmin(covariance_min_eig[slc]))
            if np.any(np.isfinite(covariance_min_eig[slc]))
            else np.nan
        ),
        "final_parameter_covariance_trace": (
            float(finite_trace[-1]) if finite_trace.size else np.nan
        ),
    }

    if print_summary:
        print(
            "EKF diagnostics "
            f"(configuration={configuration}, experiment={experiment}):"
        )
        print(
            "  failed={failed}, failure_index={failure_index}, "
            "failure_reason={failure_reason}".format(**summary)
        )
        print(
            "  NIS median={median_nis:.4g}, max={max_nis:.4g}; "
            "max ||dtheta||={max_parameter_update_norm:.4g}".format(**summary)
        )
        print(
            "  min eig(P)={min_covariance_eig:.4g}; "
            "final trace(P_theta)={final_parameter_covariance_trace:.4g}".format(
                **summary
            )
        )
    return summary


def plot_ekf_diagnostics(
    result,
    *,
    configuration=0,
    experiment=0,
    save=False,
    show=False,
    fig_dir=None,
    file_prefix="sysid_diagnostics",
    evaluation_max=None,
    close=True,
):
    """Plot NIS, parameter update norm, and covariance health for one run."""
    timing = _run_timing(result, configuration, experiment)
    if not timing or "ekf_diagnostics" not in timing:
        return None

    diagnostics = timing["ekf_diagnostics"]
    nis = np.asarray(diagnostics["nis"], dtype=float)
    parameter_update_norm = np.asarray(
        diagnostics["parameter_update_norm"], dtype=float
    )
    parameter_covariance_trace = np.asarray(
        diagnostics["parameter_covariance_trace"], dtype=float
    )
    covariance_min_eig = np.asarray(
        diagnostics["covariance_min_eig"], dtype=float
    )

    if evaluation_max is None:
        evaluation_max = nis.shape[0]
    evaluation_max = min(int(evaluation_max), nis.shape[0])
    axis = np.arange(evaluation_max)

    figure, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    axes[0].semilogy(axis, nis[:evaluation_max], label="NIS")
    axes[0].axhline(3.84, color="k", linestyle="--", linewidth=1, label="95%")
    axes[0].axhline(6.63, color="r", linestyle="--", linewidth=1, label="99%")
    axes[0].set_title("EKF normalized innovation squared")
    axes[0].legend()

    axes[1].semilogy(
        axis,
        parameter_update_norm[:evaluation_max],
        label="parameter update norm",
    )
    axes[1].set_title("Parameter update norm")
    axes[1].legend()

    axes[2].plot(
        axis,
        covariance_min_eig[:evaluation_max],
        label="minimum eigenvalue",
    )
    axes[2].axhline(0.0, color="k", linestyle="--", linewidth=1)
    axes[2].set_title("Minimum eigenvalue of P")
    axes[2].legend()

    axes[3].semilogy(
        axis,
        parameter_covariance_trace[:evaluation_max],
        label="trace(P_theta)",
    )
    axes[3].set_title("Parameter covariance trace")
    axes[3].set_xlabel("Training index")
    axes[3].legend()

    for axis_obj in axes:
        axis_obj.grid(True, alpha=0.3)
    figure.tight_layout()

    if save:
        save_plot(
            figure,
            f"{file_prefix}_ekf_c{configuration}_e{experiment}",
            fig_dir=fig_dir,
        )
    if show:
        plt.show()
    elif close:
        plt.close(figure)
    return figure


def plot_experiment_scores(
    result,
    *,
    system_name,
    model_name,
    save=True,
    formats=("pdf",),
    evaluation_max=None,
    artifact_exp_type=None,
):
    """Create the standard test RMSE and R2 mean/SEM figures."""
    if evaluation_max is None:
        evaluation_max = result.N_train_max
    evaluation_max = min(int(evaluation_max), result.N_train_max)
    if artifact_exp_type is None:
        artifact_exp_type = result.exp_type
    return create_standard_mean_sem_score_figures(
        result.scores,
        n_train_init=result.N_train_init,
        evaluation_max=evaluation_max,
        exp_type=result.exp_type,
        artifact_exp_type=artifact_exp_type,
        al_methods=result.AL_method_set,
        delta_set=result.delta_set,
        alpha_set=result.alpha_set,
        system_name=system_name,
        model_name=model_name,
        pred=result.pred,
        is_scale=result.isScale,
        is_const=result.isConst,
        save=save,
        formats=formats,
    )


def plot_experiment_diagnostics(
    result,
    *,
    configuration=0,
    experiment=0,
    plot=False,
    save=False,
    show=False,
    fig_dir=None,
    file_prefix="sysid_diagnostics",
    evaluation_max=None,
):
    """Plot one run's outputs and train/test score histories.

    The function does nothing by default. Set ``plot=True`` to create the
    figures, and independently enable ``save`` or ``show`` if desired.
    """
    if not plot:
        return ()

    samples = result.samples
    scores = result.scores
    Y_train = samples["Y_train"][configuration, experiment]
    Y_test = samples["Y_test"][configuration, experiment]
    Yhat_train = samples["Yhat_train"][configuration, experiment]
    Yhat_test = samples["Yhat_test"][configuration, experiment]
    r2_train = scores["R2_train"][configuration, experiment]
    r2_test = scores["R2_test"][configuration, experiment]
    rmse_train = scores["rmse_train"][configuration, experiment]
    rmse_test = scores["rmse_test"][configuration, experiment]

    if evaluation_max is None:
        evaluation_max = Y_train.shape[0]
    evaluation_max = min(int(evaluation_max), Y_train.shape[0])
    train_axis = np.arange(evaluation_max)
    test_axis = np.arange(Y_test.shape[0])
    score_axis = np.arange(result.N_train_init, evaluation_max)
    figures = []

    fig_output, axes = plt.subplots(2, 1, figsize=(8, 6))
    axes[0].plot(train_axis, Y_train[:evaluation_max, 0], label="measured")
    axes[0].plot(
        train_axis, Yhat_train[:evaluation_max, 0], label="predicted"
    )
    axes[0].set_title("Y (training data)")
    axes[1].plot(test_axis, Y_test[:, 0], label="measured")
    axes[1].plot(test_axis, Yhat_test[:, 0], label="predicted")
    axes[1].set_title("Y (test data)")
    for axis in axes:
        axis.legend()
        axis.grid(True, alpha=0.3)
    fig_output.tight_layout()
    figures.append(("output", fig_output))

    fig_r2, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(
        score_axis,
        r2_train[result.N_train_init:evaluation_max, 0],
        label="R²",
    )
    axes[0].set_title("R² (training data)")
    axes[1].plot(
        score_axis,
        r2_test[result.N_train_init:evaluation_max, 0],
        label="R²",
    )
    axes[1].set_title("R² (test data)")
    for axis in axes:
        axis.legend()
        axis.grid(True, alpha=0.3)
    fig_r2.tight_layout()
    figures.append(("r2", fig_r2))

    fig_rmse, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(
        score_axis,
        rmse_train[result.N_train_init:evaluation_max, 0],
        label="RMSE",
    )
    axes[0].set_title("RMSE (training data)")
    axes[1].plot(
        score_axis,
        rmse_test[result.N_train_init:evaluation_max, 0],
        label="RMSE",
    )
    axes[1].set_title("RMSE (test data)")
    for axis in axes:
        axis.legend()
        axis.grid(True, alpha=0.3)
    fig_rmse.tight_layout()
    figures.append(("rmse", fig_rmse))

    ekf_figure = plot_ekf_diagnostics(
        result,
        configuration=configuration,
        experiment=experiment,
        save=False,
        show=False,
        evaluation_max=evaluation_max,
        close=False,
    )
    if ekf_figure is not None:
        figures.append(("ekf", ekf_figure))

    if save:
        for suffix, figure in figures:
            save_plot(
                figure,
                f"{file_prefix}_{suffix}_c{configuration}_e{experiment}",
                fig_dir=fig_dir,
            )
    if show:
        plt.show()
    else:
        for _, figure in figures:
            plt.close(figure)

    return tuple(figure for _, figure in figures)
