import numpy as np
from pathlib import Path
import time
# import pdb

import jax
from jax_sysid.utils import vec_reshape
from activesysid.active_sysid import active_learning_sysid
from activesysid.predict import predict
from activesysid.data_save_load import load_data_pkl
from activesysid.utils import compute_constraint_violation, init_results, save_results
from example.experiment_utils import update_params
import gc
from copy import deepcopy
# from memory_profiler import profile


def _is_cmp_al_mode(exp_type):
    """Return True for cmp_al variants (for example cmp_al5), but not cmp_alpha."""
    exp_type = exp_type.lower()
    return exp_type == "cmp_al" or (
        exp_type.startswith("cmp_al") and exp_type != "cmp_alpha"
    )


def experiment_figure_dir(system_name):
    """Return the shared experiment-artifact figure directory."""
    return (
        Path(__file__).resolve().parent
        / "experiments"
        / "artifacts"
        / "figures"
        / str(system_name)
    )


def print_training_summary(
    system,
    model,
    u_set,
    *,
    system_name=None,
    exp_type,
    AL_method_set,
    delta_set,
    alpha_set,
    N_exp,
    N_train_init,
    N_train_max,
    N_test,
    pred,
    init_set_trainning,
    qx,
    qy,
    Ts,
    temporality,
    isScale,
    isConst,
    score_interval,
    rho_x,
    rho_th,
    Qx_cov,
    Qy_cov,
    Qth_cov,
    noise_mode,
    initial_ekf_epochs,
):
    """Print the key configuration of a system-identification experiment."""
    exp_type = exp_type.lower()
    system_name = system_name or system.__class__.__name__
    u_set = np.asarray(u_set).reshape(-1, system.nu)

    if _is_cmp_al_mode(exp_type):
        comparison_name = "AL methods"
        comparison_values = AL_method_set
    elif exp_type == 'cmp_delta':
        comparison_name = "IDW-AL delta set"
        comparison_values = delta_set
    elif exp_type == 'cmp_delta_idwuy':
        comparison_name = "IDWuy-AL delta set"
        comparison_values = delta_set
    elif exp_type == 'cmp_ekf':
        comparison_name = "EKF setting set"
        comparison_values = AL_method_set
    elif exp_type == 'cmp_alpha':
        comparison_name = "alpha set"
        comparison_values = alpha_set
    elif exp_type == 'cmp_idw_grid':
        comparison_name = "IDW-AL grid"
        comparison_values = {
            "baseline_methods": AL_method_set[:-1],
            "delta_set": delta_set,
            "alpha_set": alpha_set,
        }
    else:
        comparison_name = "comparison set"
        comparison_values = "unknown"

    input_ranges = [
        f"[{u_set[:, i].min():g}, {u_set[:, i].max():g}]"
        for i in range(system.nu)
    ]

    print("\n" + "=" * 64)
    print("TRAINING CONFIGURATION")
    print("=" * 64)
    print(f"System             : {system_name}")
    print(f"Experiment type    : {exp_type}")
    print(f"{comparison_name:<19}: {comparison_values}")
    print(f"Model              : {model.__class__.__name__}")
    print(
        f"Dimensions         : system nx={system.nx}, model nx={model.nx}, "
        f"ny={system.ny}, nu={system.nu}"
    )
    print(f"Prediction         : {pred}")
    print(f"Initial training   : {init_set_trainning}")
    print(f"Initial EKF epochs : {initial_ekf_epochs}")
    print(
        f"EKF covariance     : rho_x={rho_x:g}, rho_th={rho_th:g}, "
        f"Qx_cov={Qx_cov:g}, Qy_cov={Qy_cov:g}, Qth_cov={Qth_cov:g}"
    )
    print(f"Experiments        : {N_exp}")
    print(
        f"Dataset sizes      : initial={N_train_init}, "
        f"maximum={N_train_max}, test={N_test}"
    )
    print(f"Input pool         : {len(u_set)} candidates, ranges={input_ranges}")
    print(f"Noise              : qx={qx:g}, qy={qy:g}, mode={noise_mode}")
    print(f"Time setting       : {temporality}, Ts={Ts:g}")
    print(
        f"Options            : scaling={bool(isScale)}, "
        f"constraints={bool(isConst)}, score interval={score_interval}"
    )
    print("=" * 64 + "\n")


def _config_value(value):
    """Return a pickle-friendly scalar/list representation for run metadata."""
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array.tolist()


def _configuration_labels(
    n_configurations,
    exp_type,
    AL_method_set,
    delta_set,
    alpha_set,
):
    """Build readable labels for the compared experiment configurations."""
    labels = []
    for n_set in range(n_configurations):
        method, delta, alpha = update_params(
            n_set, exp_type, AL_method_set, delta_set, alpha_set
        )
        if exp_type.lower() == "cmp_ekf":
            labels.append(AL_method_set[n_set])
        elif method.lower() == 'idw' and exp_type.lower() == 'cmp_delta':
            labels.append(f"IDW-AL (delta={delta:g})")
        elif method.lower() == 'idwuy' and exp_type.lower() == 'cmp_delta_idwuy':
            labels.append(f"IDWuy-AL (delta={delta:g})")
        elif method.lower() == 'idw' and exp_type.lower() == 'cmp_alpha':
            labels.append(f"IDW-AL (alpha={alpha:g})")
        elif method.lower() == 'idw' and exp_type.lower() == 'cmp_idw_grid':
            labels.append(f"IDW-AL (delta={delta:g}, alpha={alpha:g})")
        elif method.lower() == 'idw':
            labels.append(f"IDW-AL (delta={delta:g}, alpha={alpha:g})")
        else:
            labels.append(method)
    return labels


def _format_timing_stats(name, stats):
    """Format one timing-statistics dictionary for terminal or text output."""
    if not stats or stats.get("n", 0) == 0:
        return f"{name}: no measured samples"
    return (
        f"{name}: n={stats['n']}, warmup={stats['warmup_samples']}, "
        f"min={stats['min']:.8f}s, median={stats['median']:.8f}s, "
        f"mean={stats['mean']:.8f}s, p25={stats['p25']:.8f}s, "
        f"p75={stats['p75']:.8f}s, max={stats['max']:.8f}s, "
        f"std={stats['std']:.8f}s"
    )


def _aggregate_timing_samples(timing_runs, timing_key):
    """Combine measured samples from several runs, excluding each warmup."""
    measured_samples = []
    for timing in timing_runs:
        stats = timing.get(timing_key)
        if not stats:
            continue
        samples = np.asarray(stats.get("samples", []), dtype=float)
        warmup = int(stats.get("warmup_samples", 0))
        measured_samples.extend(samples[warmup:])

    samples = np.asarray(measured_samples, dtype=float)
    if samples.size == 0:
        return None
    return {
        "n": samples.size,
        "warmup_samples": 0,
        "min": float(np.min(samples)),
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "p25": float(np.percentile(samples, 25)),
        "p75": float(np.percentile(samples, 75)),
        "max": float(np.max(samples)),
        "std": (
            float(np.std(samples, ddof=1))
            if samples.size > 1 else 0.0
        ),
    }


def save_timing_summary(
    scores,
    *,
    system_name,
    exp_type,
    AL_method_set,
    delta_set,
    alpha_set,
    total_simulation_time,
):
    """Save per-run timing details and aggregate runtime statistics."""
    if not (_is_cmp_al_mode(exp_type) or exp_type.lower() == "cmp_ekf"):
        return None

    timings = scores['timings']
    labels = _configuration_labels(
        timings.shape[0],
        exp_type,
        AL_method_set,
        delta_set,
        alpha_set,
    )
    timing_keys = (
        ("active_learning", "Active learning"),
        ("EKF_step", "EKF_step"),
        ("EKF_measurement_update", "EKF measurement update"),
        ("EKF_time_update", "EKF time update"),
        ("EKF_set", "EKF_set"),
        ("learn_x0", "Learn x0"),
        ("model_fit", "model.fit"),
    )

    system_name = system_name or "System"
    output_dir = experiment_figure_dir(system_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"timing_summary_{exp_type.lower()}.txt"

    lines = [
        "Runtime summary",
        "=" * 100,
        f"System: {system_name}",
        f"Experiment type: {exp_type}",
        f"End-to-end simulation time: {total_simulation_time:.4f}s "
        f"({total_simulation_time / 60.0:.2f}min)",
        "",
        "Per-run timing details:",
    ]

    all_wall_times = []
    for n_set, label in enumerate(labels):
        lines.extend(["", f"Configuration {n_set}: {label}", "-" * 100])
        for n_exp in range(timings.shape[1]):
            timing = timings[n_set, n_exp]
            if not timing:
                lines.append(f"Experiment {n_exp}: no timing data")
                continue

            wall_time = float(timing.get("total_run_time", np.nan))
            if np.isfinite(wall_time):
                all_wall_times.append(wall_time)
            lines.append(
                f"Experiment {n_exp}: total wall time={wall_time:.8f}s"
            )
            if timing.get("failed", False):
                lines.append(
                    "  FAILED: "
                    f"training index={timing.get('failure_index')}, "
                    f"reason={timing.get('failure_reason')}"
                )
            for key, display_name in timing_keys:
                stats = timing.get(key)
                if stats and stats.get("n", 0) > 0:
                    lines.append(
                        "  " + _format_timing_stats(display_name, stats)
                    )

    lines.extend(["", "Summary across experiments:", "=" * 100])
    summary_rows = []
    for n_set, label in enumerate(labels):
        timing_runs = [
            timings[n_set, n_exp]
            for n_exp in range(timings.shape[1])
            if timings[n_set, n_exp]
        ]
        wall_times = np.asarray(
            [
                timing.get("total_run_time", np.nan)
                for timing in timing_runs
            ],
            dtype=float,
        )
        wall_times = wall_times[np.isfinite(wall_times)]
        if wall_times.size:
            summary = (
                f"{label}: total wall time across runs={np.sum(wall_times):.4f}s, "
                f"mean={np.mean(wall_times):.4f}s, "
                f"std={np.std(wall_times):.4f}s, "
                f"min={np.min(wall_times):.4f}s, "
                f"max={np.max(wall_times):.4f}s"
            )
            lines.append(summary)
            summary_rows.append(summary)

        for key, display_name in timing_keys:
            stats = _aggregate_timing_samples(timing_runs, key)
            if stats:
                lines.append(
                    "  " + _format_timing_stats(display_name, stats)
                )
        lines.append("")

    if all_wall_times:
        lines.append(
            f"Overall active-learning calls: {len(all_wall_times)} runs, "
            f"total={np.sum(all_wall_times):.4f}s "
            f"({np.sum(all_wall_times) / 60.0:.2f}min), "
            f"mean={np.mean(all_wall_times):.4f}s per run"
        )
    lines.append(
        f"End-to-end simulation time: {total_simulation_time:.4f}s "
        f"({total_simulation_time / 60.0:.2f}min)"
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nRUNTIME SUMMARY")
    print("=" * 64)
    for summary in summary_rows:
        print(summary)
    print("\nPER-STEP TIMING SUMMARY")
    print("=" * 64)
    for n_set, label in enumerate(labels):
        timing_runs = [
            timings[n_set, n_exp]
            for n_exp in range(timings.shape[1])
            if timings[n_set, n_exp]
        ]
        print(f"{label}:")
        for key, display_name in timing_keys:
            stats = _aggregate_timing_samples(timing_runs, key)
            if stats and stats["n"] > 0:
                print(
                    f"  {display_name}: n={stats['n']}, "
                    f"median={stats['median']:.8f}s, "
                    f"mean={stats['mean']:.8f}s, "
                    f"p75={stats['p75']:.8f}s, "
                    f"max={stats['max']:.8f}s"
                )
    if all_wall_times:
        print(
            f"Overall: {np.sum(all_wall_times):.4f}s "
            f"({np.sum(all_wall_times) / 60.0:.2f}min)"
        )
    print(
        f"End-to-end simulation: {total_simulation_time:.4f}s "
        f"({total_simulation_time / 60.0:.2f}min)"
    )
    print(f"Timing details saved to: {output_path}")

    return output_path


def save_constraint_violation_summary(
    samples,
    scores,
    system,
    *,
    system_name,
    exp_type,
    AL_method_set,
    delta_set,
    alpha_set,
    N_train_init,
    exclude_nonfinite_runs=False,
    output_suffix="",
    y_min_override=None,
    y_max_override=None,
    evaluation_max=None,
):
    """Compute and save training constraint violations after initialization."""
    const = system.get("const") if isinstance(system, dict) else system.const
    if const is None:
        raise ValueError("Saved system metadata does not contain constraints")
    y_min = const['y_min'] if y_min_override is None else y_min_override
    y_max = const['y_max'] if y_max_override is None else y_max_override
    N_train_max = samples['Y_train'].shape[2]
    if evaluation_max is None:
        evaluation_max = N_train_max
    evaluation_max = int(evaluation_max)
    if not N_train_init < evaluation_max <= N_train_max:
        raise ValueError(
            "evaluation_max must be greater than N_train_init and no greater "
            "than the saved training length"
        )
    Y_train = samples['Y_train'][:, :, N_train_init:evaluation_max, :]

    mean_violation, count_violation = compute_constraint_violation(
        Y_train, y_min, y_max
    )
    valid_run_mask = np.all(np.isfinite(Y_train), axis=(2, 3))
    if exclude_nonfinite_runs:
        mean_violation = np.asarray(mean_violation, dtype=float)
        count_violation = np.asarray(count_violation, dtype=float)
        mean_violation[~valid_run_mask] = np.nan
        count_violation[~valid_run_mask] = np.nan

    n_evaluated = Y_train.shape[2] * Y_train.shape[3]
    count_percentage = 100.0 * count_violation / n_evaluated

    scores['mean_constraint_violation'] = mean_violation
    scores['count_constraint_violation'] = count_violation
    scores['constraint_violation_percentage'] = count_percentage
    scores['constraint_violation_valid_run'] = valid_run_mask

    labels = _configuration_labels(
        mean_violation.shape[0],
        exp_type,
        AL_method_set,
        delta_set,
        alpha_set,
    )

    if not system_name:
        system_name = (
            system.get("system_name") or system.get("class_name")
            if isinstance(system, dict)
            else system.__class__.__name__
        )
    output_dir = experiment_figure_dir(system_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"constraint_violation_{exp_type.lower()}{output_suffix}.txt"
    )

    lines = [
        "Constraint violation summary (training data)",
        "=" * 64,
        f"System: {system_name}",
        f"Experiment type: {exp_type}",
        f"Evaluated training indices: [{N_train_init}, {evaluation_max})",
        f"Evaluated samples per run: {Y_train.shape[2]}",
        f"Output dimensions: {Y_train.shape[3]}",
        f"Lower bound: {np.asarray(y_min)}",
        f"Upper bound: {np.asarray(y_max)}",
        f"Exclude non-finite runs: {exclude_nonfinite_runs}",
        (
            "Valid runs per configuration: "
            f"{np.sum(valid_run_mask, axis=1).tolist()}"
        ),
        (
            "Excluded run indices per configuration: "
            f"{[np.flatnonzero(~row).tolist() for row in valid_run_mask]}"
        ),
        "",
        "Mean constraint violation (rows=configurations, columns=experiments):",
        np.array2string(mean_violation, precision=8, suppress_small=False),
        "",
        "Count constraint violation (rows=configurations, columns=experiments):",
        np.array2string(count_violation),
        "",
        "Summary across experiments:",
        (
            f"{'Configuration':<34}"
            f"{'Mean violation':>18}"
            f"{'Std violation':>18}"
            f"{'Mean count':>14}"
            f"{'Violation %':>16}"
        ),
        "-" * 100,
    ]
    for index, label in enumerate(labels):
        valid_count = int(np.sum(valid_run_mask[index]))
        lines.append(
            f"{label:<34}"
            f"{np.nanmean(mean_violation[index]):>18.8g}"
            f"{np.nanstd(mean_violation[index]):>18.8g}"
            f"{np.nanmean(count_violation[index]):>14.4f}"
            f"{np.nanmean(count_percentage[index]):>15.4f}%"
            f"  ({valid_count}/{valid_run_mask.shape[1]} valid)"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Mean constraint violation (training data):", mean_violation)
    print("Count constraint violation (training data):", count_violation)
    print(f"Constraint violation summary saved to: {output_path}")

    return mean_violation, count_violation, count_percentage


def recompute_constraint_violation_from_pkl(
    *,
    exp_type="cmp_al",
    system_name="two_tank",
    model_name="RNN",
    isNoise=1,
    isScale=1,
    isConst=1,
    exclude_nonfinite_runs=True,
    output_suffix=None,
    y_min=None,
    y_max=None,
    evaluation_max=None,
):
    """Reload a saved experiment and regenerate its violation report."""
    results = load_data_pkl(
        True,
        exp_type,
        isNoise,
        system_name=system_name,
        model_name=model_name,
        isScale=isScale,
        isConst=isConst,
    )
    if output_suffix is None:
        output_suffix = "" if isConst else "_no_const"

    return save_constraint_violation_summary(
        results["samples"],
        results["scores"],
        results["system"],
        system_name=system_name,
        exp_type=results["exp_type"],
        AL_method_set=results["AL_method_set"],
        delta_set=results["delta_set"],
        alpha_set=results["alpha_set"],
        N_train_init=results["N_train_init"],
        exclude_nonfinite_runs=exclude_nonfinite_runs,
        output_suffix=output_suffix,
        y_min_override=y_min,
        y_max_override=y_max,
        evaluation_max=evaluation_max,
    )


# @profile
def simulation(system, model, u_set, N_train_init, N_train_max, N_test = None, AL_method_set = ['passive', 'idw'], N_exp = 10, pred="jax-sysid", exp_type = 'cmp_al', qx = 0.0, qy = 0.0, AL_method = 'idw', delta_set = [1.0], alpha_set = [1.0], rho_x = 1e-3, rho_th = 1e-3, Qx_cov = 1e-10, Qy_cov = 1, Qth_cov = 1e-10, verbose = False, isScale = False, isConst = False, init_set_trainning = "jax-sysid", Ts = 1., temporality="discrete", score_interval=1, share_init_model=True, system_name=None, fixed_test_set=False, evaluation_max=None, initial_ekf_epochs=1, experiment_seeds=None, record_ekf_diagnostics=False, ekf_config_set=None, acquisition_config_set=None, noise_mode="additive", ekf_nis_gate=None, uncertainty_weight=1.0, history_refresh_interval=1):

    record_timings = _is_cmp_al_mode(exp_type) or exp_type.lower() == 'cmp_ekf'
    simulation_start = time.perf_counter() if record_timings else None

    if acquisition_config_set is not None:
        N_set = len(acquisition_config_set)
    elif _is_cmp_al_mode(exp_type):
        N_set = len(AL_method_set)
    elif exp_type.lower() in ('cmp_delta', 'cmp_delta_idwuy'):
        N_set = len(AL_method_set) + len(delta_set) - 1
    elif exp_type.lower() == 'cmp_alpha':
        N_set = len(AL_method_set) + len(alpha_set) - 1
    elif exp_type.lower() == 'cmp_idw_grid':
        N_set = len(AL_method_set) - 1 + len(delta_set) * len(alpha_set)
    elif exp_type.lower() == 'cmp_ekf':
        if ekf_config_set is None or len(ekf_config_set) == 0:
            raise ValueError("ekf_config_set is required for exp_type='cmp_ekf'")
        N_set = len(ekf_config_set)
    else:
        raise ValueError(
            "exp_type must be 'cmp_al', 'cmp_delta', "
            "'cmp_delta_idwuy', 'cmp_alpha', 'cmp_idw_grid', or 'cmp_ekf'"
        )

    u_set = vec_reshape(u_set)
    effective_N_test = N_train_max if N_test is None else N_test
    print_training_summary(
        system,
        model,
        u_set,
        system_name=system_name,
        exp_type=exp_type,
        AL_method_set=AL_method_set,
        delta_set=delta_set,
        alpha_set=alpha_set,
        N_exp=N_exp,
        N_train_init=N_train_init,
        N_train_max=N_train_max,
        N_test=effective_N_test,
        pred=pred,
        init_set_trainning=init_set_trainning,
        qx=qx,
        qy=qy,
        Ts=Ts,
        temporality=temporality,
        isScale=isScale,
        isConst=isConst,
        score_interval=score_interval,
        rho_x=rho_x,
        rho_th=rho_th,
        Qx_cov=Qx_cov,
        Qy_cov=Qy_cov,
        Qth_cov=Qth_cov,
        noise_mode=noise_mode,
        initial_ekf_epochs=initial_ekf_epochs,
    )

    if experiment_seeds is None:
        experiment_seeds = np.arange(N_exp, dtype=int)
    else:
        experiment_seeds = np.asarray(experiment_seeds, dtype=int).reshape(-1)
        if experiment_seeds.size != N_exp:
            raise ValueError(
                "experiment_seeds must contain exactly N_exp entries, "
                f"got {experiment_seeds.size} for N_exp={N_exp}"
            )
    print(f"Experiment seeds   : {experiment_seeds.tolist()}")

    # initialize
    samples_sim, scores_sim = init_results(N_set, N_exp, system.nu, system.nx, system.ny, N_train_max, effective_N_test)
    scores_sim["experiment_seeds"] = experiment_seeds.copy()
    scores_sim["run_config"] = {
        "system_name": system_name,
        "exp_type": exp_type,
        "AL_method_set": list(AL_method_set),
        "delta_set": list(delta_set),
        "alpha_set": list(alpha_set),
        "uncertainty_weight": float(uncertainty_weight),
        "history_refresh_interval": int(history_refresh_interval),
        "acquisition_config_set": deepcopy(acquisition_config_set),
        "N_exp": int(N_exp),
        "N_train_init": int(N_train_init),
        "N_train_max": int(N_train_max),
        "N_test": int(effective_N_test),
        "pred": pred,
        "init_set_trainning": init_set_trainning,
        "initial_ekf_epochs": int(initial_ekf_epochs),
        "rho_x": _config_value(rho_x),
        "rho_th": _config_value(rho_th),
        "Qx_cov": _config_value(Qx_cov),
        "Qy_cov": _config_value(Qy_cov),
        "Qth_cov": _config_value(Qth_cov),
        "noise_mode": noise_mode,
        "isScale": int(isScale),
        "isConst": int(isConst),
        "score_interval": int(score_interval),
        "experiment_seeds": experiment_seeds.tolist(),
        "ekf_config_set": ekf_config_set,
    }
    # Every experiment seed must start from the same untouched model template.
    # Methods/configurations within one seed still share the fitted initial
    # estimator through ``shared_init_cache`` below.
    model_template = deepcopy(model)

    # Experiment
    for n_exp in range(N_exp):

        experiment_seed = int(experiment_seeds[n_exp])
        np.random.seed(experiment_seed)
        seed_model = deepcopy(model_template)

        # Generate initial dataset
        init_set = dict()
        init_set['U'] = np.array([np.random.choice(u_set[:,i], N_train_init) for i in range(system.nu)]).T
        init_set['Y'], init_set['X'] = predict(system.x0, init_set['U'], system.state_fcn, system.output_fcn, system.params, qx=qx, qy=qy, return_X=True, temporality=temporality, Ts=Ts, noise_mode=noise_mode)
        # init_set['Y'], init_set['X'] = system.predict(system.x0, init_set['U'], qx, qy)

        # Generate test dataset
        test_set = dict()
        N_test = effective_N_test
        if fixed_test_set:
            # Decouple the test inputs from N_train_init so experiments with
            # different initial-set sizes are evaluated on exactly the same
            # test inputs for each repetition.
            test_rng = np.random.RandomState(100_000 + experiment_seed)
            test_set['U'] = np.array([
                test_rng.choice(u_set[:, i], N_test)
                for i in range(system.nu)
            ]).T
        else:
            test_set['U'] = np.array([np.random.choice(u_set[:,i], N_test) for i in range(system.nu)]).T
        test_set['Y'] = predict(system.x0, test_set['U'], system.state_fcn, system.output_fcn, system.params, qx=qx, qy=qy, temporality=temporality, Ts=Ts, noise_mode=noise_mode)

        shared_init_cache = None
        can_share_init_model = share_init_model and exp_type.lower() != "cmp_ekf"
        for n_set in range(N_set):
            print("Simulation        : %2d/%2d \n" % (n_exp, N_exp))
            print("Experiment        : %2d/%2d \n" % (n_set, N_set))
            np.random.seed(experiment_seed * 10)

            AL_method, delta, alpha = update_params(n_set, exp_type, AL_method_set, delta_set, alpha_set)
            run_system = system
            run_isConst = isConst
            if acquisition_config_set is not None:
                acquisition_config = acquisition_config_set[n_set]
                AL_method = acquisition_config.get("AL_method", AL_method)
                delta = acquisition_config.get("delta", delta)
                alpha = acquisition_config.get("alpha", alpha)
                run_isConst = bool(acquisition_config.get("use_constraints", isConst))
                if run_isConst:
                    run_system = deepcopy(system)
                    run_system.const = deepcopy(system.const)
                    run_system.const["flag"] = int(acquisition_config.get("constraint_flag", 1))
                    run_system.const["y_min"] = acquisition_config["y_min"]
                    run_system.const["y_max"] = acquisition_config["y_max"]
            run_rho_x = rho_x
            run_rho_th = rho_th
            run_Qx_cov = Qx_cov
            run_Qy_cov = Qy_cov
            run_Qth_cov = Qth_cov
            run_initial_ekf_epochs = initial_ekf_epochs
            if exp_type.lower() == "cmp_ekf":
                ekf_config = ekf_config_set[n_set]
                AL_method = ekf_config.get("AL_method", AL_method)
                run_rho_x = ekf_config.get("rho_x", run_rho_x)
                run_rho_th = ekf_config.get("rho_th", run_rho_th)
                run_Qx_cov = ekf_config.get("Qx_cov", run_Qx_cov)
                run_Qy_cov = ekf_config.get("Qy_cov", run_Qy_cov)
                run_Qth_cov = ekf_config.get("Qth_cov", run_Qth_cov)
                run_initial_ekf_epochs = ekf_config.get(
                    "initial_ekf_epochs", run_initial_ekf_epochs
                )
                print("EKF setting:", ekf_config.get("label", n_set))
                print(
                    "EKF params:",
                    f"rho_x={run_rho_x}",
                    f"rho_th={run_rho_th}",
                    f"Qx_cov={run_Qx_cov}",
                    f"Qy_cov={run_Qy_cov}",
                    f"Qth_cov={run_Qth_cov}",
                    f"initial_ekf_epochs={run_initial_ekf_epochs}",
                )

            print("AL_method:", AL_method)
            if AL_method.lower() in ('idw', 'idwuy'):
                print("delta:", delta)
                if AL_method.lower() == 'idw':
                    print("alpha:", alpha)

            run_start = time.perf_counter() if record_timings else None
            result = active_learning_sysid(
                run_system, seed_model, u_set, N_train_init, N_train_max, pred = pred, init_set = init_set, test_set = test_set, delta = delta, alpha = alpha, qx = qx, qy = qy, AL_method = AL_method, verbose = verbose, isScale = isScale, isConst = run_isConst, init_set_trainning = init_set_trainning, Ts = Ts, rho_x = run_rho_x, rho_th = run_rho_th, Qx_cov = run_Qx_cov, Qy_cov = run_Qy_cov, Qth_cov = run_Qth_cov, temporality = temporality, seed = experiment_seed * 10, score_interval = score_interval, init_cache = shared_init_cache if can_share_init_model else None, return_init_cache = can_share_init_model, initial_ekf_epochs = run_initial_ekf_epochs, record_ekf_diagnostics = record_ekf_diagnostics, noise_mode = noise_mode, ekf_nis_gate=ekf_nis_gate, uncertainty_weight=uncertainty_weight, history_refresh_interval=history_refresh_interval)

            if can_share_init_model:
                models, samples, scores, shared_init_cache = result
            else:
                models, samples, scores = result
            if record_timings:
                scores['timings']['total_run_time'] = (
                    time.perf_counter() - run_start
                )

            # Save results
            samples_sim, scores_sim = save_results(samples_sim, scores_sim, samples, scores, n_exp, n_set)

            del samples, scores, models
            gc.collect()

            # debug #
            # print("############################################")
            # print("scores_sim['R2_test']", scores_sim['R2_test'][n_set, n_exp, :, :])

        # Reuse compiled fixed-shape functions across methods in one experiment,
        # then bound cache growth before the next independently generated run.
        del shared_init_cache
        gc.collect()
        # Keep cache behavior aligned with the reference implementation.
        # Clearing here changes execution behavior between otherwise identical
        # configurations and is unnecessary for benchmark reproduction.
        # jax.clear_caches()

    if record_timings:
        save_timing_summary(
            scores_sim,
            system_name=system_name,
            exp_type=exp_type,
            AL_method_set=AL_method_set,
            delta_set=delta_set,
            alpha_set=alpha_set,
            total_simulation_time=time.perf_counter() - simulation_start,
        )

    if isConst and system.const is not None:
        save_constraint_violation_summary(
            samples_sim,
            scores_sim,
            system,
            system_name=system_name,
            exp_type=exp_type,
            AL_method_set=AL_method_set,
            delta_set=delta_set,
            alpha_set=alpha_set,
            N_train_init=N_train_init,
            exclude_nonfinite_runs=True,
            evaluation_max=evaluation_max,
        )

    return samples_sim, scores_sim
