"""Shared plotting and plot-data helpers for experiment scripts.

This module is intentionally system-independent. Experiment drivers provide a
small configuration object instead of embedding plotting logic for each plant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
import numpy as np

from example.analysis.figure_template import (
    create_ieee_figure_template,
    save_publication_figure,
)


DEFAULT_COLORS = (
    "#02304a",
    "#feb705",
    "#fa8600",
    "#219ebc",
    "#ba68c8",
    "#2a9d8f",
)

PANEL_TITLE_FONT_SIZE = 6.0

EXPERIMENT_FIGURE_ROOT = (
    Path(__file__).resolve().parent / "experiments" / "artifacts" / "figures"
)

METHOD_COLORS = {
    "passive": "#02304a",
    "idwuy": "#ba68c8",
    "idwuy-al": "#ba68c8",
    "gsx": "#feb705",
    "igs": "#fa8600",
    "idw": "#219ebc",
    "idw-al": "#219ebc",
}


@dataclass(frozen=True)
class ConstraintPlotConfig:
    """System-specific labels and limits for shared constraint-study plots."""

    y_min: float
    y_max: float
    artifact_prefix: str
    system_plot_name: str
    system_display_name: str
    training_ylim: tuple[float, float] | None = None
    training_yticks: tuple[float, ...] | None = None
    multiseed_ylim: tuple[float, float] | None = None
    multiseed_yticks: tuple[float, ...] | None = None
    y_axis_scale: float = 1e-2
    dense_training_plot: bool = False
    sample_time_seconds: float | None = None
    smoothing_window: int = 9


def apply_axis_multiplier(ax, scale=None):
    """Show tick values at their natural order of magnitude.

    When ``scale`` is omitted, infer it from the largest finite plotted value.
    Values in the units decade (for example, 6.0) need no multiplier, whereas
    values such as 0.6 are displayed with a ``10^-1`` multiplier.
    """
    if scale is None:
        plotted = [np.asarray(line.get_ydata(), dtype=float) for line in ax.lines]
        finite = [values[np.isfinite(values) & (values != 0)] for values in plotted]
        finite = [values for values in finite if values.size]
        if not finite:
            return ax
        maximum = max(float(np.max(np.abs(values))) for values in finite)
        exponent = int(np.floor(np.log10(maximum)))
        scale = 10.0**exponent
    else:
        exponent = int(np.floor(np.log10(scale)))

    formatter = mticker.FuncFormatter(lambda value, _: f"{value / scale:g}")
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(formatter)
    if exponent == 0:
        return ax

    ax.text(
        0.0,
        1.01,
        rf"$\times 10^{{{exponent}}}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
    )
    return ax


def finite_score_checkpoints(values, n_train_init):
    """Return scored checkpoints, including the shared initial model."""
    values = np.asarray(values)
    valid = np.any(np.isfinite(values[..., 0]), axis=(0, 1))
    valid[: max(int(n_train_init) - 1, 0)] = False
    return np.flatnonzero(valid)


def adaptive_sample_tick_interval(sample_count, fraction=0.2, base=100):
    """Choose an x-tick interval from a fraction of the largest sample count.

    The interval is rounded to the nearest multiple of ``base`` and never made
    smaller than ``base``.  For example, totals of 2000 and 1800 both produce
    a 400-sample interval.
    """
    sample_count = np.asarray(sample_count, dtype=float)
    finite = sample_count[np.isfinite(sample_count)]
    if not finite.size:
        return base
    raw_interval = fraction * float(np.max(finite))
    rounded_interval = np.floor(raw_interval / base + 0.5) * base
    return max(base, int(rounded_interval))


def apply_sample_x_limits(ax, sample_count, left_margin=0.01):
    """Set score-plot limits with 1% space before the first sample by default."""
    sample_count = np.asarray(sample_count, dtype=float)
    finite = sample_count[np.isfinite(sample_count)]
    if not finite.size:
        return ax
    first = float(np.min(finite))
    last = float(np.max(finite))
    span = max(last - first, last, 1.0)
    ax.set_xlim(first - left_margin * span, last)
    return ax


def mean_and_sem(values, axis=0):
    """Return NaN-aware mean and standard error of the mean."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    mean = np.nanmean(values, axis=axis)
    std = np.nanstd(values, axis=axis, ddof=1)
    sem = np.divide(std, np.sqrt(count), out=np.zeros_like(std), where=count > 1)
    return mean, sem


def _draw_mean_sem_curves(
    ax,
    values,
    labels,
    *,
    n_train_init,
    colors=DEFAULT_COLORS,
    shadow_alpha=0.12,
    linestyles=None,
    markers=None,
    markevery=None,
):
    """Draw mean curves and SEM bands shared by score plot types."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 4 or values.shape[-1] != 1:
        raise ValueError(
            "Score arrays must have shape "
            "(configurations, experiments, samples, 1)"
        )
    if len(labels) != values.shape[0]:
        raise ValueError(
            f"Expected {values.shape[0]} labels, received {len(labels)}"
        )

    checkpoints = finite_score_checkpoints(values, n_train_init)
    sample_count = checkpoints + 1
    for index, label in enumerate(labels):
        mean, sem = mean_and_sem(values[index][:, checkpoints, 0])
        color = colors[index % len(colors)]
        linestyle = "-" if linestyles is None else linestyles[index]
        marker = None if markers is None else markers[index]
        ax.plot(
            sample_count,
            mean,
            color=color,
            linewidth=1.15,
            linestyle=linestyle,
            marker=marker,
            markersize=2.4,
            markeredgewidth=0.5,
            markevery=markevery,
            label=label,
            zorder=3,
        )
        ax.fill_between(
            sample_count,
            mean - sem,
            mean + sem,
            color=color,
            alpha=shadow_alpha,
            linewidth=0,
            zorder=1,
        )
    return sample_count


def draw_mean_sem_rmse(
    ax,
    rmse_test,
    labels,
    *,
    n_train_init,
    colors=DEFAULT_COLORS,
    shadow_alpha=0.12,
    scale=1e-3,
    show_xlabel=True,
    yscale="linear",
    ylim=None,
    yticks=None,
    linestyles=None,
    markers=None,
    markevery=None,
):
    """Draw test-RMSE means with SEM bands on an existing axes."""
    sample_count = _draw_mean_sem_curves(
        ax,
        rmse_test,
        labels,
        n_train_init=n_train_init,
        colors=colors,
        shadow_alpha=shadow_alpha,
        linestyles=linestyles,
        markers=markers,
        markevery=markevery,
    )

    ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    apply_axis_multiplier(ax, scale)
    ax.set_ylabel("")
    ax.set_xlabel("Queried samples" if show_xlabel else "")
    if yscale == "log":
        # A narrow log-scale RMSE range can contain no interior major decade
        # ticks. Draw the minor grid as well so horizontal guides remain
        # visible, while keeping them lighter than the major grid.
        ax.grid(axis="y", which="major", color="0.85", linewidth=0.55)
        ax.grid(axis="y", which="minor", color="0.92", linewidth=0.4)
    else:
        ax.grid(axis="y", which="major", color="0.88", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    apply_sample_x_limits(ax, sample_count)
    ax.xaxis.set_major_locator(
        mticker.MultipleLocator(adaptive_sample_tick_interval(sample_count))
    )
    ax.margins(x=0)
    return ax


def draw_mean_sem_r2(
    ax,
    r2_test,
    labels,
    *,
    n_train_init,
    colors=DEFAULT_COLORS,
    shadow_alpha=0.12,
    show_xlabel=True,
    ylim=None,
    yticks=None,
    linestyles=None,
    markers=None,
    markevery=None,
):
    """Draw test-R2 means with SEM bands on an existing axes."""
    sample_count = _draw_mean_sem_curves(
        ax,
        r2_test,
        labels,
        n_train_init=n_train_init,
        colors=colors,
        shadow_alpha=shadow_alpha,
        linestyles=linestyles,
        markers=markers,
        markevery=markevery,
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.set_ylabel("")
    ax.set_xlabel("Queried samples" if show_xlabel else "")
    ax.grid(axis="y", color="0.88", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    apply_sample_x_limits(ax, sample_count)
    ax.xaxis.set_major_locator(
        mticker.MultipleLocator(adaptive_sample_tick_interval(sample_count))
    )
    ax.margins(x=0)
    return ax


def _create_mean_sem_figure(
    panels,
    *,
    score_key,
    draw_panel,
    nrows=1,
    ncols=1,
    width=None,
    aspect=None,
    sharex=False,
    sharey=False,
    hspace=None,
    wspace=0.18,
):
    """Create a publication-style mean/SEM figure for one score type."""
    if len(panels) != nrows * ncols:
        raise ValueError(
            f"The number of {score_key} panels must equal nrows * ncols"
        )
    if hspace is None:
        hspace = 0.15 if nrows > 1 else 0.12
    fig, axes = create_ieee_figure_template(
        nrows,
        ncols,
        sharex=sharex,
        sharey=sharey,
        width=width,
        aspect=aspect,
        font_size=7,
        hspace=hspace,
        wspace=wspace,
    )
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    for panel_index, (ax, panel) in enumerate(zip(axes_array, panels)):
        row = panel_index // ncols
        draw_kwargs = {
            "n_train_init": panel["n_train_init"],
            "colors": panel.get("colors", DEFAULT_COLORS),
            "shadow_alpha": panel.get("shadow_alpha", 0.12),
            "show_xlabel": row == nrows - 1,
            "ylim": panel.get("ylim"),
            "yticks": panel.get("yticks"),
            "linestyles": panel.get("linestyles"),
            "markers": panel.get("markers"),
            "markevery": panel.get("markevery"),
        }
        if score_key == "rmse_test":
            draw_kwargs.update(
                scale=panel.get("scale", 1e-3),
                yscale=panel.get("yscale", "linear"),
            )
        draw_panel(ax, panel[score_key], panel["labels"], **draw_kwargs)
        ax.text(
            panel.get("title_x", 0.16),
            1.015,
            panel["title"],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=PANEL_TITLE_FONT_SIZE,
            fontweight="bold",
        )
        if panel.get("show_legend", True):
            legend_kwargs = {
                "loc": "upper right",
                "ncol": 2,
                "frameon": True,
                "framealpha": 0.78,
                "facecolor": "white",
                "edgecolor": "none",
                "fontsize": 5.7,
                "columnspacing": 0.7,
                "handlelength": 1.7,
                "handletextpad": 0.4,
                "borderpad": 0.3,
                "labelspacing": 0.25,
            }
            legend_kwargs.update(panel.get("legend_kwargs", {}))
            ax.legend(**legend_kwargs)

    if nrows > 1:
        fig.subplots_adjust(
            left=0.11, right=0.985, bottom=0.11, top=0.96, hspace=hspace,
        )
    else:
        fig.subplots_adjust(left=0.11, right=0.985, bottom=0.17, top=0.94)
    return fig, axes


def create_mean_sem_rmse_figure(
    panels,
    *,
    nrows=1,
    ncols=1,
    width=None,
    aspect=None,
    sharex=False,
    sharey=False,
    hspace=None,
    wspace=0.18,
):
    """Create one publication-style figure from RMSE panel dictionaries."""
    return _create_mean_sem_figure(
        panels,
        score_key="rmse_test",
        draw_panel=draw_mean_sem_rmse,
        nrows=nrows,
        ncols=ncols,
        width=width,
        aspect=aspect,
        sharex=sharex,
        sharey=sharey,
        hspace=hspace,
        wspace=wspace,
    )


def create_mean_sem_r2_figure(
    panels,
    *,
    nrows=1,
    ncols=1,
    width=None,
    aspect=None,
    sharex=False,
    sharey=False,
    hspace=None,
    wspace=0.18,
):
    """Create one publication-style figure from R2 panel dictionaries."""
    return _create_mean_sem_figure(
        panels,
        score_key="r2_test",
        draw_panel=draw_mean_sem_r2,
        nrows=nrows,
        ncols=ncols,
        width=width,
        aspect=aspect,
        sharex=sharex,
        sharey=sharey,
        hspace=hspace,
        wspace=wspace,
    )


def display_method_label(label):
    """Normalize common acquisition-method labels for publication figures."""
    text = str(label).strip()
    normalized = text.lower()
    exact = {
        "passive": "Passive",
        "idwuy": "IDWuy-AL",
        "gsx": "GSx",
        "igs": "iGS",
        "idw": "IDW-AL",
        "idwz": r"IDW-AL $\beta_s=0$",
    }
    if normalized in exact:
        return exact[normalized]
    for internal, display in (
        ("idwuy", "IDWuy-AL"),
        ("idwz", "IDW-ALz"),
        ("idw", "IDW-AL"),
    ):
        if normalized.startswith(internal):
            return display + text[len(internal):]
    return text


def compact_scientific(value):
    """Format a numeric sweep value as compact scientific notation."""
    value = float(value)
    if not np.isfinite(value):
        return f"{value:g}"
    if value == 0:
        return "0e0"
    mantissa, exponent = f"{value:.12e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def score_series_labels(
    n_configurations,
    exp_type,
    al_methods,
    delta_set,
    alpha_set,
    ekf_config_set=None,
):
    """Build labels for every score-array configuration."""
    exp_type = exp_type.lower()
    al_methods = list(al_methods)
    if exp_type == "cmp_ekf" and ekf_config_set is not None:
        labels = [
            config.get("label", f"EKF {index}")
            for index, config in enumerate(ekf_config_set)
        ]
    elif exp_type == "cmp_ekf" or (
        exp_type.startswith("cmp_al") and exp_type != "cmp_alpha"
    ):
        labels = [display_method_label(label) for label in al_methods]
    elif exp_type in ("cmp_delta", "cmp_delta_idwuy"):
        swept_method = "IDWuy-AL" if exp_type == "cmp_delta_idwuy" else "IDW-AL"
        labels = [display_method_label(label) for label in al_methods[:-1]]
        labels.extend(
            rf"{swept_method} $\delta={compact_scientific(value)}$"
            for value in delta_set
        )
    elif exp_type == "cmp_alpha":
        labels = [display_method_label(label) for label in al_methods[:-1]]
        labels.extend(
            rf"IDW-AL $\alpha={compact_scientific(value)}$"
            for value in alpha_set
        )
    elif exp_type == "cmp_idw_grid":
        labels = [display_method_label(label) for label in al_methods[:-1]]
        labels.extend(
            rf"IDW-AL $\delta={compact_scientific(delta)},\ "
            rf"\alpha={compact_scientific(alpha)}$"
            for delta in delta_set
            for alpha in alpha_set
        )
    else:
        labels = [display_method_label(label) for label in al_methods]

    if len(labels) != n_configurations:
        raise ValueError(
            f"Experiment mode {exp_type!r} produced {len(labels)} labels for "
            f"{n_configurations} score configurations"
        )
    return labels


def _standard_colors(labels):
    """Use method-stable colors when labels name distinct methods."""
    base_methods = [str(label).strip().lower().split()[0] for label in labels]
    if len(set(base_methods)) != len(base_methods):
        return DEFAULT_COLORS
    return tuple(
        METHOD_COLORS.get(method, DEFAULT_COLORS[index % len(DEFAULT_COLORS)])
        for index, method in enumerate(base_methods)
    )


def _system_display_name(system_name):
    return {
        "oxidation": "Ethylene oxidation",
        "unbalanced_disk": "Unbalanced disk",
        "two_tank": "Two-tank",
        "robot_arm": "Robot arm",
    }.get(system_name.lower(), system_name.replace("_", " ").title())


def _standard_score_stem(
    metric,
    exp_type,
    system_name,
    model_name,
    is_scale,
    is_const,
    pred,
):
    stem = (
        f"{metric}_{exp_type}_{system_name}_{model_name}_"
        f"{int(bool(is_scale))}_{int(bool(is_const))}"
    )
    if pred is not None:
        stem += f"_{pred}"
    return f"{stem}_mean_sem"


def create_standard_mean_sem_score_figures(
    scores,
    *,
    n_train_init,
    evaluation_max,
    exp_type,
    artifact_exp_type,
    al_methods,
    delta_set,
    alpha_set,
    system_name,
    model_name,
    pred,
    is_scale,
    is_const,
    save=True,
    formats=("pdf",),
):
    """Create and optionally save the standard RMSE and R2 mean/SEM figures."""
    rmse_test = np.asarray(scores["rmse_test"], dtype=float)[
        ..., :evaluation_max, :
    ]
    r2_test = np.asarray(scores["R2_test"], dtype=float)[
        ..., :evaluation_max, :
    ]
    if rmse_test.shape != r2_test.shape:
        raise ValueError("RMSE and R2 score arrays must have matching shapes")
    labels = score_series_labels(
        rmse_test.shape[0], exp_type, al_methods, delta_set, alpha_set
    )
    colors = _standard_colors(labels)
    linestyles = tuple(
        "--" if str(label).strip().lower() == "passive" else "-"
        for label in labels
    )
    constraint_text = "with y constraints" if is_const else "no y constraints"
    system_display = _system_display_name(system_name)
    common_panel = {
        "labels": labels,
        "n_train_init": n_train_init,
        "colors": colors,
        "linestyles": linestyles,
        "legend_kwargs": {"ncol": 1, "handlelength": 2.4},
    }
    rmse_panel = {
        **common_panel,
        "rmse_test": rmse_test,
        "title": f"Test RMSE: {system_display} - {constraint_text}",
        "yscale": "log",
        "scale": None,
    }
    r2_panel = {
        **common_panel,
        "r2_test": r2_test,
        "title": rf"Test $R^2$: {system_display} - {constraint_text}",
        "title_x": 0.03,
        "legend_kwargs": {
            "loc": "lower right",
            "ncol": 1,
            "handlelength": 2.4,
        },
    }
    rmse_figure, _ = create_mean_sem_rmse_figure(
        [rmse_panel], width="ieee_single_rmse", aspect="short"
    )
    r2_figure, _ = create_mean_sem_r2_figure(
        [r2_panel], width="ieee_single_rmse", aspect="short"
    )

    if save:
        formats = tuple(dict.fromkeys(str(suffix).lower() for suffix in formats))
        unsupported = set(formats) - {"pdf", "png", "svg"}
        if unsupported:
            raise ValueError(
                f"Unsupported score-figure formats: {sorted(unsupported)}"
            )
        output_dir = EXPERIMENT_FIGURE_ROOT / system_name
        for metric, figure in (("RMSE", rmse_figure), ("R2", r2_figure)):
            stem = _standard_score_stem(
                metric,
                artifact_exp_type,
                system_name,
                model_name,
                is_scale,
                is_const,
                pred,
            )
            for suffix in formats:
                save_publication_figure(
                    figure,
                    output_dir / f"{stem}.{suffix}",
                    dpi=600,
                    pad_inches=0.02,
                )
    return rmse_figure, r2_figure


def save_plot_data(data, output_path):
    """Save canonical plotting data next to its figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.with_suffix(".pkl").open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def select_representative_test_prediction(
    samples,
    labels,
    *,
    method="IDW",
    output_index=0,
    run_index=None,
    experiment_seeds=None,
):
    """Select a final test prediction, using the median-RMSE run by default."""
    normalized_labels = [str(label).strip().lower() for label in labels]
    normalized_method = method.strip().lower()
    matches = [
        index for index, label in enumerate(normalized_labels)
        if label == normalized_method
    ]
    if not matches:
        matches = [
            index for index, label in enumerate(normalized_labels)
            if label.startswith(normalized_method)
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one configuration for {method!r}, found {len(matches)}"
        )

    configuration = matches[0]
    y_runs = np.asarray(samples["Y_test"], dtype=float)[
        configuration, :, :, output_index
    ]
    yhat_runs = np.asarray(samples["Yhat_test"], dtype=float)[
        configuration, :, :, output_index
    ]
    if y_runs.shape != yhat_runs.shape:
        raise ValueError("Y_test and Yhat_test must have matching shapes")

    run_rmse = np.sqrt(np.nanmean((y_runs - yhat_runs) ** 2, axis=1))
    finite_runs = np.flatnonzero(np.isfinite(run_rmse))
    if finite_runs.size == 0:
        raise ValueError(f"No finite test predictions found for {method!r}")
    median_rmse = float(np.nanmedian(run_rmse[finite_runs]))
    if run_index is None:
        run_index = int(
            finite_runs[np.argmin(np.abs(run_rmse[finite_runs] - median_rmse))]
        )
    if run_index < 0 or run_index >= y_runs.shape[0]:
        raise IndexError(
            f"run_index must be in [0, {y_runs.shape[0] - 1}], got {run_index}"
        )

    y = y_runs[run_index]
    yhat = yhat_runs[run_index]
    residual = y - yhat
    centered = y - np.nanmean(y)
    denominator = np.nansum(centered ** 2)
    r2_percent = (
        100.0 * (1.0 - np.nansum(residual ** 2) / denominator)
        if denominator > 0 else np.nan
    )
    seed = None
    if experiment_seeds is not None:
        seeds = np.asarray(experiment_seeds)
        if run_index < seeds.size:
            seed = int(seeds[run_index])

    return {
        "method": str(labels[configuration]),
        "configuration_index": configuration,
        "run_index": run_index,
        "experiment_seed": seed,
        "selection": "closest to median final test RMSE",
        "median_rmse": median_rmse,
        "rmse": float(run_rmse[run_index]),
        "r2_percent": float(r2_percent),
        "sample_index": np.arange(y.size),
        "y": y,
        "yhat": yhat,
        "residual": residual,
    }


def render_test_prediction_comparison(
    data,
    output_path,
    *,
    system_name="System",
    y_scale=1e-2,
    residual_scale=1e-3,
    max_samples=200,
):
    """Render a publication-style measured/predicted test-set comparison."""
    x = np.asarray(data["sample_index"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    yhat = np.asarray(data["yhat"], dtype=float)
    residual = np.asarray(data["residual"], dtype=float)
    if not (x.shape == y.shape == yhat.shape == residual.shape):
        raise ValueError("Prediction plot arrays must have matching shapes")
    if max_samples is not None:
        sample_count = min(max(int(max_samples), 1), x.size)
        x = x[:sample_count]
        y = y[:sample_count]
        yhat = yhat[:sample_count]
        residual = residual[:sample_count]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = create_ieee_figure_template(
        2,
        1,
        sharex=True,
        width="ieee_single_tight",
        aspect="balanced",
        font_size=7,
        hspace=0.13,
        gridspec_kw={"height_ratios": (1.65, 1.0)},
    )
    ax_output, ax_residual = axes
    palette = {
        "measured": "#2878B5",
        "predicted": "#D55E00",
        "residual": "#5B8E7D",
        "negative": "#C44E52",
        "zero": "#222222",
    }

    measured_line, = ax_output.plot(
        x,
        y,
        color=palette["measured"],
        linewidth=0.62,
        alpha=0.88,
        label=r"$y$",
        rasterized=True,
        zorder=2,
    )
    predicted_line, = ax_output.plot(
        x,
        yhat,
        color=palette["predicted"],
        linewidth=0.92,
        linestyle=(0, (4.0, 2.0)),
        alpha=0.96,
        label=r"$\hat{y}$",
        rasterized=True,
        zorder=3,
    )
    _apply_upper_left_axis_multiplier(ax_output, y_scale)
    ax_output.text(
        0.16, 1.015, f"a  Test output: {system_name} (first {x.size} samples)",
        transform=ax_output.transAxes, ha="left", va="bottom",
        fontsize=PANEL_TITLE_FONT_SIZE, fontweight="bold",
    )
    ax_output.legend(
        handles=(measured_line, predicted_line),
        loc="upper right",
        ncol=2,
        frameon=False,
        fontsize=6,
        columnspacing=0.8,
        handlelength=2.2,
        handletextpad=0.45,
    )

    residual_colors = np.where(
        residual >= 0.0, palette["residual"], palette["negative"]
    )
    ax_residual.bar(
        x,
        residual,
        width=0.82,
        color=residual_colors,
        alpha=0.82,
        edgecolor="none",
        rasterized=True,
        zorder=2,
    )
    ax_residual.axhline(
        0.0, color=palette["zero"], linewidth=0.8, zorder=3,
    )
    _apply_upper_left_axis_multiplier(ax_residual, residual_scale)
    ax_residual.text(
        0.16, 1.015, "b  Prediction residual",
        transform=ax_residual.transAxes, ha="left", va="bottom",
        fontsize=PANEL_TITLE_FONT_SIZE, fontweight="bold",
    )
    ax_residual.add_patch(
        Rectangle(
            (0.64, 0.015),
            0.345,
            0.335,
            transform=ax_residual.transAxes,
            facecolor="white",
            edgecolor="none",
            alpha=0.82,
            zorder=4,
        )
    )
    ax_residual.text(
        0.975, 0.155, "(full test)",
        transform=ax_residual.transAxes, ha="right", va="center",
        fontsize=5.2, zorder=5,
    )
    metric_rows = (
        (
            "RMSE",
            rf"{data['rmse'] / residual_scale:.3f} $\times 10^{{{int(np.log10(residual_scale))}}}$",
        ),
        (r"$R^2$", f"{data['r2_percent']:.2f}%"),
    )
    for row_y, (label, value) in zip((0.205, 0.105), metric_rows):
        ax_residual.text(
            0.700, row_y, label,
            transform=ax_residual.transAxes, ha="right", va="center",
            fontsize=6, zorder=5,
        )
        ax_residual.text(
            0.720, row_y, "=",
            transform=ax_residual.transAxes, ha="center", va="center",
            fontsize=6, zorder=5,
        )
        ax_residual.text(
            0.740, row_y, value,
            transform=ax_residual.transAxes, ha="left", va="center",
            fontsize=6, zorder=5,
        )
    ax_residual.set_xlabel("Test samples")

    for ax in axes:
        ax.set_ylabel("")
        ax.grid(axis="y", color="0.88", linewidth=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0)
    total_samples = x.size
    tick_interval = 500 if total_samples >= 1000 else max(total_samples // 4, 1)
    ax_residual.xaxis.set_major_locator(mticker.MultipleLocator(tick_interval))
    ax_residual.set_xlim(0, total_samples)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.15, top=0.95)

    export_kwargs = {"dpi": 600, "bbox_inches": "tight", "pad_inches": 0.0}
    save_publication_figure(fig, output_path, **export_kwargs)
    plt.close(fig)
    return output_path


def _rolling_median(values, window):
    """Return a centered rolling median without changing array length."""
    values = np.asarray(values, dtype=float)
    window = max(int(window), 1)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    result = np.full(values.shape, np.nan)
    for index in range(values.size):
        start = max(index - radius, 0)
        stop = min(index + radius + 1, values.size)
        local = values[start:stop]
        if np.any(np.isfinite(local)):
            result[index] = np.nanmedian(local)
    return result


def _apply_upper_left_axis_multiplier(ax, scale):
    """Scale y tick labels and place the multiplier outside the upper-left."""
    exponent = int(np.floor(np.log10(scale)))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: f"{value / scale:g}")
    )
    ax.text(
        0.0,
        1.01,
        rf"$\times 10^{{{exponent}}}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
    )


def render_dense_constraint_training_plot(
    data, output_path, config: ConstraintPlotConfig,
):
    """Render dense oscillatory outputs and signed constraint margins."""
    output_path = output_path.with_name(f"{config.artifact_prefix}_cmp_const.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = create_ieee_figure_template(
        3,
        1,
        sharex=True,
        sharey=False,
        width="ieee_single_tight",
        aspect="column_compact",
        font_size=7,
        hspace=0.16,
        gridspec_kw={"height_ratios": (1.85, 0.48, 0.48)},
    )
    ax_output, ax_penalty_margin, ax_idw_cbf_margin = axes
    x_initial = np.asarray(data["initial_sample_index"], dtype=float)
    x = np.asarray(data["queried_sample_index"], dtype=float)
    total_samples = len(x_initial) + len(x)
    y_penalty = np.asarray(data["penalty_p_output"], dtype=float)
    y_idw_cbf = np.asarray(data["penalty_idw_cbf_output"], dtype=float)
    window = config.smoothing_window

    palette = {
        "initial": "#8A8A8A",
        "penalty": "#2878B5",
        "idw_cbf": "#D55E00",
        "margin": "#5B8E7D",
        "cap": "#8064A2",
        "bound": "#222222",
        "unsafe": "#C44E52",
    }

    ax_output.plot(
        x_initial,
        data["initial_output"],
        color=palette["initial"],
        linewidth=0.65,
        alpha=0.72,
        label="Initial",
        zorder=2,
    )
    for values, color, label in (
        (y_penalty, palette["penalty"], r"Penalty $p$"),
        (y_idw_cbf, palette["idw_cbf"], r"$p^{\mathrm{idw}}+p^{\mathrm{cbf}}$"),
    ):
        ax_output.plot(
            x, values, color=color, linewidth=0.52, alpha=0.40, zorder=2,
            rasterized=True,
        )
        ax_output.plot(
            x,
            _rolling_median(values, window),
            color=color,
            linewidth=1.30,
            label=f"{label} ({window}-pt median)",
            zorder=4,
        )

    lower_actual = np.asarray(data["actual_lower_margin_bound"], dtype=float)
    upper_actual = np.asarray(data["actual_upper_margin_bound"], dtype=float)
    lower_actual_trend = _rolling_median(lower_actual, window)
    upper_actual_trend = _rolling_median(upper_actual, window)
    ax_output.fill_between(
        x,
        lower_actual,
        upper_actual,
        where=np.isfinite(lower_actual) & np.isfinite(upper_actual),
        color=palette["margin"],
        alpha=0.055,
        linewidth=0,
        zorder=0,
        rasterized=True,
    )
    ax_output.plot(
        x, lower_actual_trend, color=palette["margin"], linestyle=":",
        linewidth=1.0, alpha=0.95, label="Adaptive margin", zorder=1,
    )
    ax_output.plot(
        x, upper_actual_trend, color=palette["margin"], linestyle=":",
        linewidth=1.0, alpha=0.95, zorder=1,
    )
    ax_output.plot(
        x, data["lower_margin_cap"], color=palette["cap"], linestyle="-.",
        linewidth=0.8, label="Margin cap", zorder=1,
    )
    ax_output.plot(
        x, data["upper_margin_cap"], color=palette["cap"], linestyle="-.",
        linewidth=0.8, zorder=1,
    )
    ax_output.axhline(
        config.y_min, color=palette["bound"], linestyle="--", linewidth=0.9,
        label="Bounds", zorder=3,
    )
    ax_output.axhline(
        config.y_max, color=palette["bound"], linestyle="--", linewidth=0.9,
        zorder=3,
    )
    ax_output.axvline(x[0], color="0.55", linewidth=0.7, zorder=1)
    ax_output.set_ylabel("")
    ax_output.text(
        0.16,
        1.015,
        f"a  Training output: {config.system_display_name} - with y constraints",
        transform=ax_output.transAxes, ha="left", va="bottom",
        fontsize=PANEL_TITLE_FONT_SIZE,
        fontweight="bold",
    )
    if config.training_ylim is not None:
        ax_output.set_ylim(*config.training_ylim)
    if config.training_yticks is not None:
        ax_output.set_yticks(config.training_yticks)
    _apply_upper_left_axis_multiplier(ax_output, 1e-2)
    legend_handles, legend_labels = ax_output.get_legend_handles_labels()
    legend_order = (0, 5, 1, 2, 4, 3)
    ax_output.legend(
        handles=[legend_handles[index] for index in legend_order],
        labels=[legend_labels[index] for index in legend_order],
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        ncol=3,
        frameon=False,
        fontsize=6,
        columnspacing=0.8,
        handlelength=2.0,
        handletextpad=0.5,
        labelspacing=0.25,
    )

    margin_panels = (
        (
            ax_penalty_margin,
            y_penalty,
            palette["penalty"],
            r"Signed distance: Penalty $p$",
        ),
        (
            ax_idw_cbf_margin,
            y_idw_cbf,
            palette["idw_cbf"],
            r"Signed distance: $p^{\mathrm{idw}}+p^{\mathrm{cbf}}$",
        ),
    )
    clearances = []
    for panel_index, (ax_margin, values, color, title) in enumerate(margin_panels):
        clearance = np.minimum(values - config.y_min, config.y_max - values)
        clearances.append(clearance)
        ax_margin.bar(
            x,
            clearance,
            width=0.82,
            color=color,
            alpha=0.82,
            edgecolor="none",
            rasterized=True,
            zorder=2,
        )
        ax_margin.axhline(
            0.0, color=palette["bound"], linewidth=0.9, zorder=3,
        )
        ax_margin.axvline(x[0], color="0.55", linewidth=0.7, zorder=1)
        ax_margin.set_ylabel("")
        if panel_index == 0:
            ax_margin.text(
                0.16,
                1.035,
                "b",
                transform=ax_margin.transAxes,
                ha="left",
                va="baseline",
                fontsize=PANEL_TITLE_FONT_SIZE,
                fontweight="bold",
            )
        ax_margin.text(
            0.205,
            1.035,
            title,
            transform=ax_margin.transAxes,
            ha="left",
            va="baseline",
            fontsize=PANEL_TITLE_FONT_SIZE,
            fontweight="bold",
        )
        _apply_upper_left_axis_multiplier(ax_margin, 1e-2)

    all_clearances = np.concatenate(clearances)
    lower_limit = min(float(np.nanmin(all_clearances)) * 1.08, 0.0)
    upper_limit = max(float(np.nanmax(all_clearances)) * 1.05, 0.0)
    for index, (ax_margin, _, _, _) in enumerate(margin_panels):
        if lower_limit < 0:
            ax_margin.axhspan(
                lower_limit,
                0.0,
                color=palette["unsafe"],
                alpha=0.08,
                zorder=-1,
                label="Violation" if index == 1 else "_nolegend_",
            )
        ax_margin.set_ylim(lower_limit, upper_limit)
    if lower_limit < 0:
        ax_idw_cbf_margin.legend(
            loc="lower right",
            frameon=False,
            fontsize=5.5,
            handlelength=1.7,
            handletextpad=0.4,
        )
    ax_idw_cbf_margin.set_xlabel("Queried samples")

    for ax in axes:
        ax.grid(axis="y", color="0.88", linewidth=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.005)
    ax_idw_cbf_margin.xaxis.set_major_locator(mticker.MultipleLocator(100))
    ax_idw_cbf_margin.set_xlim(0, total_samples)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.11, top=0.96)
    export_kwargs = {"dpi": 600, "bbox_inches": "tight", "pad_inches": 0.0}
    save_publication_figure(fig, output_path, **export_kwargs)
    plt.close(fig)
    return output_path


def render_constraint_training_plot(
    data, output_path, config: ConstraintPlotConfig,
):
    """Render a single-seed constraint comparison from canonical plot data."""
    if config.dense_training_plot:
        return render_dense_constraint_training_plot(data, output_path, config)

    output_path = output_path.with_name(f"{config.artifact_prefix}_cmp_const.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = create_ieee_figure_template(width="single", aspect="short")
    x_initial = data["initial_sample_index"]
    x = data["queried_sample_index"]

    ax.plot(
        x_initial, data["initial_output"], color="0.6", linewidth=1.2,
        label="Initial trajectory", zorder=3,
    )
    for key, color, label in (
        ("penalty_p_output", "#0072B2", r"Penalty $p$"),
        (
            "penalty_idw_cbf_output",
            "#D55E00",
            r"$p^{\mathrm{idw}} + p^{\mathrm{cbf}}$",
        ),
    ):
        ax.plot(x, data[key], color=color, linewidth=1.2, label=label, zorder=3)

    ax.plot(x, data["lower_bound"], "k--", linewidth=0.9, label="Bounds")
    ax.plot(x, data["upper_bound"], "k--", linewidth=0.9)
    ax.plot(
        x, data["lower_margin_cap"], color="#009E73",
        linestyle="-.", linewidth=1.0, label="Margin cap",
    )
    ax.plot(
        x, data["upper_margin_cap"], color="#009E73",
        linestyle="-.", linewidth=1.0,
    )
    ax.plot(
        x, data["actual_lower_margin_bound"], color="#CC79A7",
        linestyle=":", linewidth=1.0, label="Actual margin", zorder=1,
    )
    ax.plot(
        x, data["actual_upper_margin_bound"], color="#CC79A7",
        linestyle=":", linewidth=1.0, zorder=1,
    )
    ax.set_xlabel("Queried Samples")
    ax.set_ylabel("")
    ax.set_title(
        f"Training output: {config.system_plot_name} (with constraints)", pad=6,
    )
    apply_axis_multiplier(ax, config.y_axis_scale)
    if config.training_ylim is not None:
        ax.set_ylim(*config.training_ylim)
    if config.training_yticks is not None:
        ax.set_yticks(config.training_yticks)
    ax.grid(True, color="0.85", linewidth=0.5)
    ax.legend(
        loc="upper right", bbox_to_anchor=(0.995, 0.995), borderaxespad=0.0,
        frameon=True, ncol=3, handlelength=1.8, columnspacing=0.9,
        labelspacing=0.3,
    )
    ax.margins(x=0.01)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.18, top=0.88)
    save_publication_figure(fig, output_path)
    save_publication_figure(fig, output_path.with_suffix(".png"))
    plt.close(fig)
    return output_path


def plot_saved_constraint_training_data(
    data_path, output_path, config: ConstraintPlotConfig,
):
    """Regenerate a constraint comparison without rerunning an experiment."""
    with data_path.open("rb") as file:
        data = pickle.load(file)
    return render_constraint_training_plot(data, output_path, config)


def build_constraint_training_plot_data(
    results, n_train_init, config: ConstraintPlotConfig,
):
    """Build the canonical plot-data dictionary from experiment results."""
    required_flags = (0, 1, 4)
    missing_flags = [flag for flag in required_flags if flag not in results]
    if missing_flags:
        raise ValueError(f"Missing comparison results for flags {missing_flags}.")

    y_without = np.asarray(results[0]["samples"]["Y_train"][:, 0], dtype=float)
    y_penalty = np.asarray(results[1]["samples"]["Y_train"][:, 0], dtype=float)
    y_idw_cbf = np.asarray(results[4]["samples"]["Y_train"][:, 0], dtype=float)
    initial_index = np.arange(n_train_init)
    queried_index = np.arange(n_train_init, y_without.size)
    queried_slice = slice(n_train_init, None)

    beta = results[4]["system"].const.get("uncertainty_beta", 1.0 / 3.0)
    margin_cap = beta * (config.y_max - config.y_min)
    actual_margin = np.asarray(
        results[4]["scores"]["confidence_margin"], dtype=float,
    )
    lower_margin = actual_margin[:, 0]
    upper_margin = actual_margin[:, 1] if actual_margin.shape[1] >= 2 else lower_margin
    plotted_lower_margin = np.full(y_idw_cbf.size, np.nan)
    plotted_upper_margin = np.full(y_idw_cbf.size, np.nan)
    margin_count = min(actual_margin.shape[0], y_idw_cbf.size - 1)
    plotted_lower_margin[1:margin_count + 1] = lower_margin[:margin_count]
    plotted_upper_margin[1:margin_count + 1] = upper_margin[:margin_count]

    return {
        "initial_sample_index": initial_index,
        "initial_output": y_without[:n_train_init],
        "queried_sample_index": queried_index,
        "penalty_p_output": y_penalty[queried_slice],
        "penalty_idw_cbf_output": y_idw_cbf[queried_slice],
        "lower_bound": np.full(queried_index.size, config.y_min),
        "upper_bound": np.full(queried_index.size, config.y_max),
        "lower_margin_cap": np.full(
            queried_index.size, config.y_min + margin_cap,
        ),
        "upper_margin_cap": np.full(
            queried_index.size, config.y_max - margin_cap,
        ),
        "actual_lower_margin_bound": (
            config.y_min + plotted_lower_margin
        )[queried_slice],
        "actual_upper_margin_bound": (
            config.y_max - plotted_upper_margin
        )[queried_slice],
    }


def plot_constraint_training_sets(
    results, n_train_init, output_path, config: ConstraintPlotConfig,
):
    """Build, save, and render a single-seed constraint comparison."""
    plot_data = build_constraint_training_plot_data(results, n_train_init, config)
    output_path = output_path.with_name(f"{config.artifact_prefix}_cmp_const.pdf")
    save_plot_data(plot_data, output_path)
    return render_constraint_training_plot(plot_data, output_path, config)


CONSTRAINT_CASE_STYLES = {
    1: ("Deterministic penalty", "#0072B2"),
    4: ("IDW margin + CBF filter", "#D55E00"),
    5: ("AERQ penalty", "#009E73"),
}


def build_constraint_case_plot_data(
    results, n_train_init, flag, config: ConstraintPlotConfig,
):
    """Build plot data for one constrained case compared with flag 0."""
    if flag not in CONSTRAINT_CASE_STYLES:
        raise ValueError(f"Unsupported constraint comparison flag: {flag}")
    missing_flags = [value for value in (0, flag) if value not in results]
    if missing_flags:
        raise ValueError(f"Missing comparison results for flags {missing_flags}.")

    baseline = np.asarray(
        results[0]["samples"]["Y_train"][:, 0], dtype=float,
    )
    case = np.asarray(
        results[flag]["samples"]["Y_train"][:, 0], dtype=float,
    )
    if baseline.shape != case.shape:
        raise ValueError("Flag 0 and constrained trajectories must have equal length.")

    queried_slice = slice(n_train_init, None)
    queried_index = np.arange(n_train_init, baseline.size)
    plotted_lower_margin = np.full(case.size, np.nan)
    plotted_upper_margin = np.full(case.size, np.nan)
    scores = results[flag].get("scores", {})
    actual_margin = np.asarray(scores.get("confidence_margin", []), dtype=float)
    if actual_margin.ndim == 1 and actual_margin.size:
        actual_margin = actual_margin[:, None]
    if actual_margin.ndim == 2 and actual_margin.shape[0]:
        lower_margin = actual_margin[:, 0]
        upper_margin = (
            actual_margin[:, 1]
            if actual_margin.shape[1] >= 2
            else lower_margin
        )
        margin_count = min(actual_margin.shape[0], case.size - 1)
        plotted_lower_margin[1:margin_count + 1] = lower_margin[:margin_count]
        plotted_upper_margin[1:margin_count + 1] = upper_margin[:margin_count]

    beta = results[flag]["system"].const.get("uncertainty_beta")
    margin_cap = (
        beta * (config.y_max - config.y_min) if beta is not None else np.nan
    )
    label, color = CONSTRAINT_CASE_STYLES[flag]
    return {
        "flag": flag,
        "case_label": label,
        "case_color": color,
        "initial_sample_index": np.arange(n_train_init),
        "initial_output": baseline[:n_train_init],
        "queried_sample_index": queried_index,
        "baseline_output": baseline[queried_slice],
        "case_output": case[queried_slice],
        "lower_bound": np.full(queried_index.size, config.y_min),
        "upper_bound": np.full(queried_index.size, config.y_max),
        "lower_margin_cap": np.full(
            queried_index.size, config.y_min + margin_cap,
        ),
        "upper_margin_cap": np.full(
            queried_index.size, config.y_max - margin_cap,
        ),
        "actual_lower_margin_bound": (
            config.y_min + plotted_lower_margin
        )[queried_slice],
        "actual_upper_margin_bound": (
            config.y_max - plotted_upper_margin
        )[queried_slice],
    }


def render_constraint_case_comparison(
    data, output_path, config: ConstraintPlotConfig,
):
    """Render one constrained training trajectory against flag 0."""
    flag = int(data["flag"])
    output_path = output_path.with_name(
        f"{config.artifact_prefix}_cmp_const_flag{flag}_vs_flag0.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = create_ieee_figure_template(width="single", aspect="short")
    x_initial = data["initial_sample_index"]
    x = data["queried_sample_index"]

    ax.plot(
        x_initial, data["initial_output"], color="0.6", linewidth=1.2,
        label="Shared initial trajectory", zorder=3,
    )
    ax.plot(
        x, data["baseline_output"], color="0.25", linewidth=1.1,
        label="No penalty (flag 0)", zorder=3,
    )
    ax.plot(
        x, data["case_output"], color=data["case_color"], linewidth=1.2,
        label=f"{data['case_label']} (flag {flag})", zorder=3,
    )
    ax.plot(x, data["lower_bound"], "k--", linewidth=0.9, label="Bounds")
    ax.plot(x, data["upper_bound"], "k--", linewidth=0.9)

    if np.any(np.isfinite(data["lower_margin_cap"])):
        ax.plot(
            x, data["lower_margin_cap"], color="#009E73", linestyle="-.",
            linewidth=1.0, label="Margin cap",
        )
        ax.plot(
            x, data["upper_margin_cap"], color="#009E73", linestyle="-.",
            linewidth=1.0,
        )
    if np.any(np.isfinite(data["actual_lower_margin_bound"])):
        ax.plot(
            x, data["actual_lower_margin_bound"], color="#CC79A7",
            linestyle=":", linewidth=1.0, label="Actual margin", zorder=1,
        )
        ax.plot(
            x, data["actual_upper_margin_bound"], color="#CC79A7",
            linestyle=":", linewidth=1.0, zorder=1,
        )

    ax.set_xlabel("Queried Samples")
    ax.set_ylabel("")
    ax.set_title(
        f"{config.system_plot_name}: flag {flag} compared with flag 0", pad=6,
    )
    apply_axis_multiplier(ax, config.y_axis_scale)
    if config.training_ylim is not None:
        ax.set_ylim(*config.training_ylim)
    if config.training_yticks is not None:
        ax.set_yticks(config.training_yticks)
    ax.grid(True, color="0.85", linewidth=0.5)
    ax.legend(
        loc="upper right", bbox_to_anchor=(0.995, 0.995), borderaxespad=0.0,
        frameon=True, ncol=2, handlelength=1.8, columnspacing=0.9,
        labelspacing=0.3,
    )
    ax.margins(x=0.01)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.18, top=0.88)
    save_publication_figure(fig, output_path)
    save_publication_figure(fig, output_path.with_suffix(".png"))
    plt.close(fig)
    return output_path


def plot_constraint_case_comparisons(
    results, n_train_init, output_path, config: ConstraintPlotConfig,
    flags=(1, 4, 5),
):
    """Save and render separate constrained-case comparisons with flag 0."""
    outputs = {}
    for flag in flags:
        plot_data = build_constraint_case_plot_data(
            results, n_train_init, flag, config,
        )
        case_output = output_path.with_name(
            f"{config.artifact_prefix}_cmp_const_flag{flag}_vs_flag0.pdf"
        )
        save_plot_data(plot_data, case_output)
        outputs[flag] = render_constraint_case_comparison(
            plot_data, case_output, config,
        )
    return outputs


def plot_saved_constraint_case_comparison(
    data_path, output_path, config: ConstraintPlotConfig,
):
    """Regenerate one flag-versus-0 figure from saved plotting data."""
    with data_path.open("rb") as file:
        plot_data = pickle.load(file)
    return render_constraint_case_comparison(plot_data, output_path, config)


def plot_constraint_one_step_prediction(
    results, n_train_init, output_path, config: ConstraintPlotConfig, flag=2,
):
    """Plot actual output and one-step prediction for one experiment case."""
    if flag not in results:
        return None
    samples = results[flag]["samples"]
    if "Y_one_step_pred_train" not in samples:
        return None

    y_actual = np.asarray(samples["Y_train"][:, 0], dtype=float)
    y_pred = np.asarray(samples["Y_one_step_pred_train"][:, 0], dtype=float)
    steps = np.arange(len(y_actual))
    finite = np.isfinite(y_pred)
    error = np.full_like(y_actual, np.nan)
    error[finite] = y_actual[finite] - y_pred[finite]

    fig, axes = create_ieee_figure_template(
        2, 1, sharex=True, width="single", aspect="balanced", hspace=0.30,
    )
    axes[0].plot(
        steps[n_train_init:], y_actual[n_train_init:], color="black",
        linewidth=1.4, label="actual y",
    )
    axes[0].plot(
        steps[finite], y_pred[finite], color="tab:orange", linestyle="--",
        linewidth=1.3, label="one-step predicted y",
    )
    axes[0].axhline(config.y_min, color="black", linestyle=":", linewidth=1.0)
    axes[0].axhline(config.y_max, color="black", linestyle=":", linewidth=1.0)
    axes[0].axvline(n_train_init - 1, color="black", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("output y")
    axes[0].legend(loc="best")

    axes[1].plot(
        steps[finite], error[finite], color="tab:red", linewidth=1.2,
        label="actual - predicted",
    )
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].axvline(n_train_init - 1, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("prediction error")
    axes[1].set_xlabel("training sample index")
    axes[1].legend(loc="best")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.14, top=0.98, hspace=0.30)

    pred_output = output_path.with_name(
        f"{output_path.stem}_case{flag}_one_step_prediction{output_path.suffix}"
    )
    save_publication_figure(fig, pred_output, dpi=250)
    plt.close(fig)
    return pred_output


def trajectory_min_max(trajectories):
    """Return columnwise extrema while preserving all-invalid columns as NaN."""
    trajectories = np.asarray(trajectories, dtype=float)
    valid = np.any(np.isfinite(trajectories), axis=0)
    minimum = np.full(trajectories.shape[1], np.nan)
    maximum = np.full(trajectories.shape[1], np.nan)
    minimum[valid] = np.nanmin(trajectories[:, valid], axis=0)
    maximum[valid] = np.nanmax(trajectories[:, valid], axis=0)
    return minimum, maximum


def render_constraint_multiseed_plot(
    plot_data, output_path, config: ConstraintPlotConfig,
):
    """Render seed trajectories, their envelope, and median for three cases."""
    flags = tuple(plot_data.get("flags", (0, 1, 4)))
    subtitles = {
        0: "Output without constraint penalty",
        1: "Output with nominal penalty",
        4: "Output with IDW--CBF penalty",
    }
    colors = {0: "#7F7F7F", 1: "#0072B2", 4: "#D55E00"}
    x_initial = np.asarray(plot_data["initial_sample_index"])
    x = np.asarray(plot_data["queried_sample_index"])
    initial_min, initial_max = trajectory_min_max(
        plot_data["initial_trajectories"],
    )
    fig, axes = create_ieee_figure_template(
        nrows=len(flags), ncols=1, sharex=True, sharey=True,
        width="single", aspect="tall" if len(flags) > 1 else "short",
        hspace=0.34,
    )
    axes = np.atleast_1d(axes)
    for ax, flag in zip(axes, flags):
        trajectories = np.asarray(plot_data[f"case{flag}_trajectories"])
        minimum, maximum = trajectory_min_max(trajectories)
        median = np.nanmedian(trajectories, axis=0)
        ax.fill_between(
            x_initial, initial_min, initial_max, color="0.65", alpha=0.18,
            linewidth=0,
        )
        ax.fill_between(
            [x_initial[-1], x[0]],
            [initial_min[-1], minimum[0]],
            [initial_max[-1], maximum[0]],
            color="0.65", alpha=0.18, linewidth=0,
        )
        for initial_trajectory, trajectory in zip(
            plot_data["initial_trajectories"], trajectories,
        ):
            ax.plot(
                x_initial, initial_trajectory, color="0.55", alpha=0.28,
                linewidth=0.65,
            )
            ax.plot(
                [x_initial[-1], x[0]],
                [initial_trajectory[-1], trajectory[0]],
                color="0.55", alpha=0.28, linewidth=0.65,
            )
        ax.fill_between(
            x, minimum, maximum, color=colors[flag], alpha=0.16,
            linewidth=0, label="Min-max range",
        )
        for trajectory in trajectories:
            ax.plot(x, trajectory, color=colors[flag], alpha=0.28, linewidth=0.65)
        ax.plot(x, median, color=colors[flag], linewidth=1.5, label="Median")
        ax.plot(
            x, np.full(x.size, config.y_min), color="black", linestyle="--",
            linewidth=0.9, label="Bounds",
        )
        ax.plot(
            x, np.full(x.size, config.y_max), color="black", linestyle="--",
            linewidth=0.9,
        )
        ax.set_title(subtitles[flag])
        if config.multiseed_ylim is not None:
            ax.set_ylim(*config.multiseed_ylim)
        if config.multiseed_yticks is not None:
            ax.set_yticks(config.multiseed_yticks)
        ax.grid(True, color="0.85", linewidth=0.5)
    axes[-1].set_xlabel("Training sample index")
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.13, top=0.92)
    save_publication_figure(fig, output_path)
    plt.close(fig)
    return output_path


def plot_saved_constraint_multiseed_data(
    data_path, output_path, config: ConstraintPlotConfig,
):
    """Regenerate a multi-seed figure without rerunning experiments."""
    with data_path.open("rb") as file:
        plot_data = pickle.load(file)
    return render_constraint_multiseed_plot(plot_data, output_path, config)


def _last_finite_score(values):
    """Return the last finite scalar in a possibly sparse score history."""
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else np.nan


def _latex_mean_std(values, scale=1.0, precision=3):
    """Format population mean and standard deviation for a LaTeX table."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)] * scale
    if values.size == 0:
        return r"$\mathrm{n/a}$"
    mean = float(np.mean(values))
    std = float(np.std(values))
    magnitude = max(abs(mean), abs(std))
    if magnitude and (magnitude < 1e-3 or magnitude >= 1e4):
        exponent = int(np.floor(np.log10(magnitude)))
        factor = 10.0**exponent
        return (
            f"$({mean / factor:.{precision}g}"
            rf"\pm{std / factor:.{precision}g})"
            rf"\!\times\!10^{{{exponent}}}$"
        )
    return f"${mean:.{precision}g}\\pm{std:.{precision}g}$"


def save_constraint_multiseed_comparison(
    all_results, seeds, n_train_init, output_dir, config: ConstraintPlotConfig,
    flags=(0, 1, 4),
):
    """Save shared multi-seed plot data, figure, and summary report."""
    flags = tuple(flags)
    labels = {
        0: "No constraint penalty",
        1: "Nominal penalty",
        4: "IDW--CBF penalty",
    }
    first_flag = flags[0]
    n_queried = (
        next(iter(all_results.values()))[first_flag]["samples"]["Y_train"].shape[0]
        - n_train_init
    )
    initial_trajectories = np.stack([
        all_results[seed][first_flag]["samples"]["Y_train"][:n_train_init, 0]
        for seed in seeds
    ])
    plot_data = {
        "seeds": np.asarray(seeds),
        "flags": np.asarray(flags),
        "initial_sample_index": np.arange(n_train_init),
        "initial_trajectories": initial_trajectories,
        "queried_sample_index": np.arange(n_train_init, n_train_init + n_queried),
    }
    lines = [
        f"{config.system_display_name} multi-seed safety comparison",
        "=" * 72,
        f"Seeds: {', '.join(map(str, seeds))}",
        "First IDW query per seed/case excluded as JAX warm-up.",
        "",
    ]

    for flag in flags:
        trajectories = np.stack([
            all_results[seed][flag]["samples"]["Y_train"][n_train_init:, 0]
            for seed in seeds
        ])
        min_y, max_y = trajectory_min_max(trajectories)
        plot_data[f"case{flag}_trajectories"] = trajectories
        plot_data[f"case{flag}_min"] = min_y
        plot_data[f"case{flag}_max"] = max_y
        counts = np.asarray([
            all_results[seed][flag]["violations"]["count"] for seed in seeds
        ], dtype=float)
        means = np.asarray([
            all_results[seed][flag]["violations"]["mean"] for seed in seeds
        ])
        maxima = np.asarray([
            all_results[seed][flag]["violations"]["max"] for seed in seeds
        ])
        failed_seeds = [
            seed for seed in seeds
            if all_results[seed][flag]["violations"]["failed"]
        ]
        idw_means = np.asarray([
            all_results[seed][flag]["scores"]["timings"]["active_learning"]["mean"]
            for seed in seeds
        ])
        violation_rates = np.asarray([
            (
                100.0 * all_results[seed][flag]["violations"]["count"]
                / all_results[seed][flag]["violations"]["valid_samples"]
            )
            if all_results[seed][flag]["violations"]["valid_samples"] else np.nan
            for seed in seeds
        ])
        final_rmse = np.asarray([
            _last_finite_score(all_results[seed][flag]["scores"]["rmse_test"])
            for seed in seeds
        ])
        plot_data[f"case{flag}_violation_count"] = counts
        plot_data[f"case{flag}_violation_rate"] = violation_rates
        plot_data[f"case{flag}_violation_mean"] = means
        plot_data[f"case{flag}_violation_max"] = maxima
        plot_data[f"case{flag}_final_rmse"] = final_rmse
        plot_data[f"case{flag}_failed_seeds"] = np.asarray(failed_seeds, dtype=int)
        plot_data[f"case{flag}_idw_mean_time"] = idw_means
        valid_means = means[np.isfinite(means)]
        valid_maxima = maxima[np.isfinite(maxima)]
        lines.extend([
            f"flag={flag} ({labels[flag]})",
            f"  numerically successful seeds: "
            f"{len(seeds) - len(failed_seeds)}/{len(seeds)}",
            "  failed seeds: "
            + (", ".join(map(str, failed_seeds)) if failed_seeds else "none"),
            f"  violation count: {counts.mean():.3f} +/- {counts.std():.3f}",
            f"  violation rate: {np.nanmean(violation_rates):.4f} "
            f"+/- {np.nanstd(violation_rates):.4f} %",
            f"  mean violation (successful seeds): "
            f"{valid_means.mean():.8g} +/- {valid_means.std():.8g}",
            f"  max violation (successful seeds): "
            f"{valid_maxima.mean():.8g} +/- {valid_maxima.std():.8g}",
            f"  final test RMSE: {np.nanmean(final_rmse):.8g} "
            f"+/- {np.nanstd(final_rmse):.8g}",
            f"  average IDW time: {idw_means.mean():.8f} "
            f"+/- {idw_means.std():.8f} s/query",
            "",
        ])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config.artifact_prefix}_cmp_const_multiseed.pdf"
    table_rows = (
        ("Violation rate [\\%]", "violation_rate", 1.0),
        ("Mean violation", "violation_mean", 1.0),
        ("Maximum violation", "violation_max", 1.0),
        ("Final test RMSE", "final_rmse", 1.0),
        ("IDW time [ms/query]", "idw_mean_time", 1000.0),
    )
    latex_headers = {
        0: r"\textbf{No penalty}",
        1: r"\textbf{Nominal $p$}",
        4: r"\textbf{IDW--CBF}",
    }
    latex_lines = [
        rf"\begin{{tabular}}{{l{'c' * len(flags)}}}",
        r"\hline",
        r"\textbf{Quantity} & " + " & ".join(
            latex_headers[flag] for flag in flags
        ) + r"\\",
        r"\hline",
    ]
    for quantity, key, scale in table_rows:
        cells = [
            _latex_mean_std(plot_data[f"case{flag}_{key}"], scale=scale)
            for flag in flags
        ]
        latex_lines.append(f"{quantity} & " + " & ".join(cells) + r"\\")
    latex_lines.extend([r"\hline", r"\end{tabular}"])
    lines.extend(["LaTeX table body", "-" * 72, *latex_lines, ""])
    save_plot_data(plot_data, output_path)
    output_path.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    return render_constraint_multiseed_plot(plot_data, output_path, config)
