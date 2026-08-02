"""Shared publication figure sizing and tightly cropped saving helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


WIDTH_MAP = {
    "ieee_single": 3.5,
    # Oversized canvas whose tight-cropped content is one IEEE column wide.
    "ieee_single_tight": 3.724,
    # Calibrated for RMSE panels with an external multiplier and long caption.
    "ieee_single_rmse": 3.68,
    "single": 4.25,
    "double": 12.0,
    "wide": 7.0,
}
ASPECT_MAP = {
    "shorter": 0.30,
    "short": 0.60,
    "balanced": 0.80,
    "tall": 0.81,
    "column_compact": 1.0,
    "column_tall": 1.34,
}


def create_ieee_figure_template(
    nrows=1,
    ncols=1,
    sharex=False,
    sharey=False,
    width=None,
    aspect=None,
    font_size=None,
    hspace=0.12,
    wspace=0.18,
    **subplot_kwargs,
):
    """Create a consistently sized publication-ready Matplotlib figure.

    When omitted, ``width`` is ``single`` for one column and ``double`` for
    two or more columns. ``aspect`` is ``short`` for one row and ``balanced``
    for two or more rows.
    """
    width = ("single" if ncols == 1 else "double") if width is None else width
    aspect = ("short" if nrows == 1 else "balanced") if aspect is None else aspect
    if width not in WIDTH_MAP:
        raise ValueError(f"Unknown figure width {width!r}; choose from {tuple(WIDTH_MAP)}")
    if aspect not in ASPECT_MAP:
        raise ValueError(f"Unknown figure aspect {aspect!r}; choose from {tuple(ASPECT_MAP)}")

    fig_w = WIDTH_MAP[width]
    fig_h = fig_w * ASPECT_MAP[aspect]
    base_font_size = 8 if font_size is None else font_size
    plt.rcParams.update(
        {
            "font.size": base_font_size,
            "axes.labelsize": base_font_size,
            "axes.titlesize": base_font_size,
            "xtick.labelsize": max(base_font_size - 1, 1),
            "ytick.labelsize": max(base_font_size - 1, 1),
            "legend.fontsize": max(base_font_size - 1, 1),
            "lines.linewidth": 1.4,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    gridspec_kw = dict(subplot_kwargs.pop("gridspec_kw", {}))
    gridspec_kw.setdefault("hspace", hspace)
    gridspec_kw.setdefault("wspace", wspace)
    return plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        sharex=sharex,
        sharey=sharey,
        gridspec_kw=gridspec_kw,
        **subplot_kwargs,
    )


def save_publication_figure(
    fig,
    output_path,
    *,
    dpi=300,
    pad_inches=0.01,
    close=False,
    **savefig_kwargs,
):
    """Save a figure with consistent tight cropping and minimal padding."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savefig_kwargs.setdefault("bbox_inches", "tight")
    savefig_kwargs.setdefault("pad_inches", pad_inches)
    # DPI also controls artists rasterized inside otherwise-vector PDF/SVG files.
    savefig_kwargs.setdefault("dpi", dpi)
    fig.savefig(output_path, **savefig_kwargs)
    if close:
        plt.close(fig)
    return output_path
