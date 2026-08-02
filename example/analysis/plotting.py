import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from activesysid.data_save_load import (
    load_data_pkl,
    save_data_pkl,
)
from activesysid.utils import replace_nan_with_previous
import matplotlib.ticker as mticker
import os
from pathlib import Path


def experiment_figures_root():
    """Return the shared experiment-artifact figure root."""
    return Path(__file__).resolve().parents[1] / "experiments" / "artifacts" / "figures"


def _is_cmp_al_mode(exp_type):
    """Return True for cmp_al variants (for example cmp_al5), but not cmp_alpha."""
    exp_type = exp_type.lower()
    return exp_type == "cmp_al" or (
        exp_type.startswith("cmp_al") and exp_type != "cmp_alpha"
    )

# Helper function to format delta without leading zeros in exponent
def format_sci_nota(delta):
    formatted = f"{delta:.0e}"  # '1e-03'
    # Remove the leading zero in the exponent
    if 'e-0' in formatted:
        formatted = formatted.replace('e-0', 'e-')
    elif 'e+0' in formatted:
        formatted = formatted.replace('e+0', 'e+')
    return formatted


def _method_labels(methods):
    """Use explicit labels for the IDW-AL uncertainty ablation."""
    if any(str(method).lower() == "idwz" for method in methods):
        return [
            r"IDW-AL $\beta_s=0$" if str(method).lower() == "idwz"
            else r"IDW-AL $\beta_s=1$" if str(method).lower() == "idw"
            else method
            for method in methods
        ]
    return methods


def _paired_qy_styles(labels):
    """Pair passive/IDW-AL curves by Qy: color encodes Qy, line style encodes method."""
    parsed = []
    for label in labels:
        parts = str(label).lower().rsplit("_qy_", 1)
        if len(parts) != 2 or parts[0] not in {"passive", "idw"}:
            return None
        parsed.append((parts[0], parts[1]))
    qy_values = list(dict.fromkeys(qy for _, qy in parsed))
    cmap = colormaps.get_cmap("tab10")
    color_by_qy = {qy: cmap.colors[index] for index, qy in enumerate(qy_values)}
    colors = [color_by_qy[qy] for _, qy in parsed]
    linestyles = ["-" if method == "passive" else "--" for method, _ in parsed]
    display_labels = [
        rf"{method.upper() if method == 'idw' else method.capitalize()} $Q_y={qy}$"
        for method, qy in parsed
    ]
    return colors, linestyles, display_labels

import matplotlib.pyplot as plt

def evaluated_score_index(score_data, index):
    """
    Keep only queried-sample indices that contain at least one finite score.

    This is useful when active_learning_sysid(..., score_interval > 1) leaves
    skipped score locations as NaN.
    """
    index = np.asarray(index, dtype=int)
    valid = np.any(np.isfinite(score_data[:, :, index, :]), axis=(0, 1, 3))
    return index[valid]

def plot_rmse_with_constraints(
        rmse_test_no_const, rmse_test_const, N_set, N_exp, AL_method_set, index,
        save=False, exp_type="cmp_al", system_name="", model_name="", delta_set=[1.0], alpha_set=[1.0], pred=None, isScale=0
):
    """
    Plot RMSE with error bars for different active learning methods,
    showing both without and with constraints in two subplots (2x1).
    """
    index = evaluated_score_index(rmse_test_no_const, index)
    # Helper to process RMSE data (replace NaN, sort, etc.)
    def process_rmse(rmse_test):
        for i in np.arange(N_exp):
            for j in np.arange(N_set):
                rmse_test[j, i] = replace_nan_with_previous(rmse_test[j, i])
        sort_rmse = np.sort(rmse_test, axis=1)
        neg_index = max(0, int(0.1 * N_exp))
        pos_index = N_exp - neg_index - 1
        return sort_rmse, neg_index, pos_index

    sort_rmse_no_const, neg_index, pos_index = process_rmse(rmse_test_no_const)
    sort_rmse_const, _, _ = process_rmse(rmse_test_const)

    # Label logic (same as plot_rmse)
    if _is_cmp_al_mode(exp_type) or exp_type.lower() == 'cmp_ekf':
        labels = _method_labels(AL_method_set)
    elif exp_type.lower() == 'cmp_delta':
        labels = [AL_method for AL_method in AL_method_set[:-1]] + [rf"$\delta$ = {format_sci_nota(delta)}" for delta in delta_set]
        if N_set < len(delta_set):
            N_set = len(AL_method_set[:-1]) + len(delta_set)
    elif exp_type.lower() == 'cmp_delta_idwuy':
        labels = [AL_method for AL_method in AL_method_set[:-1]] + [rf"IDWuy-AL $\delta$ = {format_sci_nota(delta)}" for delta in delta_set]
        if N_set < len(delta_set):
            N_set = len(AL_method_set[:-1]) + len(delta_set)
    elif exp_type.lower() == 'cmp_alpha':
        labels = AL_method_set[:-1] + [rf"$\alpha$ = {format_sci_nota(alpha)}" for alpha in alpha_set]
        if N_set < len(alpha_set):
            N_set = len(AL_method_set[:-1]) + len(alpha_set)
    else:
        raise ValueError("exp_type must be 'cmp_al', 'cmp_ekf', 'cmp_delta', 'cmp_delta_idwuy', or 'cmp_alpha'")

    # Colors
    if N_set <= 5:
        colors0_rgb = [[2,48,74],[254,183,5],[250,134,0],[33,158,188],[186,104,200]]
        cols0 = [tuple(c/255 for c in rgb) for rgb in colors0_rgb]
        colors1_rgb = [[14,96,107],[255,194,75],[246,111,105],[21,151,165],[186, 104, 200]]
        cols1 = [tuple(c/255 for c in rgb) for rgb in colors1_rgb]
        colors2_rgb = [[2,48,74],[254,183,5],[246,111,105],[33,158,188],[186,104,200]]
        cols = [tuple(c/255 for c in rgb) for rgb in colors2_rgb]
    else:
        cmap = plt.colormaps.get_cmap('tab20')
        cols = cmap.colors

    # Change from 1x2 to 2x1 subplots
    fig, axs = plt.subplots(2, 1, figsize=(7, 8), sharex=True, sharey=True)
    if system_name.upper() == "OXIDATION":
        SYS_NAME = "Ethylene Oxidation"
    elif system_name.upper() == "TWO_TANK":
        SYS_NAME = "Two Tank"
    elif system_name.upper() == "ROBOT_ARM":
        SYS_NAME = "Robot Arm"
    else:
        SYS_NAME = system_name.upper()
    titles = [f"RMSE - {SYS_NAME} - {model_name.upper()} (Without Constraints)", 
              f"RMSE - {SYS_NAME} - {model_name.upper()} (With Constraints)"]
    rmse_sets = [sort_rmse_no_const, sort_rmse_const]

    for ax, rmse_set, title, isScale in zip(axs, rmse_sets, titles, [0, 1]):
        for i in range(N_set):
            M = np.mean(rmse_set[i, neg_index:pos_index+1, index, 0], axis=1)
            # Plot RMSE on a semilogarithmic scale
            # when isConst == 1, use dashed line for the first method
            if i == 0 and isScale == 1:
                ax.semilogy(index, M, linewidth=2.0, linestyle='--', color=cols[i])
            else:
                ax.semilogy(index, M, linewidth=2.0, color=cols[i])
            # Error bars
            index2_step = max(len(index) // 4, 1)
            index2_init = max(len(index) // 15, 1)
            index2 = np.arange(index2_init * i + index2_step, len(index) - 1, index2_step)
            neg = np.abs(rmse_set[i, neg_index, index].reshape(-1) - M)
            pos = np.abs(rmse_set[i, pos_index, index].reshape(-1) - M)
            error_bar = ax.errorbar(index[index2], M[index2],
                                   yerr=[neg[index2], pos[index2]],
                                   ecolor=cols[i], fmt='x', color=cols[i],
                                   capsize=5, capthick=2, linestyle='None')
            try:
                bars = error_bar.lines[2]
                for bar in bars:
                    bar.set_linestyle('--')
                    bar.set_linewidth(1.5)
            except IndexError:
                pass
        ax.set_title(title)
        ax.legend(labels, loc='best')
        # ax.set_ylabel('RMSE (log scale)')
        ax.grid(True, which='both', axis='both')
    axs[1].set_xlabel('Queried Samples')
    # axs[1].legend(labels, loc='best')
    # plt.suptitle(f"RMSE Comparison - {system_name.upper()} - {model_name.upper()}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        fig_dir = f"{system_name}" if system_name != "" else ''
        fig_name = f"RMSE_cmp_constraint_{exp_type}_{system_name}_{model_name}_{isScale}"
        if pred is not None:
            fig_name += f"_{pred}"
        save_plot(fig, fig_name, fig_dir)

    plt.show()
    return fig

def plot_rmse(
        rmse_test, N_set, N_exp, AL_method_set, index, save=False, exp_type="cmp_al", system_name = "", model_name = "", delta_set = [1.0], alpha_set = [1.0], pred = None, isScale = 0, isConst = 0
):
    """
    Plot RMSE with error bars for different active learning methods.

    Parameters:
        rmse_test (np.ndarray): RMSE test results with shape (N_set, N_exp, len(index), 1).
        N_set (int): Number of active learning methods.
        N_exp (int): Number of experiments.
        AL_method_set (list): List of active learning method names.
        index (np.ndarray): Array of queried sample indices.
        exp_type (str): Experiment type (default: "cmp_al").

    Returns:
        fRMSE (plt.Figure): Figure object for the RMSE plot.
    """

    index = evaluated_score_index(rmse_test, index)

    # Replace NaN values with the previous value in the array
    for i in np.arange(N_exp):
        for j in np.arange(N_set):
            rmse_test[j,i] = replace_nan_with_previous(rmse_test[j,i])

    sort_rmse = np.sort(rmse_test, axis=1)
    neg_index = max(0, int(0.1 * N_exp))
    pos_index = N_exp - neg_index - 1

    index2_step = max(len(index) // 4, 1)
    index2_init = max(len(index) // 15, 1)

    # Set the y-axis label based on the system type
    if _is_cmp_al_mode(exp_type) or exp_type.lower() == 'cmp_ekf':
        labels = _method_labels(AL_method_set)
    elif exp_type.lower() == 'cmp_delta':
        # Create labels where only delta_set items are prefixed with 'delta ='
        # labels = AL_method_set[:-1] + [f"$\delta$ = {delta:.0e}" for delta in delta_set]

        # Create combined labels: AL_method_set labels remain unchanged, delta_set labels get 'delta =' prefix
        # labels = AL_method_set[:-1] + [f"$\delta$ = {format_sci_nota(delta)}" for delta in delta_set]

        labels = [AL_method for AL_method in AL_method_set[:-1]] + [rf"$\delta$ = {format_sci_nota(delta)}" for delta in delta_set]

        if N_set < len(delta_set):
            N_set = len(AL_method_set[:-1]) + len(delta_set)
    elif exp_type.lower() == 'cmp_delta_idwuy':
        labels = [AL_method for AL_method in AL_method_set[:-1]] + [rf"IDWuy-AL $\delta$ = {format_sci_nota(delta)}" for delta in delta_set]
        if N_set < len(delta_set):
            N_set = len(AL_method_set[:-1]) + len(delta_set)
    elif exp_type.lower() == 'cmp_alpha':
        # Create combined labels: AL_method_set labels remain unchanged, delta_set labels get 'delta =' prefix
        labels = AL_method_set[:-1] + [rf"$\alpha$ = {format_sci_nota(alpha)}" for alpha in alpha_set]
        if N_set < len(alpha_set):
            N_set = len(AL_method_set[:-1]) + len(alpha_set)
    elif exp_type.lower() == 'cmp_idw_grid':
        labels = AL_method_set[:-1] + [
            rf"$\delta$ = {format_sci_nota(delta)}, $\alpha$ = {format_sci_nota(alpha)}"
            for delta in delta_set
            for alpha in alpha_set
        ]
        N_set = len(labels)
    else:
        raise ValueError(
            "exp_type must be 'cmp_al', 'cmp_ekf', 'cmp_delta', "
            "'cmp_delta_idwuy', 'cmp_alpha', or 'cmp_idw_grid'"
        )

    # Use a colormap with enough distinct colors (e.g., 'tab20' provides 20 distinct colors)
    if N_set <= 5:
        colors0_rgb = [[2,48,74],
                    [254,183,5],
                    [250,134,0],
                    [33,158,188],
                    [186, 104, 200]]
        cols = [tuple(c/255 for c in rgb) for rgb in colors0_rgb]

        colors1_rgb = [[38,70,83],
                    [230,111,81],
                    [42,157,142],
                    [243,162,97],
                    [186, 104, 200]]
        colors1 = [tuple(c/255 for c in rgb) for rgb in colors1_rgb]

        colors_rgb = [[14,96,107],
                    [255,194,75],
                    [246,111,105],
                    [21,151,165],
                    [186, 104, 200]]
        cols2 = [tuple(c/255 for c in rgb) for rgb in colors_rgb]
    else:
        cmap = colormaps.get_cmap('tab20')
        cols = cmap.colors  # This will have 20 colors
    paired_qy = _paired_qy_styles(labels)
    line_styles = ["-"] * N_set
    if paired_qy is not None:
        cols, line_styles, labels = paired_qy
    
    fRMSE = plt.figure(figsize=(8, 4))

    for i in range(N_set):
        # Compute mean RMSE
        M = np.mean(sort_rmse[i, neg_index:pos_index+1, index, 0], axis=1)

        # Plot RMSE on a semilogarithmic scale

        # plt.semilogy(index, M, linewidth=2.0, label=AL_method_set[i], color=cols[i])
        if isConst == 1:
            if i == 0:
                plt.semilogy(index, M, linewidth=2.0, linestyle='--', color=cols[i])
            else:
                plt.semilogy(index, M, linewidth=2.0, color=cols[i])
        else:
            plt.semilogy(index, M, linewidth=2.0, color=cols[i], linestyle=line_styles[i])
        # plt.semilogy(index, M, linewidth=2.0)

        # Determine indices for error bars
        index2 = np.arange(index2_init * i + index2_step, len(index) - 1, index2_step)

        # Calculate negative and positive errors
        neg = np.abs(sort_rmse[i, neg_index, index].reshape(-1) - M)
        pos = np.abs(sort_rmse[i, pos_index, index].reshape(-1) - M)

        # Plot error bars with markers only (no connecting lines)
        error_bar = plt.errorbar(index[index2], M[index2],
            yerr=[neg[index2], pos[index2]], 
            ecolor=cols[i],
            fmt='x',
            color=cols[i],
            capsize=5,
            capthick=2,
            linestyle='None'
        )

        # Customize the error bar vertical lines to be dashed
        # error_bar contains (data_line, caplines, barlines)
        # Access the vertical error bars
        try:
            bars = error_bar.lines[2]  # The third element contains the error bar lines
            for bar in bars:
                bar.set_linestyle('--')
                bar.set_linewidth(1.5)  # Optional: adjust line width
        except IndexError:
            print(f"Warning: Unable to access error bar lines for method '{AL_method_set[i]}'.")

    # Add labels, title, legend, and grid
    title = f"RMSE - {system_name.upper()} - {model_name.upper()}"
    plt.title(title)
    plt.xlabel('Queried Samples')
    plt.yscale('log')

    plt.legend(labels, loc='best')
    plt.grid(True, which='both', axis='both')
    plt.tight_layout()
    
    # Save the figure
    if save:
        if system_name != "":
            fig_dir = f"{system_name}"
        else:
            fig_dir = ''
        fig_name = f"RMSE_{exp_type}_{system_name}_{model_name}_{isScale}_{isConst}"
        if pred is not None:
            fig_name += f"_{pred}"
        save_plot(fRMSE, fig_name, fig_dir)

    plt.show()
    return fRMSE

def plot_r2(
        r2_test, N_set, N_exp, AL_method_set, index, save=False, exp_type="cmp_al", system_name = "", model_name = "", delta_set = [1.0], alpha_set = [1.0], pred = None, isScale = 0, isConst = 0
):
    """
    Plot R2 with error bars for different active learning methods.

    Parameters:
        r2_test (np.ndarray): R2 test results with shape (N_set, N_exp, len(index), 1).
        N_set (int): Number of active learning methods.
        N_exp (int): Number of experiments.
        AL_method_set (list): List of active learning method names.
        index (np.ndarray): Array of queried sample indices.
        exp_type (str): Experiment type (default: "cmp_al").

    Returns:
        fR2 (plt.Figure): Figure object for the R2 plot.
    """

    index = evaluated_score_index(r2_test, index)

    # Replace NaN values with the previous value in the array
    for i in np.arange(N_exp):
        for j in np.arange(N_set):

            r2_test[j,i] = replace_nan_with_previous(r2_test[j,i])

    sort_r2 = np.sort(r2_test, axis=1)
    neg_index = max(0, int(0.1 * N_exp))
    pos_index = N_exp - neg_index - 1

    index2_step = max(len(index) // 4, 1)
    index2_init = max(len(index) // 15, 1)

    # Set the y-axis label based on the system type
    if _is_cmp_al_mode(exp_type) or exp_type.lower() == 'cmp_ekf':
        labels = _method_labels(AL_method_set)
    elif exp_type.lower() == 'cmp_delta':
        # Create labels where only delta_set items are prefixed with 'delta ='
        # labels = AL_method_set[:-1] + [f"$\delta$ = {delta:.0e}" for delta in delta_set]

        # Create combined labels: AL_method_set labels remain unchanged, delta_set labels get 'delta =' prefix
        # labels = AL_method_set[:-1] + [f"$\delta$ = {format_sci_nota(delta)}" for delta in delta_set]
        labels = [AL_method for AL_method in AL_method_set[:-1]] + [rf"$\delta$ = {format_sci_nota(delta)}" for delta in delta_set]

        if N_set < len(delta_set):
            N_set = len(AL_method_set[:-1]) + len(delta_set)
    elif exp_type.lower() == 'cmp_delta_idwuy':
        labels = [AL_method for AL_method in AL_method_set[:-1]] + [rf"IDWuy-AL $\delta$ = {format_sci_nota(delta)}" for delta in delta_set]
        if N_set < len(delta_set):
            N_set = len(AL_method_set[:-1]) + len(delta_set)
    elif exp_type.lower() == 'cmp_alpha':
        # Create combined labels: AL_method_set labels remain unchanged, delta_set labels get 'delta =' prefix
        labels = AL_method_set[:-1] + [f"$\\alpha$ = {format_sci_nota(alpha)}" for alpha in alpha_set]
        if N_set < len(alpha_set):
            N_set = len(AL_method_set[:-1]) + len(alpha_set)
    elif exp_type.lower() == 'cmp_idw_grid':
        labels = AL_method_set[:-1] + [
            rf"$\delta$ = {format_sci_nota(delta)}, $\alpha$ = {format_sci_nota(alpha)}"
            for delta in delta_set
            for alpha in alpha_set
        ]
        N_set = len(labels)
    else:
        raise ValueError(
            "exp_type must be 'cmp_al', 'cmp_ekf', 'cmp_delta', "
            "'cmp_delta_idwuy', 'cmp_alpha', or 'cmp_idw_grid'"
        )

    # Use a colormap with enough distinct colors (e.g., 'tab20' provides 20 distinct colors)
    if N_set <= 5:
        colors0_rgb = [[2,48,74],
                    [254,183,5],
                    [250,134,0],
                    [33,158,188],
                    [186, 104, 200]]
        cols = [tuple(c/255 for c in rgb) for rgb in colors0_rgb]

        colors1_rgb = [[38,70,83],
                    [230,111,81],
                    [42,157,142],
                    [243,162,97],
                    [186, 104, 200]]
        colors1 = [tuple(c/255 for c in rgb) for rgb in colors1_rgb]

        colors_rgb = [[14,96,107],
                    [255,194,75],
                    [246,111,105],
                    [21,151,165],
                    [186, 104, 200]]
        cols2 = [tuple(c/255 for c in rgb) for rgb in colors_rgb]
    else:
        cmap = colormaps.get_cmap('tab20')
        cols = cmap.colors  # This will have 20 colors
    paired_qy = _paired_qy_styles(labels)
    line_styles = ["-"] * N_set
    if paired_qy is not None:
        cols, line_styles, labels = paired_qy

    fR2 = plt.figure(figsize=(8, 4))

    for i in range(N_set):
        # Compute mean R2
        M = np.mean(sort_r2[i, neg_index:pos_index+1, index, 0], axis=1)

        # Plot R2 on a semilogarithmic scale

        # plt.semilogy(index, M, linewidth=2.0, label=AL_method_set[i], color=cols[i])
        if isConst == 1:
            if i == 0:
                plt.plot(index, M, linewidth=2.0, color=cols[i], linestyle='--')
            else:
                plt.plot(index, M, linewidth=2.0, color=cols[i])
                    
        else:
            plt.plot(index, M, linewidth=2.0, color=cols[i], linestyle=line_styles[i])

        # Determine indices for error bars
        index2 = np.arange(index2_init * i + index2_step, len(index) - 1, index2_step)

        # Calculate negative and positive errors
        neg = np.abs(sort_r2[i, neg_index, index].reshape(-1) - M)
        pos = np.abs(sort_r2[i, pos_index, index].reshape(-1) - M)

        # Plot error bars with markers only (no connecting lines)
        error_bar = plt.errorbar(index[index2], M[index2],
            yerr=[neg[index2], pos[index2]], 
            ecolor=cols[i],
            fmt='x',
            color=cols[i],
            capsize=5,
            capthick=2,
            linestyle='None'
        )

        # Customize the error bar vertical lines to be dashed
        # error_bar contains (data_line, caplines, barlines)
        # Access the vertical error bars
        try:
            bars = error_bar.lines[2]  # The third element contains the error bar lines
            for bar in bars:
                bar.set_linestyle('--')
                bar.set_linewidth(1.5)  # Optional: adjust line width
        except IndexError:
            print(f"Warning: Unable to access error bar lines for method '{AL_method_set[i]}'.")

    # Set y-axis to display percentages and remove scientific notation
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    # Add labels, title, legend, and grid
    title = f"R2 - {system_name.upper()} - {model_name.upper()}"
    plt.title(title)
    plt.xlabel('Queried Samples')

    plt.legend(labels, loc='best')
    plt.grid(True, which='both', axis='both')
    plt.tight_layout()
    
    # Save the figure
    if save:
        if system_name != "":
            fig_dir = f"{system_name}"
        else:
            fig_dir = ''
        fig_name = f"R2_{exp_type}_{system_name}_{model_name}_{isScale}_{isConst}"
        if pred is not None:
            fig_name += f"_{pred}"
        save_plot(fR2, fig_name, fig_dir)

    plt.show()
    return fR2

def plot_bfr(
        bfr_test, N_set, N_exp, AL_method_set, index, save=False, exp_type="cmp_al", system_name = "", model_name = "", delta_set = [1.0], alpha_set = [1.0], pred = None, isScale = 0, isConst = 0
):
    """
    Plot BFR with error bars for different active learning methods.

    Parameters:
        bfr_test (np.ndarray): BFR test results with shape (N_set, N_exp, len(index), 1).
        N_set (int): Number of active learning methods.
        N_exp (int): Number of experiments.
        AL_method_set (list): List of active learning method names.
        index (np.ndarray): Array of queried sample indices.
        exp_type (str): Experiment type (default: "cmp_al").

    Returns:
        fBFR (plt.Figure): Figure object for the BFR plot.
    """

    index = evaluated_score_index(bfr_test, index)

    # Replace NaN values with the previous value in the array
    for i in np.arange(N_exp):
        for j in np.arange(N_set):

            bfr_test[j,i] = replace_nan_with_previous(bfr_test[j,i])

    sort_bfr = np.sort(bfr_test, axis=1)
    neg_index = max(0, int(0.1 * N_exp))
    pos_index = N_exp - neg_index - 1

    index2_step = max(len(index) // 4, 1)
    index2_init = max(len(index) // 15, 1)

    # Set the y-axis label based on the system type
    if _is_cmp_al_mode(exp_type) or exp_type.lower() == 'cmp_ekf':
        labels = _method_labels(AL_method_set)
    elif exp_type.lower() == 'cmp_delta':
        # Create labels where only delta_set items are prefixed with 'delta ='
        # labels = AL_method_set[:-1] + [f"$\delta$ = {delta:.0e}" for delta in delta_set]

        # Create combined labels: AL_method_set labels remain unchanged, delta_set labels get 'delta =' prefix
        # labels = AL_method_set[:-1] + [f"$\delta$ = {format_sci_nota(delta)}" for delta in delta_set]
        labels = [AL_method for AL_method in AL_method_set[:-1]] + [rf"$\delta$ = {format_sci_nota(delta)}" for delta in delta_set]

        if N_set < len(delta_set):
            N_set = len(AL_method_set[:-1]) + len(delta_set)
    elif exp_type.lower() == 'cmp_delta_idwuy':
        labels = [AL_method for AL_method in AL_method_set[:-1]] + [rf"IDWuy-AL $\delta$ = {format_sci_nota(delta)}" for delta in delta_set]
        if N_set < len(delta_set):
            N_set = len(AL_method_set[:-1]) + len(delta_set)
    elif exp_type.lower() == 'cmp_alpha':
        # Create combined labels: AL_method_set labels remain unchanged, delta_set labels get 'delta =' prefix
        labels = AL_method_set[:-1] + [f"$\\alpha$ = {format_sci_nota(alpha)}" for alpha in alpha_set]
        if N_set < len(alpha_set):
            N_set = len(AL_method_set[:-1]) + len(alpha_set)
    elif exp_type.lower() == 'cmp_idw_grid':
        labels = AL_method_set[:-1] + [
            rf"$\delta$ = {format_sci_nota(delta)}, $\alpha$ = {format_sci_nota(alpha)}"
            for delta in delta_set
            for alpha in alpha_set
        ]
        N_set = len(labels)
    else:
        raise ValueError(
            "exp_type must be 'cmp_al', 'cmp_ekf', 'cmp_delta', "
            "'cmp_delta_idwuy', 'cmp_alpha', or 'cmp_idw_grid'"
        )

    # Use a colormap with enough distinct colors (e.g., 'tab20' provides 20 distinct colors)
    if N_set <= 5:
        colors0_rgb = [[2,48,74],
                    [254,183,5],
                    [250,134,0],
                    [33,158,188],
                    [186, 104, 200]]
        cols = [tuple(c/255 for c in rgb) for rgb in colors0_rgb]

        # colors1_rgb = [[38,70,83],
        #             [230,111,81],
        #             [42,157,142],
        #             [243,162,97],
        #             [186, 104, 200]]
        # colors1 = [tuple(c/255 for c in rgb) for rgb in colors1_rgb]

        colors_rgb = [[14,96,107],
                    [255,194,75],
                    [246,111,105],
                    [21,151,165],
                    [186, 104, 200]]
        cols2 = [tuple(c/255 for c in rgb) for rgb in colors_rgb]
    else:
        cmap = colormaps.get_cmap('tab20')
        cols = cmap.colors  # This will have 20 colors

    fBFR = plt.figure(figsize=(8, 4))

    for i in range(N_set):
        # Compute mean BFR
        M = np.mean(sort_bfr[i, neg_index:pos_index+1, index, 0], axis=1)

        # Plot BFR on a semilogarithmic scale

        # plt.semilogy(index, M, linewidth=2.0, label=AL_method_set[i], color=cols[i])
        if isConst == 1:
            if i == 0:
                plt.plot(index, M, linewidth=2.0, color=cols[i], linestyle='--')
            else:
                plt.plot(index, M, linewidth=2.0, color=cols[i])
                    
        else:
            plt.plot(index, M, linewidth=2.0, color=cols[i])

        # Determine indices for error bars
        index2 = np.arange(index2_init * i + index2_step, len(index) - 1, index2_step)

        # Calculate negative and positive errors
        neg = np.abs(sort_bfr[i, neg_index, index].reshape(-1) - M)
        pos = np.abs(sort_bfr[i, pos_index, index].reshape(-1) - M)

        # Plot error bars with markers only (no connecting lines)
        error_bar = plt.errorbar(index[index2], M[index2],
            yerr=[neg[index2], pos[index2]], 
            ecolor=cols[i],
            fmt='x',
            color=cols[i],
            capsize=5,
            capthick=2,
            linestyle='None'
        )

        # Customize the error bar vertical lines to be dashed
        # error_bar contains (data_line, caplines, barlines)
        # Access the vertical error bars
        try:
            bars = error_bar.lines[2]  # The third element contains the error bar lines
            for bar in bars:
                bar.set_linestyle('--')
                bar.set_linewidth(1.5)  # Optional: adjust line width
        except IndexError:
            print(f"Warning: Unable to access error bar lines for method '{AL_method_set[i]}'.")

    # Set y-axis to display percentages and remove scientific notation
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    # Add labels, title, legend, and grid
    title = f"BFR - {system_name.upper()} - {model_name.upper()}"
    plt.title(title)
    plt.xlabel('Queried Samples')

    plt.legend(labels, loc='best')
    plt.grid(True, which='both', axis='both')
    plt.tight_layout()
    
    # Save the figure
    if save:
        if system_name != "":
            fig_dir = f"{system_name}"
        else:
            fig_dir = ''
        fig_name = f"BFR_{exp_type}_{system_name}_{model_name}_{isScale}_{isConst}"
        if pred is not None:
            fig_name += f"_{pred}"
        save_plot(fBFR, fig_name, fig_dir)

    plt.show()
    return fBFR

def plot_Y(Y_train, Yhat_train, Y_test, Yhat_test, N_set, N_exp, AL_method_set, save=False, exp_type="cmp_al", system_name = "", model_name = "", delta_set = [1.0], alpha_set = [1.0], pred = None, isScale = 1, isConst = 0, Ts = 1.0, y_min = 0.0, y_max = 1.0):
    # for AL_method in AL_method_set:

    N_train_max = Y_train.shape[2]
    N_test = Y_test.shape[2]
    T_train = np.arange(N_train_max)*Ts
    T_test = np.arange(N_test)*Ts

    for i in np.arange(N_set):
        AL_method = AL_method_set[i]

        fig, ax = plt.subplots(2, 1, figsize=(8, 6))

        m = N_train_max
        ax[0].plot(T_train[0:m], Y_train[i, -1, 0:m, 0], label='measured')
        ax[0].plot(T_train[0:m], Yhat_train[i, -1, 0:m, 0], label='estimated', linestyle='--')
        if isConst == 1:
            ax[0].plot(T_train[0:m], np.ones((m))*y_min, linestyle='--', color='grey')
            ax[0].plot(T_train[0:m], np.ones((m))*y_max, linestyle='--', color='grey')
        ax[0].legend()
        ax[0].set_title('Y (training data)')

        n = N_test
        ax[1].plot(T_test[0:n], Y_test[i, -1, 0:n, 0], label='measured')
        ax[1].plot(T_test[0:n], Yhat_test[i, -1, 0:n, 0], label='estimated', linestyle='--')
        ax[1].legend()
        ax[1].set_title('Y (test data)')

        plt.tight_layout()

        if save:
            if system_name != "":
                fig_dir = f"{system_name}"
            else:
                fig_dir = ''
            fig_name = f"Y_{exp_type}_{system_name}_{model_name}_{AL_method}_{isScale}_{isConst}"
            if pred is not None:
                fig_name += f"_{pred}"
            save_plot(fig, fig_name, fig_dir)

        plt.close()

def save_plot(fig, figName = 'plot', fig_dir = '', fig_sub_dir = ''):
    """
    Save a Matplotlib figure to a specified directory structure.

    This function ensures that the directory structure exists by creating
    any missing directories. It then saves the provided figure in the 
    designated sub-directory with the specified filename.

    Parameters:
        fig_dir (str): The main figure directory inside experiment artifacts.
        fig_sub_dir (str): The sub-directory within fig_dir.
        fig (matplotlib.figure.Figure): The Matplotlib figure to save.
        figName (str): The name of the figure file (without extension).

    Example:
        fig, ax = plt.subplots()
        # ... plotting commands ...
        save_Figure('experiment1', 'results', fig, 'rmse_plot')
    """
    # Define the base figures directory
    base_fig_dir = experiment_figures_root()

    # Ensure the base figures directory exists
    os.makedirs(base_fig_dir, exist_ok=True)

    # Define the main figure directory path
    main_fig_dir = os.path.join(base_fig_dir, fig_dir)
    os.makedirs(main_fig_dir, exist_ok=True)

    # Define the sub-directory path
    sub_fig_dir = os.path.join(main_fig_dir, fig_sub_dir)
    os.makedirs(sub_fig_dir, exist_ok=True)

    # Construct the full file path
    fig_path = os.path.join(sub_fig_dir, f"{figName}.pdf")
    fig_path1 = os.path.join(sub_fig_dir, f"{figName}.png")

    # Save the figure
    fig.savefig(fig_path, format='pdf', bbox_inches='tight')
    fig.savefig(fig_path1, format='png', bbox_inches='tight')
    print(f"Figure saved to {fig_path}")

## Example usage of plot_rmse function

def rmse_r2_save_plot(system_name = 'two_tank', nx = 2, isConst = 1, isScale = 1, isSave = 0, isNoise = 1):
    raise RuntimeError(
        "Legacy HDF5 loading is not included in the public release. " +
        "Use rmse_bfr_r2_save_plot_pkl instead."
    )

    exp_type = 'cmp_al'; 
    # exp_type = 'cmp_delta'; 
    # exp_type = 'cmp_alpha'

    # isConst = 1
    # isScale = 1
    # isNoise = 1
    # nx = 2
    loaddata = True  # set to True to load data
    samples, scores, N_train_max, N_train_init, N_test, nx, ny, nu, AL_method, delta, alpha, N_exp, N_set, AL_method_set, delta_set, alpha_set, pred, qx, qy, rho_x, rho_th, Qx_cov, Qy_cov, Qth_cov = load_data(loaddata, exp_type, nx, isNoise, system_name=system_name, model_name="RNN", isScale = isScale, isConst = isConst)
    # samples, scores, N_train_max, N_train_init, N_test, nx, ny, nu, AL_method, delta, alpha, N_exp, N_set, AL_method_set, delta_set, alpha_set, pred, qx, qy = load_data(loaddata, exp_type, nx, isNoise, system_name=system_name, model_name="RNN", isScale = isScale, isConst = isConst)
    rmse_test = scores['rmse_test']
    r2_test = scores['R2_test']

    isSave = True
    # plot rmse score
    fRMSE = plot_rmse(rmse_test, N_set, N_exp, AL_method_set, np.arange(N_train_init-1, N_train_max), save=isSave, exp_type=exp_type, system_name=system_name, model_name='RNN', pred=pred, delta_set=delta_set, alpha_set=alpha_set, isScale = isScale, isConst = isConst)
    # plot r2 score
    fR2 = plot_r2(r2_test, N_set, N_exp, AL_method_set, np.arange(N_train_init-1, N_train_max), save=isSave, exp_type=exp_type, system_name=system_name, model_name='RNN', pred=pred, delta_set=delta_set, alpha_set=alpha_set, isScale = isScale, isConst = isConst)
    # plot Y_train/Y_test and Yhat_train/Yhat_test
    fY = plot_Y(samples['Y_train'], samples['Yhat_train'], samples['Y_test'], samples['Yhat_test'], N_set, N_exp, AL_method_set, save=isSave, exp_type=exp_type, system_name=system_name, model_name='RNN', pred=pred, delta_set=delta_set, alpha_set=alpha_set, isScale = isScale, isConst = isConst)

def rmse_bfr_r2_save_plot_pkl(exp_type = 'cmp_al', system_name = 'two_tank', model_name = "RNN", isConst = 1, isScale = 1, isSave = 1, isNoise = 1):
    # exp_type = 'cmp_al'; 
    # exp_type = 'cmp_delta'; 
    # exp_type = 'cmp_alpha'

    # isConst = 1
    # isScale = 1
    # isNoise = 1
    # nx = 2
    loaddata = True  # set to True to load data
    results = load_data_pkl(loaddata, exp_type, isNoise, system_name=system_name, model_name=model_name, isScale = isScale, isConst = isConst)
    samples = results['samples']
    scores = results['scores']
    N_train_max = results['N_train_max']
    N_train_init = results['N_train_init']
    N_test = results['N_test']
    N_set = results['N_set']
    N_exp = results['N_exp']
    AL_method_set = results['AL_method_set']
    delta_set = results['delta_set']
    alpha_set = results['alpha_set']
    pred = results['pred']

    system = results['system']
    const = system.const
    y_min = const['y_min']
    y_max = const['y_max']
    Ts = system.Ts

    rmse_test = scores['rmse_test']
    r2_test = scores['R2_test']
    bfr_test = scores['BFR_test']

    # plot rmse score
    fRMSE = plot_rmse(rmse_test, N_set, N_exp, AL_method_set, np.arange(N_train_init-1, N_train_max), save=isSave, exp_type=exp_type, system_name=system_name, model_name=model_name, pred=pred, delta_set=delta_set, alpha_set=alpha_set, isScale = isScale, isConst = isConst)
    # plot r2 score
    # fR2 = plot_r2(r2_test, N_set, N_exp, AL_method_set, np.arange(N_train_init-1, N_train_max), save=isSave, exp_type=exp_type, system_name=system_name, model_name=model_name, pred=pred, delta_set=delta_set, alpha_set=alpha_set, isScale = isScale, isConst = isConst)
    # plot bfr score
    # fBFR = plot_bfr(bfr_test, N_set, N_exp, AL_method_set, np.arange(N_train_init-1, N_train_max), save=isSave, exp_type=exp_type, system_name=system_name, model_name=model_name, pred=pred, delta_set=delta_set, alpha_set=alpha_set, isScale = isScale, isConst = isConst)
    if _is_cmp_al_mode(exp_type):
        # plot Y_train/Y_test and Yhat_train/Yhat_test
        fY = plot_Y(samples['Y_train'], samples['Yhat_train'], samples['Y_test'], samples['Yhat_test'], N_set, N_exp, AL_method_set, save=isSave, exp_type=exp_type, system_name=system_name, model_name=model_name, pred=pred, delta_set=delta_set, alpha_set=alpha_set, isScale = isScale, isConst = isConst, Ts = Ts, y_min=y_min, y_max=y_max)


    # results = load_data_pkl(loaddata, exp_type, isNoise, system_name=system_name, model_name=model_name, isScale = isScale, isConst = 0)
    # scores = results['scores']
    # rmse_test_no_const = scores['rmse_test']
    # results = load_data_pkl(loaddata, exp_type, isNoise, system_name=system_name, model_name=model_name, isScale = isScale, isConst = 1)
    # scores = results['scores']
    # rmse_test_const = scores['rmse_test']
    # fRMSE_const = plot_rmse_with_constraints(rmse_test_no_const, rmse_test_const, N_set, N_exp, AL_method_set, np.arange(N_train_init-1, N_train_max), save=isSave, exp_type=exp_type, system_name=system_name, model_name=model_name, pred=pred, delta_set=delta_set, alpha_set=alpha_set, isScale = isScale)

def rmse_const_save_plot_pkl(exp_type = 'cmp_al', system_name = 'two_tank', model_name = "RNN", isScale = 1, isSave = 1, isNoise = 1):
    # exp_type = 'cmp_al'; 
    # exp_type = 'cmp_delta'; 
    # exp_type = 'cmp_alpha'

    # isConst = 1
    # isScale = 1
    # isNoise = 1
    # nx = 2
    loaddata = True  # set to True to load data
    results = load_data_pkl(loaddata, exp_type, isNoise, system_name=system_name, model_name=model_name, isScale = isScale, isConst = 0)
    samples = results['samples']
    scores = results['scores']
    N_train_max = results['N_train_max']
    N_train_init = results['N_train_init']
    N_set = results['N_set']
    N_exp = results['N_exp']
    AL_method_set = results['AL_method_set']
    delta_set = results['delta_set']
    alpha_set = results['alpha_set']
    pred = results['pred']

    rmse_test_no_const = scores['rmse_test']

    results = load_data_pkl(loaddata, exp_type, isNoise, system_name=system_name, model_name=model_name, isScale = isScale, isConst = 1)
    scores = results['scores']
    rmse_test_const = scores['rmse_test']

    fRMSE_const = plot_rmse_with_constraints(rmse_test_no_const, rmse_test_const, N_set, N_exp, AL_method_set, np.arange(N_train_init-1, N_train_max), save=isSave, exp_type=exp_type, system_name=system_name, model_name=model_name, pred=pred, delta_set=delta_set, alpha_set=alpha_set, isScale = isScale)

def constraint_violation_pkl(system_name = 'two_tank', model_name = "RNN", isConst = 1, isScale = 1, isSave = 0, isNoise = 1):
    exp_type = 'cmp_al'; 
    # exp_type = 'cmp_delta'; 
    # exp_type = 'cmp_alpha'

    # isConst = 1
    # isScale = 1
    # isNoise = 1
    # nx = 2
    isLoad = True  # set to True to load data
    results = load_data_pkl(isLoad, exp_type, isNoise, system_name=system_name, model_name=model_name, isScale = isScale, isConst = isConst)

    samples = results['samples']
    Y_train = samples['Y_train']
    N_train_init = results['N_train_init']
    N_train_max = results['N_train_max']
    AL_method_set = results['AL_method_set']
    system = results['system']
    const = system.const
    y_min = const['y_min']
    y_max = const['y_max']
    mean_violation, count_violation = compute_constraint_violation(Y_train[:,:,N_train_init:,:], y_min, y_max)
    mean_violation_per_method = np.mean(mean_violation, axis=1)  # shape: (N_set,)
    count_violation_per_method = np.mean(count_violation, axis=1)  # shape: (N_set,)
    count_violation_percentage_per_method = 100 * np.mean(count_violation, axis=1)/(N_train_max - N_train_init)  # shape: (N_set,)
    
    print(f"y_min: {const['y_min']}")
    print(f"y_max: {const['y_max']}")
    print(f"Active learning method: {AL_method_set}")
    print(f"Mean violation per method: {mean_violation_per_method}")
    print(f"Count violation per method: {count_violation_per_method}")
    print(f"Count violation percentage per method: {count_violation_percentage_per_method}")

    return mean_violation_per_method, count_violation_per_method, count_violation_percentage_per_method

def load_save_pkl(exp_type = 'cmp_al', system_name = 'two_tank', model_name = "RNN", isConst = 1, isScale = 1, isSave = 1, isNoise = 1):
    loaddata = True  # set to True to load data

    results = load_data_pkl(loaddata, exp_type, isNoise, system_name=system_name, model_name=model_name, isScale = isScale, isConst = isConst)
    results_alpha = load_data_pkl_0(loaddata, 'cmp_alpha', isNoise, system_name=system_name, model_name=model_name, isScale = isScale, isConst = isConst)
    samples = results['samples']
    scores = results['scores']
    system = results['system']
    model = "RNN"
    pred = results['pred']
    exp_type = results['exp_type']
    AL_method_set = results['AL_method_set']
    delta_set = results['delta_set']
    alpha_set = results['alpha_set']
    N_train_max = results['N_train_max']
    N_train_init = results['N_train_init']
    N_test = results['N_test']
    N_set = results['N_set']
    N_exp = results['N_exp']
    rho_x = results['rho_x']
    rho_th = results['rho_th']
    Qx_cov = results['Qx_cov']
    Qy_cov = results['Qy_cov']
    Qth_cov = results['Qth_cov']
    isScale = results['isScale']
    isConst = results['isConst']

    rmse_test = scores['rmse_test']
    r2_test = scores['R2_test']
    bfr_test = scores['BFR_test']

    # results_alpha = load_data_pkl(loaddata, 'cmp_alpha', isNoise, system_name=system_name, model_name=model_name, isScale = isScale, isConst = isConst)

    samples_alpha = results_alpha['samples']
    scores_alpha = results_alpha['scores']

    # Replace the first row of each element in samples and scores with the corresponding one in samples_alpha
    for key in samples:
        print(f"Replacing samples for key: {key}")
        samples[key][3, ...] = samples_alpha[key][0, ...]
    for key in scores:
        print(f"Replacing scores for key: {key}")
        scores[key][3, ...] = scores_alpha[key][0, ...]

    # model_name = "RNN3"
    save_data_pkl(isSave, samples, scores, system, model, pred, exp_type, N_train_init, N_train_max, N_test, N_exp, N_set, AL_method_set, delta_set, alpha_set, rho_x, rho_th, Qx_cov, Qy_cov, Qth_cov, system_name, model_name, isScale, isConst)

if __name__ == "__main__":
    # test_plot_rmse()
    # test_save_plot()
    # oxidation_save_plot()
    # robot_arm_save_plot()
    # rmse_r2_save_plot(system_name = 'two_tank', nx = 2, isConst = 1, isScale = 1, isSave = 0, isNoise = 1)
    # rmse_r2_save_plot(system_name = 'robot_arm', nx = 5, isConst = 1, isScale = 1, isSave = 1, isNoise = 1)
    # rmse_r2_save_plot(system_name = 'oxidation', nx = 4, isConst = 1, isScale = 1, isSave = 1, isNoise = 1)

    # rmse_r2_save_plot(system_name = 'two_tank', nx = 2, isConst = 0, isScale = 1, isSave = 1, isNoise = 1)
    # rmse_r2_save_plot(system_name = 'robot_arm', nx = 5, isConst = 0, isScale = 1, isSave = 1, isNoise = 1)
    # rmse_r2_save_plot(system_name = 'oxidation', nx = 4, isConst = 0, isScale = 1, isSave = 1, isNoise = 1)

    # system_set = ['two_tank']
    # system_set = ['oxidation', 'robot_arm']
    system_set = ['two_tank', 'robot_arm', 'oxidation']
    for system_name in system_set:
        for isConst in [0, 1]:
            constraint_violation_pkl(system_name = system_name, model_name = "RNN", isConst = isConst, isScale = 1, isSave = 0, isNoise = 1)

    import matplotlib.pyplot as plt
    plt.show = lambda *args, **kwargs: None
    isSave = 1
    isConst = 0
    # system_set = ['two_tank']
    # system_set = ['robot_arm']
    system_set = ['two_tank', 'robot_arm', 'oxidation']
    isConst_set = [0, 1]
    for system_name in system_set:
        for isConst in isConst_set:
            # rmse_bfr_r2_save_plot_pkl(system_name = system_name, model_name = "RNN", isConst = isConst, isScale = 1, isSave = isSave, isNoise = 1)
            rmse_const_save_plot_pkl(exp_type = 'cmp_al', system_name = system_name, model_name = "RNN", isScale = 1, isSave = isSave, isNoise = 1)
            # rmse_bfr_r2_save_plot_pkl(exp_type = 'cmp_delta', system_name = system_name, model_name = "RNN", isConst = isConst, isScale = 1, isSave = isSave, isNoise = 1)
            # rmse_bfr_r2_save_plot_pkl(exp_type = 'cmp_alpha', system_name = system_name, model_name = "RNN", isConst = isConst, isScale = 1, isSave = isSave, isNoise = 1)
            # load_save_pkl(exp_type = 'cmp_al', system_name = system_name, model_name = "RNN", isConst = isConst, isScale = 1, isSave = isSave, isNoise = 1)

    import importlib
    importlib.reload(plt)  # Restore plt.show to default

    plt.show()  # This will show all figures that are still open





    # rmse_bfr_r2_save_plot_pkl(system_name = 'oxidation', model_name = "RNN", isConst = isConst, isScale = 1, isSave = isSave, isNoise = 1)
    # rmse_bfr_r2_save_plot_pkl(system_name = 'two_tank', model_name = "RNN", isConst = isConst, isScale = 1, isSave = isSave, isNoise = 1)
    # rmse_bfr_r2_save_plot_pkl(system_name = 'robot_arm', model_name = "RNN", isConst = isConst, isScale = 1, isSave = isSave, isNoise = 1)
    # isConst = 1
    # rmse_bfr_r2_save_plot_pkl(system_name = 'oxidation', model_name = "RNN", isConst = isConst, isScale = 1, isSave = isSave, isNoise = 1)
    # rmse_bfr_r2_save_plot_pkl(system_name = 'two_tank', model_name = "RNN", isConst = isConst, isScale = 1, isSave = isSave, isNoise = 1)
    # rmse_bfr_r2_save_plot_pkl(system_name = 'robot_arm', model_name = "RNN", isConst = isConst, isScale = 1, isSave = isSave, isNoise = 1)
