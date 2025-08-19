import seaborn as sns
palette = sns.color_palette("Set2", 5)

from visualization.parse_bench import gpu_energy, gpu_energy_x_cost, gpu_edp, gpu_edp_x_cost
from visualization.parse_bench import chip_pool_energy, chip_pool_energy_x_cost, chip_pool_edp, chip_pool_edp_x_cost
from visualization.parse_bench import homo_energy, homo_energy_x_cost, homo_edp, homo_edp_x_cost
from visualization.parse_bench import homo_per_net_energy, homo_per_net_energy_x_cost, homo_per_net_edp, homo_per_net_edp_x_cost
from visualization.parse_bench import ideal_energy, ideal_energy_x_cost, ideal_edp, ideal_edp_x_cost

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def make_broken_y_axes_in(ax_parent, height_ratios=(1, 3), hspace=0.05):
    """
    Turn a single subplot cell (ax_parent) into (ax_top, ax_bottom) stacked
    axes using a sub-GridSpec (keeps ax_parent, just hides it).
    """
    fig = ax_parent.figure
    spec = ax_parent.get_subplotspec()
    sub = spec.subgridspec(2, 1, height_ratios=height_ratios, hspace=hspace)

    ax_parent.set_visible(False)   # keep but hide
    ax_top    = fig.add_subplot(sub[0, 0])
    ax_bottom = fig.add_subplot(sub[1, 0], sharex=ax_top)
    return ax_top, ax_bottom

def format_with_suffix(v, digits=2):
    """Format number with K/M suffixes, no scientific for small values."""
    if v >= 1e6:
        return f"{v/1e6:.{digits}f}m"
    elif v >= 1e3:
        return f"{v/1e3:.{digits}f}k"
    else:
        return f"{v:.{digits}f}"

def plot_all_energy_subplot_broken(
    ax_top,
    ax_bottom,
    gpu_energy: dict,
    chip_pool_energy: dict,
    homo_per_net_energy: dict,
    homo_energy: dict,
    ideal_energy: dict,
    metric: str = "min_energy",
    title: str = "Energy Comparison",
    ylabel: str = "Energy (J)",
    show_legend: bool = False,
    break_point: float = 1.2,
    gap: float = 0.02,        # visual split around break_point
    top_margin: float = 2.6, # headroom on the top axis
    break_mark_size: float = 12.0,
    hide_x_ticks: bool = False
):
    """
    Draw bars once on BOTH axes; use y-lims to create a cut-out and slanted marks
    exactly like the Matplotlib example you sent.
    """
     # ---- collect names ----
    net_names = sorted(set(gpu_energy) | set(chip_pool_energy) |
                       set(homo_per_net_energy) | set(homo_energy) | set(ideal_energy))
    if "geometric_mean" in net_names:
        net_names.remove("geometric_mean")
        net_names.append("geometric_mean")

    setups = ["GPU", "Homogeneous All Nets", "Homogeneous Per Net", "Chiplet Pool", "Ideal"]
    setup_dicts = {
        "GPU": gpu_energy,
        "Homogeneous All Nets": homo_energy,
        "Homogeneous Per Net": homo_per_net_energy,
        "Chiplet Pool": chip_pool_energy,
        "Ideal": ideal_energy
    }
    colors = {
        "GPU": palette[0],
        "Homogeneous All Nets": palette[1],
        "Homogeneous Per Net": palette[2],
        "Chiplet Pool": palette[3],
        "Ideal": palette[4]
    }

    x = np.arange(len(net_names))
    width = 0.15

    # ---- normalization ----
    NORM = "Homogeneous All Nets"
    norms = {net: setup_dicts[NORM][net][metric]
             for net in net_names if net in setup_dicts[NORM]}

    # ---- compute values once ----
    all_vals, series_vals = [], {}
    for setup in setups:
        vals = []
        for net in net_names:
            if net in setup_dicts[setup]:
                v = setup_dicts[setup][net][metric] / norms[net]
                vals.append(v)
                if np.isfinite(v): all_vals.append(v)
            else:
                vals.append(np.nan)
        series_vals[setup] = vals

    ymax = max(all_vals) if all_vals else break_point
    ymin = min(all_vals) if all_vals else 0
    any_break = ymax > break_point + 1e-12

    top_bar_containers = []
    # ---- draw bars on BOTH axes (no removals) ----
    for i, setup in enumerate(setups):
        vals = series_vals[setup]
        ax_bottom.bar(x + i*width, vals, width, label=setup, color=colors[setup])
        bars_top = ax_top.bar(x + i*width, vals, width, label=setup, color=colors[setup])
        top_bar_containers.append((bars_top, vals))

    # ---- y-limits (like your example) ----
    low_max  = max(break_point - gap, 0)
    high_min = break_point + gap
    high_max = ymax * (1 + top_margin) if any_break else break_point * (1 + top_margin)

    ax_top.set_ylim(high_min, high_max)      # outliers only
    ax_bottom.set_ylim(0, low_max)           # main detail

    ax_top.set_yscale('log')  # log scale for top axis

    if ymin < 1e-1:
        ax_bottom.set_ylim(1e-3, low_max)
        ax_bottom.set_yscale('log')  # log scale for bottom axis

    # ---- ticks/labels ----
    if hide_x_ticks:
        ax_bottom.set_xticks([])
    else:
        ax_bottom.set_xticks(x + (len(setups)-1)*width/2)
        ax_bottom.set_xticklabels(net_names, rotation=25, ha="right", fontsize=9)

    if ylabel:
        ax_bottom.set_ylabel(ylabel, fontsize=10, labelpad=3, fontweight='bold')
        ax_bottom.yaxis.set_label_coords(-0.04, 0.8)
        # ax_top.set_title(title, fontsize=10, fontweight='bold')

    # Legend on top axis (optional)
    if show_legend:
        ax_top.legend(loc='upper right', fontsize=8)

    # grids + tick params
    for a in (ax_top, ax_bottom):
        a.grid(True, alpha=0.3, axis='y', which='major')
        a.tick_params(axis='y', labelsize=6, pad=0, rotation=90)

    # ---- hide touching spines and set tick sides like example ----
    if any_break:
        ax_top.spines.bottom.set_visible(False)
        ax_bottom.spines.top.set_visible(False)
        ax_top.xaxis.tick_top()
        ax_top.tick_params(labeltop=False)  # no labels on top
        ax_bottom.xaxis.tick_bottom()
    else:
        # if no break needed, hide the top axis to avoid empty panel
        ax_top.set_visible(False)

    if any_break:
        for bars_top, vals in top_bar_containers:
            for rect, v in zip(bars_top, vals):
                if not (np.isfinite(v) and v > break_point):
                    continue
                cx = rect.get_x() + rect.get_width() / 2.0
                ax_top.annotate(
                    f"{v:.1f}X",           # value followed by 'X'
                    xy=(cx, 1.2),
                    xytext=(7, 5),         # vertical offset in points
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    clip_on=False,
                    rotation=90,  # vertical text
                )

    # ---- slanted cut-out markers (exact Matplotlib example style) ----
    if any_break:
        d = .5  # proportion of vertical to horizontal extent
        kwargs = dict(marker=[(-1, -d), (1, d)],
                      markersize=break_mark_size, linestyle="none",
                      color='k', mec='k', mew=1, clip_on=False)
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
        ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

    for label in ax_bottom.get_xticklabels():
        label.set_transform(label.get_transform() + 
                        matplotlib.transforms.ScaledTranslation(6/72, 0, ax_bottom.figure.dpi_scale_trans))

def create_combined_plot_direct(savepath: str = "graphs/combined_comparison.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))

    # 1) Energy
    ax_top, ax_bottom = make_broken_y_axes_in(axes[0,0], height_ratios=(1, 2.5), hspace=0.05)
    plot_all_energy_subplot_broken(
        ax_top, ax_bottom,
        gpu_energy=gpu_energy,
        chip_pool_energy=chip_pool_energy,
        homo_per_net_energy=homo_per_net_energy,
        homo_energy=homo_energy,
        ideal_energy=ideal_energy,
        metric="min_energy",
        title="Energy Comparison",
        ylabel="Normalized Energy",
        show_legend=False,
        break_point=1.1,
        hide_x_ticks=True
    )

    # 2) Energy × Cost
    ax_top, ax_bottom = make_broken_y_axes_in(axes[0,1], height_ratios=(1, 2.5), hspace=0.05)
    plot_all_energy_subplot_broken(
        ax_top, ax_bottom,
        gpu_energy=gpu_energy_x_cost,
        chip_pool_energy=chip_pool_energy_x_cost,
        homo_per_net_energy=homo_per_net_energy_x_cost,
        homo_energy=homo_energy_x_cost,
        ideal_energy=ideal_energy_x_cost,
        metric="min_energy",
        title="Energy × Cost Comparison",
        ylabel="Normalized Energy × Cost",
        show_legend=False,
        break_point=1.1,
        hide_x_ticks=True
    )

    # 3) EDP
    ax_top, ax_bottom = make_broken_y_axes_in(axes[1,0], height_ratios=(1, 2.5), hspace=0.05)
    plot_all_energy_subplot_broken(
        ax_top, ax_bottom,
        gpu_energy=gpu_edp,
        chip_pool_energy=chip_pool_edp,
        homo_per_net_energy=homo_per_net_edp,
        homo_energy=homo_edp,
        ideal_energy=ideal_edp,
        metric="min_energy",
        title="EDP Comparison",
        ylabel="Normalized EDP",
        show_legend=False,
        break_point=1.1,
    )

    # 4) EDP × Cost (legend here)
    ax_top, ax_bottom = make_broken_y_axes_in(axes[1,1], height_ratios=(1, 2.5), hspace=0.05)
    plot_all_energy_subplot_broken(
        ax_top, ax_bottom,
        gpu_energy=gpu_edp_x_cost,
        chip_pool_energy=chip_pool_edp_x_cost,
        homo_per_net_energy=homo_per_net_edp_x_cost,
        homo_energy=homo_edp_x_cost,
        ideal_energy=ideal_edp_x_cost,
        metric="min_energy",
        title="EDP × Cost Comparison",
        ylabel="Normalized EDP × Cost ",
        show_legend=False,
        break_point=1.1
    )

    handles, labels = ax_top.get_legend_handles_labels()  # use the last panel’s top axis

    print("handles", handles)
    print("labels", labels)

         # If you prefer, you can still call tight_layout; subgridspec usually handles it well.
    plt.tight_layout(rect=[0, 0, 1, 0.95],w_pad=-2, h_pad=0.8)

    fig.legend(
        handles, labels,
        loc='center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=5,
        frameon=False,
        fontsize=10
    )

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    plt.show()
    return fig

# run
create_combined_plot_direct(savepath='graphs/combined_comparison_norm_two_row.pdf')



# plot_all_energy(
#     gpu_energy=gpu_energy,
#     chip_pool_energy=chip_pool_energy,
#     homo_per_net_energy=homo_per_net_energy,
#     homo_energy=homo_energy,
#     metric="min_energy",
#     title="Energy Comparison",
#     savepath="graphs/energy_comparison.png"
# )


# plot_all_energy(
#     gpu_energy=gpu_energy_x_cost,
#     chip_pool_energy=chip_pool_energy_x_cost,
#     homo_per_net_energy=homo_per_net_energy_x_cost,
#     homo_energy=homo_energy_x_cost,
#     metric="min_energy",
#     title="Enegy X Cost Comparison",
#     savepath="graphs/energy_x_cost_comparison.png"
# )

# plot_all_energy(
#     gpu_energy=gpu_edp,
#     chip_pool_energy=chip_pool_edp,
#     homo_per_net_energy=homo_per_net_edp,
#     homo_energy=homo_edp,
#     metric="min_energy",
#     title="EDP Comparison",
#     savepath="graphs/edp_comparison.png"
# )

# plot_all_energy(
#     gpu_energy=gpu_edp_x_cost,
#     chip_pool_energy=chip_pool_edp_x_cost,
#     homo_per_net_energy=homo_per_net_edp_x_cost,
#     homo_energy=homo_edp_x_cost,
#     metric="min_energy",
#     title="EDP X Cost Comparison",
#     savepath="graphs/edp_x_cost_comparison.png"
# )
