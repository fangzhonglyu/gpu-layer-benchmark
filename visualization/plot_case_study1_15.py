import csv, re
from pathlib import Path

import seaborn as sns
palette = sns.color_palette("Set2", 4)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import matplotlib.font_manager as fm
from matplotlib.patches import Patch

from matplotlib import rcParams
# # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams['font.family'] = 'serif'

# Optionally, specify a list of preferred serif fonts in order of preference
# Matplotlib will try to find the first available font from this list
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Computer Modern Roman']


def parse_case_to_dict(filename):
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)

        cost_unaware = {}
        cost_aware = {}

        # grab the n-th row (0-based)
        for row in reader:
            dict = cost_aware if row["cost_aware"] == "True" else cost_unaware
            dict[row["net"]] = {
                "prefill": float(row["best_value_prefill"]),
                "decode": float(row["best_value_decode"]),
                "e2e": float(row["e2e_best_value"]),
            }

    return cost_unaware, cost_aware

ours_cost_unaware, ours_cost_aware = parse_case_to_dict(Path("result_archive/old_bench/case_study1_15_our.csv"))
baseline_cost_unaware, baseline_cost_aware = parse_case_to_dict(Path("result_archive/old_bench/case_study1_15_baseline.csv"))

def plot_case_study():
    # One subplot per row (cost-unaware, cost-aware), all three metrics combined in each
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)

    # Define order of metrics
    metrics = ["e2e", "prefill", "decode"]
    col_titles = ["E2E", "Prefill", "Decode"]

    # Hatching to distinguish full-pipeline (E2E) vs per-token (prefill/decode)
    e2e_hatch = ""
    per_token_hatch = "///"

    # Nets (x-axis categories)
    nets_unaware = list(baseline_cost_unaware.keys()) or list(ours_cost_unaware.keys())
    nets_aware = list(baseline_cost_aware.keys()) or list(ours_cost_aware.keys())

    def nice_net(name: str) -> str:
        # Shorten long names like Chatbot_OPT-66B -> Chatbot
        return re.sub(r"_OPT.*$", "", name)

    def draw_row(ax, nets, base_dict, our_dict, row_title):
        num_metrics = len(metrics)
        num_nets = len(nets)
        width = 0.3
        gap = 0.5  # reduced visual gap between metric groups
        group_stride = num_nets + gap

        xticks = []
        xticklabels = []
        baseline_handle = ours_handle = None

        # Horizontal reference at Baseline=1
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        for j, metric in enumerate(metrics):
            start = j * group_stride
            xs = start + np.arange(num_nets)

            baseline_vals = np.array([base_dict[n][metric] for n in nets], dtype=float)
            ours_vals = np.array([our_dict[n][metric] for n in nets], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                base_norm = np.ones_like(baseline_vals)
                ours_norm = np.where(baseline_vals != 0, ours_vals / baseline_vals, np.nan)

            hatch = e2e_hatch if metric == "e2e" else per_token_hatch
            b1 = ax.bar(
                xs - width / 2,
                base_norm,
                width,
                color=palette[0],
                edgecolor="black",
                linewidth=0.3,
                hatch=hatch,
                label="Baseline",
            )
            b2 = ax.bar(
                xs + width / 2,
                ours_norm,
                width,
                color=palette[1],
                edgecolor="black",
                linewidth=0.3,
                hatch=hatch,
                label="Ours",
            )
            baseline_handle, ours_handle = b1[0], b2[0]

            # Collect ticks and labels (metric on first line, net on second)
            xticks.extend(xs.tolist())
            xticklabels.extend([f"{col_titles[j]}\n{nice_net(n)}" for n in nets])

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=35, ha="right", va="top", fontsize=14)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        for label in ax.get_xticklabels():
            label.set_transform(label.get_transform() +
                                matplotlib.transforms.ScaledTranslation(12/72, 0, ax.figure.dpi_scale_trans))

        # Labels
        ax.set_ylabel("Normalized Energy × cost" if row_title == "Cost-unaware" else "Normalized Energy", fontsize=14, fontweight='bold')

        # Scale a bit of headroom
        ax.set_ylim(bottom=0)

        return baseline_handle, ours_handle

    h1, h2 = draw_row(axs[0], nets_unaware, baseline_cost_unaware, ours_cost_unaware, "Cost-unaware")
    draw_row(axs[1], nets_aware, baseline_cost_aware, ours_cost_aware, "Cost-aware")

    # Proxy patch to explain hatch meaning (Per Token)
    per_token_proxy = Patch(facecolor="white", edgecolor="black", hatch=per_token_hatch, label="Per Token")
    end_to_end_proxy = Patch(facecolor="white", edgecolor="black", hatch=e2e_hatch, label="E2E")
    baseline_proxy = Patch(facecolor=palette[0], edgecolor="black", label="Baseline")
    ours_proxy = Patch(facecolor=palette[1], edgecolor="black", label="Ours")

    # Single shared legend (Baseline, Ours, and hatch meaning)
    fig.legend([baseline_proxy, ours_proxy, per_token_proxy, end_to_end_proxy], ["Baseline", "Ours", "Per Token", "End-to-End"], loc="upper center", ncol=4, frameon=False, fontsize=14,
        bbox_to_anchor=(0.5, 1.03))

    # Tighten layout and leave space for legend
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=0.8)

    # Save and show
    Path("graphs").mkdir(parents=True, exist_ok=True)
    fig.savefig("graphs/case_study1_15_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig("graphs/case_study1_15_comparison.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot_case_study()
