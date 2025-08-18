import csv, re
from pathlib import Path

def parse_old_bench_to_dict(filename, row_index=0):
    data = {}
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)

        # grab the n-th row (0-based)
        for i, row in enumerate(reader):
            if i == row_index:
                for col in row:
                    if col.endswith("_min_energy"):
                        net_name = col.replace("_min_energy", "")
                        latency_col = f"{net_name}_latency"

                        if latency_col in row:
                            data[net_name] = {
                                "min_energy": float(row[col]),
                                "latency": float(row[latency_col])
                            }
                        else:
                            raise ValueError(f"Column {latency_col} not found in the CSV file {filename}.")

                    elif col.endswith("_min_edp"):
                        net_name = col.replace("_min_edp", "")
                        latency_col = f"{net_name}_latency"

                        if latency_col in row:
                            data[net_name] = {
                                "min_energy": float(row[col]),
                                "latency": float(row[latency_col])
                            }
                        else:
                            raise ValueError(f"Column {latency_col} not found in the CSV file {filename}.")
                break  # stop after reaching the desired row
    return data

# Example: get data from the 3rd row (index=2)
homo_energy             = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064606_energy_False.csv", row_index=2)
homo_energy_x_cost      = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064613_energy_True.csv", row_index=2)
homo_edp                = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064618_edp_False.csv", row_index=2)
homo_edp_x_cost         = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064623_edp_True.csv", row_index=2)

homo_per_net_energy         = homo_energy.copy()
homo_per_net_energy_x_cost  = homo_energy_x_cost.copy()
homo_per_net_edp            = homo_edp.copy()
homo_per_net_edp_x_cost     = homo_edp_x_cost.copy()

chip_pool_energy        = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064606_energy_False.csv", row_index=9)
chip_pool_energy_x_cost = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064613_energy_True.csv", row_index=9)
chip_pool_edp           = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064618_edp_False.csv", row_index=9)
chip_pool_edp_x_cost    = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064623_edp_True.csv", row_index=9)

def parse_gpu_bench_dir_to_dict(root_dir: str):
    # Patterns from the file’s last lines
    lat_pat = re.compile(r"Bottleneck Latency \(ms\):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    energy_pat = re.compile(r"Total Pipeline Energy with Idle \(J\):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    results = {}
    for path in Path(root_dir).rglob("*.txt"):
        net_name = path.stem  # e.g., mobilenet_v3_small_b1_seq1

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue  # skip unreadable files silently

        # Get the last occurrence (in case similar lines appear earlier)
        lat_val = None
        energy_val = None

        for m in lat_pat.finditer(text):
            lat_val = float(m.group(1))
        for m in energy_pat.finditer(text):
            energy_val = float(m.group(1))

        if lat_val is not None and energy_val is not None:
            # Keep the same dict format: {"min_energy": ..., "latency": ...}
            results[net_name] = {
                "min_energy": energy_val,  # (J) from "Total Pipeline Energy with Idle"
                "latency": lat_val,        # (ms) from "Bottleneck Latency"
            }

    return results

def calculate_gpu_edp_dict(gpu_energy_dict):
    gpu_edp_dict = {}
    for net_name, values in gpu_energy_dict.items():
        energy_joules = values["min_energy"]
        latency_seconds = values["latency"]
        gpu_edp_dict[net_name] = {
            "min_edp": energy_joules * latency_seconds,
            "latency": values["latency"]
        }
    return gpu_edp_dict

def calculate_gpu_edp_x_cost_dict(gpu_energy_dict, gpu_cost:float):
    gpu_edp_x_cost_dict = {}
    for net_name, values in gpu_energy_dict.items():
        energy_joules = values["min_energy"]
        latency_seconds = values["latency"]
        gpu_edp_x_cost_dict[net_name] = {
            "min_edp": energy_joules * latency_seconds * gpu_cost,
            "latency": values["latency"]
        }

def calculate_gpu_energy_x_cost_dict(gpu_energy_dict, gpu_cost:float):
    gpu_energy_x_cost_dict = {}
    for net_name, values in gpu_energy_dict.items():
        energy_joules = values["min_energy"]
        gpu_energy_x_cost_dict[net_name] = {
            "min_energy": energy_joules * gpu_cost,
            "latency": values["latency"]
        }
    return gpu_energy_x_cost_dict


gpu_energy = parse_gpu_bench_dir_to_dict("result_archive/NVIDIA A100-SXM4-40GB")
gpu_energy_x_cost = calculate_gpu_energy_x_cost_dict(gpu_energy, 10000)  # Example cost
gpu_edp = calculate_gpu_edp_dict(gpu_energy)
gpu_edp_x_cost = calculate_gpu_edp_x_cost_dict(gpu_energy, 10000)  # Example cost


import numpy as np
import matplotlib.pyplot as plt


def plot_all_energy(
    gpu_energy: dict,
    chip_pool_energy: dict,
    homo_per_net_energy: dict,
    homo_energy: dict,
    metric: str = "min_energy",   # or "latency"
    title: str = "Energy Comparison",
    savepath: str | None = None,
):
    """
    Plots grouped bar chart:
        x-axis: all net_names (union across all dicts)
        y-axis: selected metric ("min_energy" or "latency")
        series: GPU, Chip Pool, Homo Per Net, Homo
    """
    # Collect all network names
    net_names = sorted(set(gpu_energy) | set(chip_pool_energy) | set(homo_per_net_energy) | set(homo_energy))

    setups = ["GPU", "Chip Pool", "Homo Per Net", "Homo"]
    setup_dicts = {
        "GPU": gpu_energy,
        "Chip Pool": chip_pool_energy,
        "Homo Per Net": homo_per_net_energy,
        "Homo": homo_energy,
    }
    colors = {
        "GPU": "tab:blue",
        "Chip Pool": "tab:orange",
        "Homo Per Net": "tab:green",
        "Homo": "tab:red",
    }

    x = np.arange(len(net_names))  # base positions
    width = 0.15                   # bar width

    fig, ax = plt.subplots(figsize=(6, 6))

    for i, setup in enumerate(setups):
        vals = []
        for net in net_names:
            if net in setup_dicts[setup]:
                vals.append(setup_dicts[setup][net][metric])
            else:
                vals.append(np.nan)  # handle missing gracefully
        ax.bar(x + i*width, vals, width, label=setup, color=colors[setup])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    fig.subplots_adjust(hspace=0.05)

    ax1.set_ylim(150, 210)  # Top subplot: for large values
    ax2.set_ylim(0, 20)     # Bottom subplot: for small values

    ax.set_ylabel(metric)
    # ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig


plot_all_energy(
    gpu_energy=gpu_energy,
    chip_pool_energy=chip_pool_energy,
    homo_per_net_energy=homo_per_net_energy,
    homo_energy=homo_energy,
    metric="min_energy",
    title="Energy Comparison",
    savepath="graphs/energy_comparison.png"
)

# plot_all_energy(
#     gpu_energy=gpu_energy,
#     chip_pool_energy=chip_pool_energy,
#     homo_per_net_energy=homo_per_net_energy,
#     homo_energy=homo_energy,
#     metric="latency",
#     title="Latency Comparison",
#     savepath="graphs/latency_comparison.png"
# )

plot_all_energy(
    gpu_energy=gpu_energy_x_cost,
    chip_pool_energy=chip_pool_energy_x_cost,
    homo_per_net_energy=homo_per_net_energy_x_cost,
    homo_energy=homo_energy_x_cost,
    metric="min_energy",
    title="Enegy X Cost Comparison",
    savepath="graphs/energy_x_cost_comparison.png"
)

plot_all_energy(
    gpu_energy=gpu_edp,
    chip_pool_energy=chip_pool_edp,
    homo_per_net_energy=homo_per_net_edp,
    homo_energy=homo_edp,
    metric="min_energy",
    title="EDP Comparison",
    savepath="graphs/edp_comparison.png"
)

plot_all_energy(
    gpu_energy=gpu_edp_x_cost,
    chip_pool_energy=chip_pool_edp_x_cost,
    homo_per_net_energy=homo_per_net_edp_x_cost,
    homo_energy=homo_edp_x_cost,
    metric="min_energy",
    title="EDP X Cost Comparison",
    savepath="graphs/edp_x_cost_comparison.png"
)
