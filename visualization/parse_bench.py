import csv, re
from pathlib import Path

CNNs = ['resnet','mobilenet','efficientnet','replknet']

def add_geometric_mean_to_dict(data_dict):
    """
    Adds a geometric mean entry to the dictionary under the specified key.
    If the key already exists, it multiplies the existing value by the new value.
    """
    min_energy_product = 1.0
    min_latency_product = 1.0

    for _, value in data_dict.items():
        if "min_energy" in value:
            min_energy_product *= value["min_energy"]
        if "latency" in value:
            min_latency_product *= value["latency"]

    n = len(data_dict)
    if n > 0:
        geometric_mean_energy = min_energy_product ** (1/n)
        geometric_mean_latency = min_latency_product ** (1/n)
    
        data_dict["geometric_mean"] = {
            "min_energy": geometric_mean_energy,
            "latency": geometric_mean_latency
        }

        print(f"Geometric mean added: {geometric_mean_energy} J, {geometric_mean_latency} ms")
    return data_dict
    

def parse_old_bench_to_dict(filename, row_index=0):
    data = {}
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)

        # grab the n-th row (0-based)
        for i, row in enumerate(reader):
            if i == row_index:
                for col in row:
                    if col.endswith("_min_energy") or col.endswith("_min_edp"):
                        net_name = col.replace("_min_energy", "").replace("_min_edp", "")
                        latency_col = f"{net_name}_latency"

                        if latency_col in row:
                            if any(cnn in net_name for cnn in CNNs):
                                net_name = net_name.replace("_seq1", "")
                            data[net_name] = {
                                "min_energy": float(row[col]),
                                "latency": float(row[latency_col])
                            }
                        else:
                            raise ValueError(f"Column {latency_col} not found in the CSV file {filename}.")
                break  # stop after reaching the desired row
    return add_geometric_mean_to_dict(data)


# Example: get data from the 3rd row (index=2)
homo_energy             = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064606_energy_False.csv", row_index=0)
homo_energy_x_cost      = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064613_energy_True.csv", row_index=0)
homo_edp                = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064618_edp_False.csv", row_index=0)
homo_edp_x_cost         = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064623_edp_True.csv", row_index=0)


chip_pool_energy        = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064606_energy_False.csv", row_index=7)
chip_pool_energy_x_cost = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064613_energy_True.csv", row_index=7)
chip_pool_edp           = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064618_edp_False.csv", row_index=7)
chip_pool_edp_x_cost    = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064623_edp_True.csv", row_index=7)

# ideal_energy            = parse_old_bench_to_dict("result_archive/old_bench/all_chiplets_20250819_065406_energy_False.csv", row_index=0)
# ideal_energy_x_cost     = parse_old_bench_to_dict("result_archive/old_bench/all_chiplets_20250819_065402_energy_True.csv", row_index=0)
# ideal_edp               = parse_old_bench_to_dict("result_archive/old_bench/all_chiplets_20250819_065415_edp_False.csv", row_index=0)
# ideal_edp_x_cost        = parse_old_bench_to_dict("result_archive/old_bench/all_chiplets_20250819_065411_edp_True.csv", row_index=0)

ideal_energy            = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064606_energy_False.csv", row_index=15)
ideal_energy_x_cost     = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064613_energy_True.csv", row_index=15)
ideal_edp               = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064618_edp_False.csv", row_index=15)
ideal_edp_x_cost        = parse_old_bench_to_dict("result_archive/old_bench/incremental_chiplet_sweep_20250818_064623_edp_True.csv", row_index=15)

def parse_per_net_bench_to_dict(filename):
    data = {}
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)

        # grab the n-th row (0-based)
        for _, row in enumerate(reader):
            network_name = row["network"]
            if any(cnn in network_name for cnn in CNNs):
                network_name = network_name.replace("_seq1", "")

            data[network_name] = {
                "min_energy": float(row["energy"] if "energy" in row else row["edp"]),
                "latency": 0
            }
    return add_geometric_mean_to_dict(data)

homo_per_net_energy         = parse_per_net_bench_to_dict("result_archive/old_bench/optimal_single_chiplet_energy_False_20250819_065323.csv")
homo_per_net_energy_x_cost  = parse_per_net_bench_to_dict("result_archive/old_bench/optimal_single_chiplet_energy_True_20250819_065323.csv")
homo_per_net_edp            = parse_per_net_bench_to_dict("result_archive/old_bench/optimal_single_chiplet_edp_False_20250819_065323.csv")
homo_per_net_edp_x_cost     = parse_per_net_bench_to_dict("result_archive/old_bench/optimal_single_chiplet_edp_True_20250819_065323.csv")

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
            if any(cnn in net_name for cnn in CNNs):
                net_name = net_name.replace("_seq1", "")
            results[net_name] = {
                "min_energy": energy_val,  # (J) from "Total Pipeline Energy with Idle"
                "latency": lat_val,        # (ms) from "Bottleneck Latency"
            }

    return add_geometric_mean_to_dict(results)

def calculate_gpu_edp_dict(gpu_energy_dict):
    gpu_edp_dict = {}
    for net_name, values in gpu_energy_dict.items():
        energy_joules = values["min_energy"]
        latency_seconds = values["latency"]
        gpu_edp_dict[net_name] = {
            "min_energy": energy_joules * latency_seconds,
            "latency": values["latency"]
        }
    return gpu_edp_dict

def calculate_gpu_edp_x_cost_dict(gpu_energy_dict, gpu_cost:float):
    gpu_edp_x_cost_dict = {}
    for net_name, values in gpu_energy_dict.items():
        energy_joules = values["min_energy"]
        latency_seconds = values["latency"]
        gpu_edp_x_cost_dict[net_name] = {
            "min_energy": energy_joules * latency_seconds * gpu_cost,
            "latency": values["latency"]
        }
    return gpu_edp_x_cost_dict

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