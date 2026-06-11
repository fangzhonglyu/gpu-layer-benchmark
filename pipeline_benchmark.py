import os
from typing import List, Tuple
import pandas as pd

from kernels import run_pipeline, set_device, profile_device_idle_power, LinkModel, TransferCurve

def save_csv_wide(all_results, filename="summary.csv"):
    # Flatten results -> long table
    df_layers = pd.json_normalize(
        all_results,
        record_path="results",
        meta=["name", "total_pipeline_latency", "total_pipeline_energy",
              "total_pipeline_energy_with_idle", "p2p_transfer_energy",
              "total_pipeline_energy_with_idle_and_p2p",
              "total_pipeline_latency_with_xfer", "bottleneck_latency",
              "bottleneck_latency_with_xfer", "max_transfer_latency_ms",
              "sum_transfer_latency_ms", "transfer_bound"],
        record_prefix="layer_",
        meta_prefix="pipeline_",
    )

    # Pivot to wide: metric__layer columns
    wide = (df_layers.pivot_table(
                index="pipeline_name",
                columns="layer_name",
                values=["layer_avg_latency_ms", "layer_avg_energy_J", "layer_avg_power_W"],
                aggfunc="first")
            .sort_index(axis=1, level=1))

    # Flatten MultiIndex columns
    wide.columns = [f"{metric}__{layer}" for metric, layer in wide.columns]
    wide = wide.reset_index().rename(columns={"pipeline_name": "pipeline"})

    # Add pipeline totals
    totals = (df_layers[["pipeline_name", "pipeline_total_pipeline_latency",
                         "pipeline_total_pipeline_energy",
                         "pipeline_total_pipeline_energy_with_idle",
                         "pipeline_p2p_transfer_energy",
                         "pipeline_total_pipeline_energy_with_idle_and_p2p",
                         "pipeline_total_pipeline_latency_with_xfer",
                         "pipeline_bottleneck_latency",
                         "pipeline_bottleneck_latency_with_xfer",
                         "pipeline_max_transfer_latency_ms",
                         "pipeline_sum_transfer_latency_ms",
                         "pipeline_transfer_bound"]]
              .drop_duplicates()
              .rename(columns={
                  "pipeline_total_pipeline_latency": "pipeline_latency_ms",
                  "pipeline_total_pipeline_energy": "pipeline_energy_J",
                  "pipeline_total_pipeline_energy_with_idle": "pipeline_energy_with_idle_J",
                  "pipeline_p2p_transfer_energy": "pipeline_p2p_energy_J",
                  "pipeline_total_pipeline_energy_with_idle_and_p2p": "pipeline_energy_with_idle_and_p2p_J",
                  "pipeline_total_pipeline_latency_with_xfer": "pipeline_latency_with_xfer_ms",
                  "pipeline_bottleneck_latency": "pipeline_bottleneck_ms",
                  "pipeline_bottleneck_latency_with_xfer": "pipeline_bottleneck_with_xfer_ms",
                  "pipeline_max_transfer_latency_ms": "pipeline_max_xfer_ms",
                  "pipeline_sum_transfer_latency_ms": "pipeline_sum_xfer_ms",
                  "pipeline_transfer_bound": "pipeline_transfer_bound",
              }))
    wide = wide.merge(totals, left_on="pipeline", right_on="pipeline_name", how="left") \
               .drop(columns=["pipeline_name"])

    num_cols = [c for c in wide.columns
            if c.startswith(("layer_avg_latency_ms__", "layer_avg_energy_J__", "layer_avg_power_W__"))
            or c in ("pipeline_latency_ms", "pipeline_energy_J", "pipeline_energy_with_idle_J",
                     "pipeline_p2p_energy_J", "pipeline_energy_with_idle_and_p2p_J",
                     "pipeline_latency_with_xfer_ms", "pipeline_bottleneck_ms",
                     "pipeline_bottleneck_with_xfer_ms", "pipeline_max_xfer_ms",
                     "pipeline_sum_xfer_ms")]

    wide[num_cols] = wide[num_cols].apply(pd.to_numeric, errors="coerce")
    wide[num_cols] = wide[num_cols].fillna(0.0)

    # Save
    wide.to_csv(filename, index=False)
    return wide

def pipeline_benchmark(output_dir:str, pipelines:List[Tuple], device_index:int = 0,
                       link_model:LinkModel = None,
                       transfer_curve:TransferCurve = None) -> List[dict]:
    """`link_model`: optional LinkModel for step-④ inter-card transfer latency. When
    None, transfer is energy-only (legacy). Build one with kernels.measure_pcie_link
    (>=2 GPUs), LinkModel.load(path), or LinkModel(bw_GBps=..., kind='manual').
    `transfer_curve`: optional size-dependent TransferCurve (preferred over link_model);
    makes per-edge bw AND pj/bit depend on the activation size. Load via
    kernels.transfer_curve_from_env or TransferCurve.measured_nccl()."""
    set_device(device_index)

    # profile_device_idle_power() # Assume 45 W idle power for now, remove inconsistency in measurements

    if transfer_curve is not None:
        print(f"[curve] step-4 transfer modelled: {transfer_curve.kind} size-dependent "
              f"(pj/bit {min(transfer_curve.pj_per_bit):.0f}-{max(transfer_curve.pj_per_bit):.0f}, "
              f"bw {min(transfer_curve.bw_GBps):.0f}-{max(transfer_curve.bw_GBps):.0f} GB/s)")
    elif link_model is not None:
        print(f"[link] step-4 transfer modelled: {link_model.kind} "
              f"{link_model.bw_GBps:.1f} GB/s, {link_model.latency_us:.1f} us")
    else:
        print("[link] step-4 transfer NOT modelled (latency assumed 0); pass link_model/transfer_curve=...")

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for pipeline in pipelines:
        # Each pipeline is (name, phases[, levels[, edges]]): `levels` is the
        # operator-DAG stage schedule (critical-path timing) and `edges` is the DAG
        # edge list (inter-chiplet P2P transfer energy).
        name, phases, *rest = pipeline
        levels = rest[0] if len(rest) > 0 else None
        edges = rest[1] if len(rest) > 1 else None
        all_results.append(run_pipeline(name, phases, output_dir, levels=levels, edges=edges,
                                        link_model=link_model, transfer_curve=transfer_curve))

    csv_file = os.path.join(output_dir, "summary.csv")
    save_csv_wide(all_results, filename=csv_file)
    print(f"Result summary saved to {csv_file}")

    return all_results