import os
from typing import List, Tuple
import pandas as pd

from kernels import run_pipeline, set_device, profile_device_idle_power

def save_csv_wide(all_results, filename="summary.csv"):
    # Flatten results -> long table
    df_layers = pd.json_normalize(
        all_results,
        record_path="results",
        meta=["name", "total_pipeline_latency", "total_pipeline_energy"],
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
    totals = (df_layers[["pipeline_name", "pipeline_total_pipeline_latency", "pipeline_total_pipeline_energy"]]
              .drop_duplicates()
              .rename(columns={
                  "pipeline_total_pipeline_latency": "pipeline_latency_ms",
                  "pipeline_total_pipeline_energy": "pipeline_energy_J",
              }))
    wide = wide.merge(totals, left_on="pipeline", right_on="pipeline_name", how="left") \
               .drop(columns=["pipeline_name"])

    num_cols = [c for c in wide.columns
            if c.startswith(("layer_avg_latency_ms__", "layer_avg_energy_J__", "layer_avg_power_W__"))
            or c in ("pipeline_latency_ms", "pipeline_energy_J")]

    wide[num_cols] = wide[num_cols].apply(pd.to_numeric, errors="coerce")
    wide[num_cols] = wide[num_cols].fillna(0.0)

    # Save
    wide.to_csv(filename, index=False)
    return wide

def pipeline_benchmark(output_dir:str, pipelines:List[Tuple], device_index:int = 0) -> List[dict]:
    set_device(device_index)

    # profile_device_idle_power() # Assume 45 W idle power for now, remove inconsistency in measurements
    
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for pipeline in pipelines:
        name, phases = pipeline
        all_results.append(run_pipeline(name, phases, output_dir))

    csv_file = os.path.join(output_dir, "summary.csv")
    save_csv_wide(all_results, filename=csv_file)
    print(f"Result summary saved to {csv_file}")

    return all_results