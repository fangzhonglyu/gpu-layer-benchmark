"""Merge every per-run benchmarks/**/summary.csv into one combined CSV.

Each pipeline_benchmark run writes a WIDE summary.csv to its own output_dir; different
models share the pipeline_* total columns but have different per-layer (metric__layer)
columns. This outer-concats them (union of columns, NaN where a model lacks a layer),
tags each row with its source run dir, and puts the common totals first.

Usage:  python merge_summaries.py [glob] [-o out.csv] [--totals-only]
        python merge_summaries.py                       # benchmarks/**/summary.csv -> benchmarks/combined_summary.csv
        python merge_summaries.py --totals-only          # only source/pipeline + pipeline_* totals (tidy cross-model table)
"""
import argparse
import glob
import os

import pandas as pd

# Columns every summary.csv shares (written by pipeline_benchmark.save_csv_wide).
TOTAL_COLS = [
    "pipeline", "pipeline_latency_ms", "pipeline_energy_J", "pipeline_energy_with_idle_J",
    "pipeline_p2p_energy_J", "pipeline_energy_with_idle_and_p2p_J",
    "pipeline_latency_with_xfer_ms", "pipeline_bottleneck_ms", "pipeline_bottleneck_with_xfer_ms",
    "pipeline_max_xfer_ms", "pipeline_sum_xfer_ms", "pipeline_transfer_bound",
]


def main():
    ap = argparse.ArgumentParser(description="Merge per-run summary.csv files into one CSV")
    ap.add_argument("pattern", nargs="?", default="benchmarks/**/summary.csv",
                    help="glob for the summary files (default benchmarks/**/summary.csv)")
    ap.add_argument("-o", "--out", default="benchmarks/combined_summary.csv")
    ap.add_argument("--totals-only", action="store_true",
                    help="keep only source/pipeline + pipeline_* totals (drop per-layer columns)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern, recursive=True))
    if not files:
        raise SystemExit(f"no summary.csv matched {args.pattern!r} (run the benchmarks first)")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.insert(0, "source", os.path.basename(os.path.dirname(f)))  # e.g. llama3.1_8b_decode
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True, sort=False)       # union of columns

    if args.totals_only:
        keep = ["source"] + [c for c in TOTAL_COLS if c in combined.columns]
        combined = combined[keep]
    else:
        # common totals first, then the per-layer columns
        front = ["source"] + [c for c in TOTAL_COLS if c in combined.columns]
        rest = [c for c in combined.columns if c not in front]
        combined = combined[front + rest]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"merged {len(files)} summaries -> {args.out}  "
          f"({combined.shape[0]} rows, {combined.shape[1]} cols)")
    for f in files:
        print(f"  + {f}")


if __name__ == "__main__":
    main()
