"""Sweep inter-card transfer SIZE and characterize bandwidth + energy-per-bit at each,
producing a size-dependent TransferCurve JSON. Unlike characterize_pcie.py (one scalar
bw/latency/pj LinkModel), this captures the strong size dependence the model needs:
small transfers are latency-bound (bandwidth collapses -> pj/bit explodes), there is an
efficiency sweet spot (~8-32 MB on the measured RTX PRO 6000 / PCIe NCCL link), and large
transfers plateau at a higher pj/bit. See kernels.TransferCurve / pcie_pjbit_curve.png.

Each size is measured by p2p_energy_benchmark.py --power (steady-state power-differencing
over NCCL send/recv -- the path PP frameworks use). The in-flight batch is sized so
batch*size ~ 256 MB: pj/bit is batch-INDEPENDENT (verified), this just keeps the link
saturated for small sizes and the stop-check granularity sane for multi-GB sizes.

Usage (node with >=2 GPUs):
    python characterize_transfer_curve.py --src 0 --dst 1 --out benchmarks/pcie_transfer_curve.json
    python characterize_transfer_curve.py --sizes_mb 1 16 64 256 1024
Then:
    PCIE_CURVE_JSON=benchmarks/pcie_transfer_curve.json python model_benchmarks/<model>.py

The baked-in kernels.TransferCurve.measured_nccl() is a saved run of this on gl1808; use
it (transfer_curve_from_env(default_measured=True)) when no GPU is available.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch

from kernels import TransferCurve

HERE = os.path.dirname(os.path.abspath(__file__))
# 64 KB .. 16 GB, log-spaced (the range over which pj/bit spans ~8x)
DEFAULT_SIZES_MB = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096, 16384]


def _measure_one(src, dst, size_mb, master_port):
    """Run p2p_energy_benchmark.py --power for one size on GPUs {src,dst}; return its
    {pj_per_bit, bw_GBps, ...} JSON. Batch sized so batch*size ~ 256 MB (cap 512)."""
    batch = max(1, min(512, int(256 / size_mb))) if size_mb > 0 else 16
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = f"{src},{dst}"          # ranks 0,1 -> physical src,dst
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
           f"--master_port={master_port}", os.path.join(HERE, "p2p_energy_benchmark.py"),
           "--power", "--power_chunk_mb", str(size_mb), "--power_batch", str(batch),
           "--ramp_s", "3", "--sample_s", "6", "--repeats", "2", "--json", tmp]
    subprocess.run(cmd, env=env, check=True)
    with open(tmp) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Characterize size-dependent GPU-GPU transfer curve")
    ap.add_argument("--src", type=int, default=0)
    ap.add_argument("--dst", type=int, default=1)
    ap.add_argument("--sizes_mb", type=float, nargs="+", default=None,
                    help="transfer sizes in MB (default 64KB..16GB log sweep)")
    ap.add_argument("--out", type=str, default="benchmarks/pcie_transfer_curve.json")
    ap.add_argument("--master_port", type=int, default=29600)
    args = ap.parse_args()

    if torch.cuda.device_count() < 2:
        raise SystemExit(f"found {torch.cuda.device_count()} GPU(s); need >=2. Use the "
                         f"baked-in kernels.TransferCurve.measured_nccl() instead.")
    sizes = args.sizes_mb or DEFAULT_SIZES_MB
    print(f"[curve] sweeping {len(sizes)} sizes on GPUs {args.src},{args.dst} ...")
    pts = []
    for i, mb in enumerate(sizes):
        r = _measure_one(args.src, args.dst, mb, args.master_port + i)
        nbytes = int(mb * 1024 * 1024)
        pts.append((nbytes, r["bw_GBps"], r["pj_per_bit"]))
        print(f"  {mb:>9.4f} MB -> {r['bw_GBps']:6.1f} GB/s, {r['pj_per_bit']:6.1f} pJ/bit")

    pts.sort()
    curve = TransferCurve(sizes_bytes=[p[0] for p in pts], bw_GBps=[p[1] for p in pts],
                          pj_per_bit=[p[2] for p in pts], kind="nccl")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    curve.save(args.out)
    print(f"\nUse it:  PCIE_CURVE_JSON={args.out} python model_benchmarks/<model>.py")


if __name__ == "__main__":
    main()
