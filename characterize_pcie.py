"""Characterize the real GPU->GPU link once, fit a LinkModel (bw, fixed latency,
pj/bit), and save it to JSON. The model benchmarks load it via PCIE_LINK_JSON to fold
step-④ inter-card transfer latency + energy into the pipeline (see kernels.combine_dag).

TWO transfer mechanisms -> pick one and stay consistent (BW and energy must come from
the SAME mechanism):

  --method nccl  (default, REALISTIC):  measures via NCCL send/recv (torchrun, 2 procs),
        the way real pipeline-parallel frameworks (Megatron/DeepSpeed/vLLM) move inter-
        stage activations. Gives BW (~48 GB/s on gl1803), fixed latency, AND both-side
        pj/bit (~185), ALL read from one measurement -- nothing hardcoded. The pj/bit
        here is the real extra cost of an activation hop (src GDDR re-read + link +
        dst GDDR write), which is NOT double-counted with per-op energy.

  --method dma   (copy-engine lower bound):  single-process cudaMemcpyPeer. Higher BW
        (~54 GB/s, no NCCL kernel overhead) and lower pj/bit (~54), but frameworks don't
        use this path. Use only if modelling an idealized copy-engine transfer.

Usage (node with >=2 GPUs):
    python characterize_pcie.py                       # nccl, src=0 dst=1 -> benchmarks/pcie_link.json
    python characterize_pcie.py --method dma
    python characterize_pcie.py --src 0 --dst 1 --out benchmarks/pcie_link.json

Then:
    PCIE_LINK_JSON=benchmarks/pcie_link.json python model_benchmarks/llama3_1_8b_benchmark.py

1 GPU? Write a model by hand:
    python -c "from kernels import LinkModel; LinkModel(bw_GBps=48, latency_us=5, pj_per_bit=185, kind='manual').save('benchmarks/pcie_link.json')"
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch

from kernels import LinkModel, measure_pcie_link, measure_pcie_transfer_power_pj, set_device

HERE = os.path.dirname(os.path.abspath(__file__))


def _torchrun_p2p(src, dst, master_port, extra_args):
    """Run p2p_energy_benchmark.py under torchrun (2 procs) on GPUs {src,dst}, dumping to a
    temp JSON which is read back and returned. Raises on failure."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = f"{src},{dst}"   # ranks 0,1 -> physical src,dst
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
           f"--master_port={master_port}", os.path.join(HERE, "p2p_energy_benchmark.py"),
           *extra_args, "--json", tmp]
    subprocess.run(cmd, env=env, check=True)
    with open(tmp) as f:
        return json.load(f)


def measure_via_nccl(src: int, dst: int, sizes_bytes, iters: int, master_port: int,
                     ramp_s: float, sample_s: float, repeats: int):
    """Characterize the NCCL link with TWO real measurements (nothing hardcoded):
      1. latency sweep  -> per-size latency points -> fit BW + fixed latency
      2. --power run    -> steady-state power-differencing pj/bit (the reliable energy;
                           the old energy-counter pj in the sweep JSON is ignored)
    Returns a LinkModel, or None on failure (torchrun unavailable etc.)."""
    # NCCL auto-scales iters to ~GBs/size; sub-MB sizes -> huge iter counts. Keep >=1MB.
    sizes_bytes = [b for b in sizes_bytes if b >= (1 << 20)] or [1 << 20, 1 << 24, 1 << 28]
    try:
        print(f"[nccl] (1/2) latency/BW sweep on GPUs {src},{dst} ...")
        lat = _torchrun_p2p(src, dst, master_port,
                            ["--sizes", *[str(b) for b in sizes_bytes], "--iters", str(iters)])
        print(f"[nccl] (2/2) steady-state power-diff pj/bit on GPUs {src},{dst} ...")
        pwr = _torchrun_p2p(src, dst, master_port + 1,
                            ["--power", "--ramp_s", str(ramp_s), "--sample_s", str(sample_s),
                             "--repeats", str(repeats)])
    except Exception as e:
        print(f"[nccl] measurement failed: {e}\n"
              f"       run manually:  CUDA_VISIBLE_DEVICES={src},{dst} torchrun "
              f"--nproc_per_node=2 p2p_energy_benchmark.py [--power] --json out.json")
        return None
    pts = [(p["bytes"], p["latency_ms"]) for p in lat["points"]]
    model = LinkModel.fit(pts, kind="pcie-nccl", pj_per_bit=pwr["pj_per_bit"])
    print(f"[nccl] fit: {model.bw_GBps:.1f} GB/s, {model.latency_us:.1f} us fixed | "
          f"pj/bit {model.pj_per_bit:.1f} (power-diff: {pwr['transfer_W']:.1f} W transfer @ "
          f"{pwr['bw_GBps']:.1f} GB/s, SM {pwr['sm_active_mhz']:.0f} vs {pwr['sm_idle_mhz']:.0f} MHz)")
    return model


def main():
    ap = argparse.ArgumentParser(description="Characterize GPU-GPU link -> LinkModel JSON")
    ap.add_argument("--method", choices=["nccl", "dma"], default="nccl",
                    help="nccl (realistic, default) or dma (copy-engine lower bound)")
    ap.add_argument("--src", type=int, default=0, help="source GPU index")
    ap.add_argument("--dst", type=int, default=1, help="destination GPU index")
    ap.add_argument("--iters", type=int, default=200, help="copies/iters per size")
    ap.add_argument("--out", type=str, default="benchmarks/pcie_link.json")
    ap.add_argument("--sizes_mb", type=float, nargs="+", default=None,
                    help="transfer sizes in MB (default sweep ~0.004 .. 256 MB)")
    ap.add_argument("--master_port", type=int, default=29577, help="torchrun port (nccl)")
    ap.add_argument("--ramp_s", type=float, default=5.0, help="power-diff ramp-to-steady (dma)")
    ap.add_argument("--sample_s", type=float, default=10.0, help="power-diff sample window (dma)")
    ap.add_argument("--repeats", type=int, default=3, help="power-diff active/idle repeats (dma)")
    args = ap.parse_args()

    n = torch.cuda.device_count()
    if n < 2:
        raise SystemExit(f"Found {n} GPU(s); need >=2. See the module docstring for the "
                         f"1-GPU fallback (write a LinkModel by hand).")

    # default size sweep spans small->large so the latency fit gets a good fixed-latency
    # intercept AND the energy steady-state (large) value
    if args.sizes_mb:
        sizes = [int(mb * 1024 * 1024) for mb in args.sizes_mb]
    else:
        sizes = [1 << 12, 1 << 16, 1 << 20, 1 << 24, 1 << 26, 1 << 28]  # 4KB..256MB

    if args.method == "nccl":
        model = measure_via_nccl(args.src, args.dst, sizes, args.iters, args.master_port,
                                 args.ramp_s, args.sample_s, args.repeats)
        if model is None:
            raise SystemExit(1)
    else:  # dma
        set_device(args.src)
        print(f"GPU {args.src}: {torch.cuda.get_device_name(args.src)}")
        print(f"GPU {args.dst}: {torch.cuda.get_device_name(args.dst)}")
        model = measure_pcie_link(src=args.src, dst=args.dst, sizes_bytes=sizes,
                                  iters=args.iters, kind="pcie-dma")
        # pj/bit by STEADY-STATE POWER DIFFERENCING (P_active - P_idle over a saturated
        # copy-engine link); reliable, unlike an NVML energy-counter delta at these sizes.
        e = measure_pcie_transfer_power_pj(src=args.src, dst=args.dst,
                                           ramp_s=args.ramp_s, sample_s=args.sample_s,
                                           repeats=args.repeats)
        model.pj_per_bit = e["pj_per_bit"]
        print(f"[dma] pj/bit {model.pj_per_bit:.1f} = {e['transfer_W']:.1f} W transfer "
              f"@ {e['bw_GBps']:.1f} GB/s (copy-engine link+GDDR movement, no NCCL SM)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print(f"\nExample transfer times with this link:")
    for mb in (1, 16, 64, 256):
        print(f"  {mb:4d} MB -> {model.latency_ms(mb*1024*1024):.4f} ms")
    print(f"\nUse it:  PCIE_LINK_JSON={args.out} python model_benchmarks/<model>_benchmark.py")


if __name__ == "__main__":
    main()
