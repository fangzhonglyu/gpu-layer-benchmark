"""
Benchmark GPU-to-GPU P2P (NCCL send/recv) link: latency + bandwidth by default, and
transfer energy (pJ/bit) via steady-state power differencing under --power.

Energy is ONLY measured with --power (P_active - P_idle over a saturated link). The old
energy-counter subtraction (total board energy - idle*t) was removed: at these transfer
sizes the NVML energy counter is too coarse and the delta read garbage (even negative).

Usage:
    # latency/BW sweep, 2 GPU, default sizes
    torchrun --nproc_per_node=2 p2p_energy_benchmark.py

    # custom sizes (bytes), custom iters
    torchrun --nproc_per_node=2 p2p_energy_benchmark.py --sizes 1048576 67108864 536870912 --iters 200

    # transfer energy (pj/bit) via steady-state power differencing
    torchrun --nproc_per_node=2 p2p_energy_benchmark.py --power --json out.json

    # specify output dir
    torchrun --nproc_per_node=2 p2p_energy_benchmark.py --output_dir benchmarks/p2p
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
from pynvml import (nvmlInit, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetPowerUsage, nvmlDeviceGetClockInfo, NVML_CLOCK_SM)


def _median(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


CHECK_EVERY = 8   # active-phase batches between stop-flag broadcasts (see power_diff_phase)


def power_diff_phase(rank, tensor, active, ramp_s, sample_s, sample_dt, batch, handles):
    """One steady-state phase. ACTIVE: both ranks stream NCCL send/recv batches until the
    (ramp_s + sample_s) window elapses; IDLE: both just wait. Rank 0 samples instantaneous
    board power + SM clock over the post-ramp steady window. Returns
    (median_power_W, median_sm_MHz, bw_Bps) on rank 0, else (None, None, None).

    Stop is TIME-DRIVEN but checked only every CHECK_EVERY batches: rank 0 owns the clock
    and BROADCASTS a stop flag once per group of batches, and both ranks run the identical
    number of batches per group, so they always stop on the same group -- no send/recv
    mismatch/deadlock, and no dependence on a guessed bandwidth (the old fixed-count
    est_bw=40e9 undershot a faster link and read 0 W; a per-BATCH broadcast instead
    over-synced and throttled throughput off the deep-pipeline peak, inflating pj/bit).
    Grouping amortizes the broadcast so throughput stays near saturation while the loop is
    still robust to any chunk size / link speed."""
    dev = tensor.device
    nbytes = batch * tensor.numel() * tensor.element_size()   # bytes per batch
    ps = []; cs = []; moved = 0

    def one_batch():
        reqs = [dist.isend(tensor, 1) if rank == 0 else dist.irecv(tensor, 0)
                for _ in range(batch)]
        for r in reqs:
            r.wait()

    def sample(t0):
        if rank == 0:
            el = time.perf_counter() - t0
            if ramp_s <= el <= ramp_s + sample_s:
                ps.append(sum(nvmlDeviceGetPowerUsage(h) for h in handles) / 1000.0)
                cs.append(max(nvmlDeviceGetClockInfo(h, NVML_CLOCK_SM) for h in handles))

    dist.barrier()
    t0 = time.perf_counter()
    if active:
        stop = torch.zeros(1, device=dev)
        while True:
            if rank == 0 and time.perf_counter() - t0 >= ramp_s + sample_s:
                stop.fill_(1.0)
            dist.broadcast(stop, src=0)            # both ranks agree when to stop (no deadlock)
            if stop.item() >= 0.5:
                break
            for _ in range(CHECK_EVERY):
                one_batch()
                moved += nbytes
                sample(t0)
    else:
        while time.perf_counter() - t0 < ramp_s + sample_s:
            time.sleep(sample_dt)
            sample(t0)
    dur = time.perf_counter() - t0

    dist.barrier()
    if rank == 0:
        return (_median(ps) if ps else 0.0, _median(cs) if cs else 0.0,
                (moved / dur) if active else 0.0)
    return None, None, None


def run_power_diff(rank, handles, chunk_bytes, ramp_s, sample_s, sample_dt, repeats,
                   batch=16):
    """Steady-state power-differencing pj/bit over NCCL (the SM-kernel-driven transfer a
    real pipeline uses). Reliable replacement for the energy-counter subtraction."""
    tensor = torch.empty(chunk_bytes // 2, dtype=torch.float16, device=f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}")
    if rank == 0:
        tensor.normal_()
    rows = []
    for _ in range(repeats):
        pa, ca, bw = power_diff_phase(rank, tensor, True, ramp_s, sample_s, sample_dt,
                                      batch, handles)
        pi, ci, _ = power_diff_phase(rank, tensor, False, ramp_s, sample_s, sample_dt,
                                     batch, handles)
        if rank == 0:
            rows.append((pa, pi, bw, ca, ci))
            print(f"[power] active {pa:.1f} W (SM {ca:.0f} MHz) | idle {pi:.1f} W (SM {ci:.0f} MHz) "
                  f"| delta {pa-pi:.1f} W | {bw/1e9:.1f} GB/s")
    if rank != 0:
        return None
    p_active = _median([r[0] for r in rows]); p_idle = _median([r[1] for r in rows])
    bw_Bps = _median([r[2] for r in rows])
    sm_active = _median([r[3] for r in rows]); sm_idle = _median([r[4] for r in rows])
    transfer_W = p_active - p_idle
    pj = transfer_W / (bw_Bps * 8) * 1e12 if bw_Bps > 0 else float("nan")
    print(f"[power] STEADY-STATE (NCCL): active {p_active:.1f} W, idle {p_idle:.1f} W, "
          f"transfer {transfer_W:.1f} W @ {bw_Bps/1e9:.1f} GB/s -> {pj:.1f} pJ/bit "
          f"| SM active {sm_active:.0f} vs idle {sm_idle:.0f} MHz")
    return {"pj_per_bit": pj, "transfer_W": transfer_W, "p_active_W": p_active,
            "p_idle_W": p_idle, "bw_GBps": bw_Bps / 1e9, "sm_active_mhz": sm_active,
            "sm_idle_mhz": sm_idle, "method": "nccl_power_diff"}


def benchmark_p2p(data_bytes, rank, iters=100):
    """Benchmark P2P send/recv LATENCY + BANDWIDTH between rank 0 and rank 1.
    Energy is NOT measured here -- use --power (steady-state power differencing) for
    pj/bit. Returns latency/BW on both ranks (only rank 0's is recorded by the caller)."""
    numel = data_bytes // 2  # float16 = 2 bytes
    tensor = torch.empty(numel, dtype=torch.float16, device=f"cuda:{rank}")
    if rank == 0:
        tensor.normal_()  # fill with data on sender side

    # Warmup
    for _ in range(10):
        if rank == 0:
            dist.send(tensor, dst=1)
        else:
            dist.recv(tensor, src=0)
    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark — async to avoid sync gaps between iterations
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    reqs = []
    for _ in range(iters):
        if rank == 0:
            reqs.append(dist.isend(tensor, dst=1))
        else:
            reqs.append(dist.irecv(tensor, src=0))
    for r in reqs:
        r.wait()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    avg_latency_ms = elapsed / iters * 1000.0
    bandwidth_GBps = data_bytes / (avg_latency_ms * 1e-3) / 1e9

    return {
        "data_bytes": data_bytes,
        "iters": iters,
        "avg_latency_ms": avg_latency_ms,
        "bandwidth_GBps": bandwidth_GBps,
    }


MIN_TOTAL_BYTES = 8 * 1024**3  # >=8 GB total/size so the latency average is stable over
                               # enough iterations even for the largest transfer sizes


def main():
    parser = argparse.ArgumentParser(description="P2P transfer energy benchmark")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[256 * 1024**2, 512 * 1024**2, 1024 * 1024**2, 2048 * 1024**2, 4096 * 1024**2],
                        help="Transfer sizes in bytes")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="benchmarks/p2p")
    parser.add_argument("--json", type=str, default=None,
                        help="dump per-size latency/BW points to this JSON (consumed by "
                             "characterize_pcie.py --method nccl); with --power, dump the "
                             "steady-state pj/bit result instead")
    parser.add_argument("--power", action="store_true",
                        help="measure transfer energy via steady-state POWER DIFFERENCING "
                             "(P_active - P_idle); otherwise only latency/BW are measured")
    parser.add_argument("--ramp_s", type=float, default=5.0)
    parser.add_argument("--sample_s", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--power_chunk_mb", type=float, default=128.0)
    parser.add_argument("--power_batch", type=int, default=16,
                        help="sends in flight per batch (power mode); batch*chunk = bytes in flight")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    nvmlInit()
    # NVML indexes PHYSICAL GPUs; torch local_rank indexes CUDA_VISIBLE_DEVICES. Map back
    # so power/energy is read from the right boards when CUDA_VISIBLE_DEVICES is set.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    phys = [int(x) for x in visible.split(",")] if visible else list(range(dist.get_world_size()))

    if args.power:
        handles_both = [nvmlDeviceGetHandleByIndex(p) for p in phys[:2]]   # rank 0 reads both
        res = run_power_diff(rank, handles_both, int(args.power_chunk_mb * 1024**2),
                             args.ramp_s, args.sample_s, 0.05, args.repeats, batch=args.power_batch)
        if rank == 0 and args.json:
            import json as _json
            with open(args.json, "w") as f:
                _json.dump(res, f, indent=2)
            print(f"JSON saved to {args.json}")
        dist.destroy_process_group()
        return

    gpu_name = torch.cuda.get_device_name(local_rank)
    if rank == 0:
        print(f"GPU: {gpu_name}")
        print(f"Transfer sizes: {[f'{s/1024**2:.0f} MB' for s in args.sizes]}")
        print(f"Iters per size: {args.iters}")
        print("-" * 80)
        print(f"{'Size':>10s}  {'Latency':>10s}  {'BW':>10s}")
        print(f"{'(MB)':>10s}  {'(ms)':>10s}  {'(GB/s)':>10s}")
        print("-" * 80)

    all_results = []
    for size in args.sizes:
        iters = max(args.iters, MIN_TOTAL_BYTES // size)
        if rank == 0 and iters > args.iters:
            print(f"  (auto-scaled iters to {iters} for {size/1024**2:.0f} MB)")

        dist.barrier()
        result = benchmark_p2p(size, rank, iters=iters)

        if rank == 0:
            print(f"{size/1024**2:>10.0f}  {result['avg_latency_ms']:>10.3f}  "
                  f"{result['bandwidth_GBps']:>10.2f}")
            all_results.append(result)

    # Save results
    if rank == 0:
        print("-" * 80)
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, "p2p_energy.txt")
        with open(output_file, "w") as f:
            f.write(f"GPU: {gpu_name}\n")
            f.write(f"Iters: {args.iters}\n\n")
            for r in all_results:
                f.write(f"Size: {r['data_bytes']/1024**2:.0f} MB\n")
                f.write(f"  Avg Latency:          {r['avg_latency_ms']:.3f} ms\n")
                f.write(f"  Bandwidth:            {r['bandwidth_GBps']:.2f} GB/s\n\n")
        print(f"Results saved to {output_file}")

        if args.json:
            import json as _json
            points = [{"bytes": int(r["data_bytes"]),
                       "latency_ms": r["avg_latency_ms"],
                       "bw_GBps": r["bandwidth_GBps"]} for r in all_results]
            with open(args.json, "w") as f:
                _json.dump({"method": "nccl_send_recv_latency", "points": points}, f, indent=2)
            print(f"JSON saved to {args.json}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()