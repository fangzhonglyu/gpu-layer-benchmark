"""
Benchmark GPU-to-GPU P2P transfer energy cost (pJ/bit).

Usage:
    # 2 GPU, default sizes
    torchrun --nproc_per_node=2 p2p_energy_benchmark.py

    # custom sizes (bytes), custom iters
    torchrun --nproc_per_node=2 p2p_energy_benchmark.py --sizes 1048576 67108864 536870912 --iters 200

    # specify output dir
    torchrun --nproc_per_node=2 p2p_energy_benchmark.py --output_dir benchmarks/p2p
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetTotalEnergyConsumption


def measure_idle_power(handle, duration=5.0):
    """Measure GPU idle power (W) over a given duration."""
    torch.cuda.synchronize()
    time.sleep(1)  # let GPU settle

    e0 = nvmlDeviceGetTotalEnergyConsumption(handle)
    time.sleep(duration)
    e1 = nvmlDeviceGetTotalEnergyConsumption(handle)

    energy_J = (e1 - e0) / 1000.0
    return energy_J / duration  # Watts


def benchmark_p2p(data_bytes, handle, rank, iters=100):
    """
    Benchmark P2P send/recv between rank 0 and rank 1.
    Returns (avg_latency_ms, transfer_energy_J_per_iter, bandwidth_GBps) on rank 0.
    """
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

    # Measure idle baseline
    idle_power = measure_idle_power(handle, duration=3.0)
    dist.barrier()

    # Benchmark — async to avoid sync gaps between iterations
    torch.cuda.synchronize()
    e0 = nvmlDeviceGetTotalEnergyConsumption(handle)
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
    e1 = nvmlDeviceGetTotalEnergyConsumption(handle)

    elapsed = t1 - t0
    total_energy_J = (e1 - e0) / 1000.0

    # Subtract idle baseline to get transfer-only energy
    transfer_energy_J = total_energy_J - idle_power * elapsed

    avg_latency_ms = elapsed / iters * 1000.0
    bandwidth_GBps = data_bytes / (avg_latency_ms * 1e-3) / 1e9

    return {
        "data_bytes": data_bytes,
        "iters": iters,
        "idle_power_W": idle_power,
        "total_energy_J": total_energy_J,
        "transfer_energy_J": transfer_energy_J,
        "transfer_energy_per_iter_J": transfer_energy_J / iters,
        "energy_per_bit_pJ": transfer_energy_J / (data_bytes * iters * 8) * 1e12,
        "avg_latency_ms": avg_latency_ms,
        "bandwidth_GBps": bandwidth_GBps,
    }


MIN_TOTAL_BYTES = 2 * 1024**3  # 2 GB total transfer to ensure reliable NVML sampling


def main():
    parser = argparse.ArgumentParser(description="P2P transfer energy benchmark")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[256 * 1024**2, 512 * 1024**2, 1024 * 1024**2, 2048 * 1024**2, 4096 * 1024**2],
                        help="Transfer sizes in bytes")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="benchmarks/p2p")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(local_rank)

    gpu_name = torch.cuda.get_device_name(local_rank)
    if rank == 0:
        print(f"GPU: {gpu_name}")
        print(f"Transfer sizes: {[f'{s/1024**2:.0f} MB' for s in args.sizes]}")
        print(f"Iters per size: {args.iters}")
        print("-" * 80)
        print(f"{'Size':>10s}  {'Latency':>10s}  {'BW':>10s}  {'E/iter':>12s}  "
              f"{'E/bit':>10s}  {'Idle P':>8s}")
        print(f"{'(MB)':>10s}  {'(ms)':>10s}  {'(GB/s)':>10s}  {'(mJ)':>12s}  "
              f"{'(pJ)':>10s}  {'(W)':>8s}")
        print("-" * 80)

    all_results = []
    for size in args.sizes:
        iters = max(args.iters, MIN_TOTAL_BYTES // size)
        if rank == 0 and iters > args.iters:
            print(f"  (auto-scaled iters to {iters} for {size/1024**2:.0f} MB)")

        dist.barrier()
        result = benchmark_p2p(size, handle, rank, iters=iters)

        # Gather energy from both ranks (sender + receiver)
        local_transfer_energy = torch.tensor([result["transfer_energy_J"]], device=f"cuda:{local_rank}")
        all_energies = [torch.zeros(1, device=f"cuda:{local_rank}") for _ in range(2)]
        dist.all_gather(all_energies, local_transfer_energy)

        combined_transfer_energy = sum(e.item() for e in all_energies)
        result["transfer_energy_both_J"] = combined_transfer_energy
        result["energy_per_bit_both_pJ"] = combined_transfer_energy / (size * args.iters * 8) * 1e12

        if rank == 0:
            print(f"{size/1024**2:>10.0f}  {result['avg_latency_ms']:>10.3f}  "
                  f"{result['bandwidth_GBps']:>10.2f}  "
                  f"{combined_transfer_energy/args.iters*1000:>12.4f}  "
                  f"{result['energy_per_bit_both_pJ']:>10.2f}  "
                  f"{result['idle_power_W']:>8.1f}")
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
                f.write(f"  Bandwidth:            {r['bandwidth_GBps']:.2f} GB/s\n")
                f.write(f"  Sender Transfer E:    {r['transfer_energy_J']:.6f} J\n")
                f.write(f"  Both-side Transfer E: {r['transfer_energy_both_J']:.6f} J\n")
                f.write(f"  Energy/bit (sender):  {r['energy_per_bit_pJ']:.2f} pJ\n")
                f.write(f"  Energy/bit (both):    {r['energy_per_bit_both_pJ']:.2f} pJ\n")
                f.write(f"  Idle Power:           {r['idle_power_W']:.1f} W\n\n")
        print(f"Results saved to {output_file}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()