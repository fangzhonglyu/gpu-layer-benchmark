import torch
import time
import os
import json
import math
import shutil
from dataclasses import dataclass, asdict
from typing import List, Tuple, Callable, Optional, Dict

from pynvml import (nvmlDeviceGetTotalEnergyConsumption, nvmlInit,
                    nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage,
                    nvmlDeviceGetClockInfo, NVML_CLOCK_SM)
import torch.nn.functional as F


COLS = shutil.get_terminal_size().columns
GREEN_DOT = "\033[32m.\033[0m"
GPU_IDLE_POWER = 45  # Assumed idle power consumption in Watts

# Inter-chiplet P2P transfer model (NVML/measured energy covers on-package HBM IO
# but NOT chiplet-to-chiplet links, so it is added here from the operator DAG edges).
DEFAULT_PJ_PER_BIT = 200.0  # energy per transferred bit across a chiplet link
DTYPE_BYTES = 2             # float16 activations

handle = None  # Global variable to hold NVML handle


# --- Inter-card transfer (step ④) latency model -----------------------------
# Step ④ (move a stage's output activation to the next card's DRAM) uses the
# copy engine / PCIe link, NOT the compute engine, so it can overlap compute.
# Whether it is actually HIDDEN depends on whether its time exceeds the slowest
# compute stage -- which is only knowable if we model its latency. The old code
# charged P2P as energy only and silently assumed t_transfer == 0; this models
# it as t(bytes) = fixed_latency + bytes / bandwidth, where the two constants are
# fit from a REAL on-machine PCIe measurement (measure_pcie_link), not guessed.
@dataclass
class LinkModel:
    """Linear transfer-latency model for one inter-card link.

        t_ms(bytes) = latency_us/1e3 + bytes / (bw_GBps * 1e6)

    bw_GBps in GB/s (1e9 byte/s); latency_us is the fixed per-transfer overhead
    (launch + link/propagation) that dominates small transfers.
    """
    bw_GBps: float
    latency_us: float = 0.0
    kind: str = "pcie"          # provenance tag: 'pcie' (measured), 'manual', ...
    pj_per_bit: Optional[float] = None   # measured inter-card transfer energy; None -> use default

    def latency_ms(self, n_bytes: float) -> float:
        return self.latency_us / 1e3 + n_bytes / (self.bw_GBps * 1e6)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        pj = f", {self.pj_per_bit:.0f} pJ/bit" if self.pj_per_bit else ""
        print(f"[link] saved {self.kind} model -> {path}  "
              f"({self.bw_GBps:.1f} GB/s, {self.latency_us:.1f} us{pj})")

    @staticmethod
    def load(path: str) -> "LinkModel":
        with open(path) as f:
            return LinkModel(**json.load(f))

    @staticmethod
    def fit(points: List[Tuple[float, float]], kind: str = "pcie",
            pj_per_bit: Optional[float] = None) -> "LinkModel":
        """Fit ms = latency + bytes/BW from [(bytes, ms), ...] -> LinkModel.

        BW comes from the OLS slope (bulk transfers determine it well). The fixed
        latency, however, is NOT taken from the OLS intercept: with x spanning orders
        of magnitude the largest transfer dominates the least-squares and washes the
        intercept to ~0. Instead it is the residual of the SMALLEST transfer
        (t_min - bytes_min/BW), where the fixed overhead is actually visible."""
        n = len(points)
        sx = sum(b for b, _ in points); sy = sum(m for _, m in points)
        sxx = sum(b * b for b, _ in points); sxy = sum(b * m for b, m in points)
        denom = n * sxx - sx * sx
        slope = (n * sxy - sx * sy) / denom if denom else 0.0
        bw_GBps = 1.0 / (slope * 1e6) if slope > 0 else float("inf")
        bmin, tmin = min(points, key=lambda p: p[0])   # smallest transfer
        latency_us = max(0.0, (tmin - slope * bmin) * 1e3)
        return LinkModel(bw_GBps=bw_GBps, latency_us=latency_us, kind=kind, pj_per_bit=pj_per_bit)


@dataclass
class TransferCurve:
    """Measured inter-card transfer characterization as a FUNCTION of transfer size.

    Both bandwidth AND energy-per-bit depend strongly on the size of a single transfer
    (see benchmarks/pcie_transfer_curve.json / pcie_pjbit_curve.png). Three regimes:
      - small transfers are latency-bound: bandwidth collapses, so pj/bit explodes
        (~877 pJ/bit @ 64 KB on the measured RTX PRO 6000 / PCIe NCCL link);
      - an efficiency sweet spot (~8-32 MB) where the link is just saturated but the
        per-transfer copy power is still low (~113 pJ/bit min);
      - large transfers plateau at a higher pj/bit (~197 >= 256 MB).
    A single scalar pj/bit (LinkModel) cannot capture this ~8x spread, so combine_dag
    prefers a TransferCurve when one is supplied.

    Values are interpolated piecewise-linearly in log2(bytes) (dependency-free; no scipy)
    and clamped to the measured endpoints outside the sampled range.
    """
    sizes_bytes: List[int]
    bw_GBps: List[float]
    pj_per_bit: List[float]
    kind: str = "nccl"
    fixed_us: float = 10.0   # per-transfer latency floor (NCCL handshake + kernel launch);
                             # measured ~10 us on gl1808, flat for transfers below ~64 KB

    def _interp(self, n_bytes: float, ys: List[float]) -> float:
        x = math.log2(max(n_bytes, 1.0))
        xs = [math.log2(s) for s in self.sizes_bytes]
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if x <= xs[i]:
                t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
                return ys[i - 1] + t * (ys[i] - ys[i - 1])
        return ys[-1]

    def pj_at(self, n_bytes: float) -> float:
        """Interpolated energy-per-bit (pJ/bit) for a single transfer of n_bytes."""
        return self._interp(n_bytes, self.pj_per_bit)

    def bw_at(self, n_bytes: float) -> float:
        """Interpolated achieved bandwidth (GB/s) for a single transfer of n_bytes."""
        return self._interp(n_bytes, self.bw_GBps)

    def latency_ms(self, n_bytes: float) -> float:
        """Transfer time (ms) = max(fixed_us floor, n_bytes / bw(n_bytes)). The bw curve is
        only measured down to ~64 KB; below that a single NCCL send/recv is a flat ~fixed_us
        latency floor (handshake + kernel launch, size-INDEPENDENT -- measured ~10 us on
        gl1808). Without the floor, clamping bw to its smallest measured value underestimates
        small-transfer latency by up to ~100x (matters for decode, whose edges are 256 B-few
        MB). For large transfers n_bytes/bw dominates and the floor is negligible; max() keeps
        the small->large transition continuous (no discontinuity at the 64 KB clamp)."""
        return max(self.fixed_us / 1e3, n_bytes / (self.bw_at(n_bytes) * 1e6))

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"[curve] saved {self.kind} transfer curve -> {path} "
              f"({len(self.sizes_bytes)} points, pj/bit "
              f"{min(self.pj_per_bit):.0f}-{max(self.pj_per_bit):.0f})")

    @staticmethod
    def load(path: str) -> "TransferCurve":
        with open(path) as f:
            return TransferCurve(**json.load(f))

    @staticmethod
    def measured_nccl() -> "TransferCurve":
        """Reference curve measured on gl1808 (RTX PRO 6000 Server Edition, PCIe, 2 GPUs,
        NCCL send/recv, steady-state power-differencing; cross-checked against energy-counter
        integration to within ~5%). Baked in so the model has a default even with no GPU."""
        pts = [
            (65536,        4.2, 877.0), (131072,       8.4, 468.0),
            (262144,      14.8, 288.0), (524288,      24.0, 195.0),
            (1048576,     28.0, 166.0), (2097152,     39.4, 130.0),
            (4194304,     42.0, 121.0), (8388608,     45.4, 117.0),
            (16777216,    47.0, 113.0), (33554432,    47.4, 115.0),
            (67108864,    48.0, 164.0), (134217728,   48.0, 170.0),
            (268435456,   47.4, 197.0), (1073741824,  47.6, 196.0),
            (4294967296,  47.6, 198.0), (17179869184, 48.0, 197.0),
        ]
        return TransferCurve(sizes_bytes=[p[0] for p in pts],
                             bw_GBps=[p[1] for p in pts],
                             pj_per_bit=[p[2] for p in pts], kind="nccl-gl1808")


def measure_pcie_link(src: int = 0, dst: int = 1,
                      sizes_bytes: List[int] = None, iters: int = 200,
                      kind: str = "pcie") -> LinkModel:
    """Measure a real GPU->GPU link by timing single-process peer copies, then
    least-squares fit (latency_us, bw_GBps). Needs >=2 visible GPUs. Uses
    `dst.copy_(src)` which lowers to cudaMemcpyPeerAsync over PCIe (peer access)
    or host-staged copy if peer access is unavailable -- either way it is the
    real path a stage output would take to the next card."""
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"measure_pcie_link needs >=2 GPUs, found {torch.cuda.device_count()}. "
            f"Build a LinkModel from a torchrun p2p_energy_benchmark run or "
            f"LinkModel(bw_GBps=..., latency_us=..., kind='manual') instead.")
    if sizes_bytes is None:
        sizes_bytes = [1 << 12, 1 << 16, 1 << 20, 1 << 24, 1 << 26, 1 << 28]  # 4KB..256MB

    sdev, ddev = f"cuda:{src}", f"cuda:{dst}"
    stream = torch.cuda.Stream(device=dst)
    pts = []  # (bytes, ms)
    print(f"[link] measuring {sdev}->{ddev} over {len(sizes_bytes)} sizes...")
    for nbytes in sizes_bytes:
        numel = max(1, nbytes // DTYPE_BYTES)
        s = torch.empty(numel, dtype=torch.float16, device=sdev).normal_()
        d = torch.empty(numel, dtype=torch.float16, device=ddev)
        # auto-scale iters so tiny transfers still accumulate measurable time
        it = max(iters, min(5000, (1 << 24) // max(1, nbytes) * iters))
        with torch.cuda.stream(stream):
            for _ in range(10):                      # warmup
                d.copy_(s, non_blocking=True)
        torch.cuda.synchronize(dst)
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(it):
                d.copy_(s, non_blocking=True)
        torch.cuda.synchronize(dst)
        ms = (time.perf_counter() - t0) / it * 1e3
        pts.append((numel * DTYPE_BYTES, ms))
        print(f"    {numel*DTYPE_BYTES/1e6:8.3f} MB  ->  {ms:9.4f} ms  "
              f"({numel*DTYPE_BYTES/(ms*1e6):7.2f} GB/s)")

    model = LinkModel.fit(pts, kind=kind)
    print(f"[link] fit: {model.bw_GBps:.1f} GB/s, {model.latency_us:.1f} us fixed "
          f"(raw copy-engine path)")
    return model


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def measure_pcie_transfer_power_pj(src: int = 0, dst: int = 1, chunk_bytes: int = 128 * 1024**2,
                                   ramp_s: float = 5.0, sample_s: float = 10.0,
                                   sample_dt: float = 0.05, repeats: int = 3) -> dict:
    """Measure inter-card transfer energy (pJ/bit) by STEADY-STATE POWER DIFFERENCING --
    the robust replacement for the energy-counter subtraction (which differenced a tiny
    transfer delta against a huge, thermally-drifting idle baseline and went negative).

    Per repeat: saturate the peer-copy link for (ramp_s + sample_s) and average the
    INSTANTANEOUS board power (nvmlDeviceGetPowerUsage, both GPUs) over the steady window
    (after the ramp); then measure idle the same way. transfer_power = P_active - P_idle;
    pj/bit = transfer_power / (bytes/s * 8). active/idle are interleaved and repeated so
    their temperatures stay comparable and outliers are rejected by the median.

    Single-process cudaMemcpyPeer (copy engine), so it captures the link + GDDR-movement
    power of moving the bytes, WITHOUT NCCL's SM-kernel overhead -- the cleaner number for
    a transfer/link model (a dedicated DMA/chiplet link wouldn't burn SM power).

    Returns {pj_per_bit, transfer_W, p_active_W, p_idle_W, bw_GBps, per_repeat}."""
    import threading
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"needs >=2 GPUs, found {torch.cuda.device_count()}")
    nvmlInit()
    h_src = nvmlDeviceGetHandleByIndex(src)
    h_dst = nvmlDeviceGetHandleByIndex(dst)
    def board_W():   # instantaneous power of both boards (W)
        return (nvmlDeviceGetPowerUsage(h_src) + nvmlDeviceGetPowerUsage(h_dst)) / 1000.0
    def sm_mhz():    # SM clocks of both boards (MHz) -- should match idle vs active for
        return (nvmlDeviceGetClockInfo(h_src, NVML_CLOCK_SM),   # a pure copy (no SM work),
                nvmlDeviceGetClockInfo(h_dst, NVML_CLOCK_SM))   # confirming the delta is real

    numel = max(1, chunk_bytes // DTYPE_BYTES)
    s = torch.empty(numel, dtype=torch.float16, device=f"cuda:{src}").normal_()
    d = torch.empty(numel, dtype=torch.float16, device=f"cuda:{dst}")
    stream = torch.cuda.Stream(device=dst)
    with torch.cuda.stream(stream):
        for _ in range(10):
            d.copy_(s, non_blocking=True)
    torch.cuda.synchronize()

    moved = {"bytes": 0}
    def feeder(stop):            # background thread keeps the link saturated
        torch.cuda.set_device(dst)
        with torch.cuda.stream(stream):
            while not stop.is_set():
                for _ in range(8):
                    d.copy_(s, non_blocking=True)
                stream.synchronize()      # bound queue depth; GPU stays busy
                moved["bytes"] += 8 * chunk_bytes

    def phase(active):
        stop, th = threading.Event(), None
        if active:
            moved["bytes"] = 0
            th = threading.Thread(target=feeder, args=(stop,), daemon=True)
            th.start()
        t0 = time.perf_counter(); b0 = moved["bytes"]; samples = []; clks = []
        while time.perf_counter() - t0 < ramp_s + sample_s:
            time.sleep(sample_dt)
            if time.perf_counter() - t0 >= ramp_s:   # skip the ramp-up to steady state
                samples.append(board_W())
                clks.append(max(sm_mhz()))           # higher of the two boards' SM clock
        dur = time.perf_counter() - t0
        if active:
            stop.set(); th.join()
        bw = (moved["bytes"] - b0) / dur if active else 0.0
        return _median(samples), bw, _median(clks)

    rows = []
    for _ in range(repeats):
        pa, bw, ca = phase(True)
        pi, _, ci = phase(False)
        rows.append((pa, pi, bw, ca, ci))
        print(f"[power] active {pa:.1f} W (SM {ca:.0f} MHz) | idle {pi:.1f} W (SM {ci:.0f} MHz) "
              f"| delta {pa-pi:.1f} W | {bw/1e9:.1f} GB/s")
    p_active = _median([r[0] for r in rows])
    p_idle = _median([r[1] for r in rows])
    bw_Bps = _median([r[2] for r in rows])
    sm_active = _median([r[3] for r in rows]); sm_idle = _median([r[4] for r in rows])
    transfer_W = p_active - p_idle
    pj_per_bit = transfer_W / (bw_Bps * 8) * 1e12 if bw_Bps > 0 else float("nan")
    sm_note = "SM idle-clocked (pure copy)" if abs(sm_active - sm_idle) < 0.15 * max(sm_idle, 1) \
              else f"WARNING: SM clock differs active {sm_active:.0f} vs idle {sm_idle:.0f} MHz"
    print(f"[power] STEADY-STATE: active {p_active:.1f} W, idle {p_idle:.1f} W, "
          f"transfer {transfer_W:.1f} W @ {bw_Bps/1e9:.1f} GB/s -> {pj_per_bit:.1f} pJ/bit "
          f"(copy-engine, both boards) | {sm_note}")
    return {"pj_per_bit": pj_per_bit, "transfer_W": transfer_W, "p_active_W": p_active,
            "p_idle_W": p_idle, "bw_GBps": bw_Bps / 1e9, "sm_active_mhz": sm_active,
            "sm_idle_mhz": sm_idle, "per_repeat": rows}


def link_model_from_env(var: str = "PCIE_LINK_JSON") -> Optional["LinkModel"]:
    """Load a LinkModel from the JSON path in env var `var` (written by
    characterize_pcie.py / LinkModel.save). Returns None if unset/missing, so
    benchmarks default to energy-only (legacy) transfer behaviour."""
    p = os.environ.get(var)
    if p and os.path.exists(p):
        m = LinkModel.load(p)
        print(f"[link] loaded {m.kind} model from {p} "
              f"({m.bw_GBps:.1f} GB/s, {m.latency_us:.1f} us)")
        return m
    if p:
        print(f"[link] WARNING: {var}={p} not found; transfer latency unmodelled")
    return None


def transfer_curve_from_env(var: str = "PCIE_CURVE_JSON",
                            default_measured: bool = False) -> Optional["TransferCurve"]:
    """Load a size-dependent TransferCurve from the JSON path in env var `var` (written by
    characterize_transfer_curve.py / TransferCurve.save). If `var` is unset and
    `default_measured` is True, fall back to the baked-in measured gl1808 NCCL curve;
    otherwise return None (so the model uses the scalar pj/bit / link_model path)."""
    p = os.environ.get(var)
    if p and os.path.exists(p):
        c = TransferCurve.load(p)
        print(f"[curve] loaded {c.kind} transfer curve from {p} "
              f"({len(c.sizes_bytes)} pts, pj/bit {min(c.pj_per_bit):.0f}-{max(c.pj_per_bit):.0f})")
        return c
    if p:
        print(f"[curve] WARNING: {var}={p} not found")
    if default_measured:
        c = TransferCurve.measured_nccl()
        print(f"[curve] using baked-in {c.kind} curve "
              f"(pj/bit {min(c.pj_per_bit):.0f}-{max(c.pj_per_bit):.0f})")
        return c
    return None


def set_device(device_index: int):
    """
    Initialize NVML and set the global handle for energy consumption measurement.
    
    Args:
        device_index (int): Index of the GPU device.
    """
    global handle
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_index)
    torch.cuda.set_device(device_index)
    print(f"Using CUDA device: {torch.cuda.get_device_name(device_index)}")

def profile_device_idle_power():
    global handle
    if handle is None:
        raise RuntimeError("NVML handle is not initialized. Call set_device() first.")
    
    print(" - Profiling GPU idle power consumption...")
    # Make sure the GPU is idle
    torch.cuda.synchronize()

    print(" - Waiting for GPU to be idle...")
    time.sleep(5)

    print(" - Measuring idle power consumption...")
    e0_mJ = nvmlDeviceGetTotalEnergyConsumption(handle)
    time.sleep(5)
    e1_mJ = nvmlDeviceGetTotalEnergyConsumption(handle)
    
    total_energy_J = (e1_mJ - e0_mJ) / 1000.0

    global GPU_IDLE_POWER
    GPU_IDLE_POWER = total_energy_J / 5.0 
    print(f"Measured avg idle power: {GPU_IDLE_POWER} W over 5 seconds")

UNROLL_FACTOR = 100

# --- Time-targeted measurement + thermal pre-heat ---------------------------
# Each operator is timed for a FIXED WALL-CLOCK WINDOW (not a fixed iteration count),
# so the NVML energy counter (mJ, periodically refreshed) integrates over a long-enough
# span regardless of how fast the kernel is. Measured on RTX PRO 6000: energy/iter
# converges by ~1.5 s (<1% run-to-run on even the smallest near-idle op); going longer
# buys nothing and can add idle-power drift. The number of graph replays is computed
# from a probe of per-replay time, so fast ops run many iters and slow ops few.
TARGET_SECONDS = float(os.environ.get("BENCH_TARGET_S", "1.5"))

# Pre-heat: heavy, power-capped GEMMs droop ~6% in clock as the die heats (2287->2115
# MHz observed); a cold/boost-clock measurement over-reports throughput. Before timing
# we replay the captured graph until the SM clock settles, so timing happens at thermal
# steady state. Light ops never droop, so this exits after the minimum and costs ~nothing.
PREHEAT_MAX_S   = float(os.environ.get("BENCH_PREHEAT_MAX_S", "8.0"))  # hard cap per op
PREHEAT_CHUNK_S = 0.2     # replay this long between SM-clock polls
PREHEAT_CLK_TOL = 0.01    # clock "settled" when <1% change between consecutive polls

LAST_PREHEAT_S = 0.0   # wall time the most recent op spent pre-heating (telemetry)

def _sm_clock_mhz() -> Optional[int]:
    try:
        return nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)
    except Exception:
        return None

def _preheat_and_probe(g: "torch.cuda.CUDAGraph") -> float:
    """Replay graph `g` until the SM clock reaches steady state (or PREHEAT_MAX_S),
    so the subsequent timed window is measured hot, not on boost clocks. Doubles as a
    probe: returns the wall-clock time of ONE g.replay(), used to size the timed window
    to TARGET_SECONDS. Light ops (no clock droop) exit after ~two polls.

    NOTE: g.replay() is ASYNC. We sync every chunk and launch a FIXED reps-per-chunk
    (sized from an initial synced probe) so we never enqueue an unbounded backlog --
    otherwise a heavy op (100s of ms/replay) queues thousands of launches at CPU speed
    and the final sync blocks for minutes."""
    # Initial synced probe: time one replay to size the per-chunk batch.
    torch.cuda.synchronize()
    c0 = time.perf_counter(); g.replay(); torch.cuda.synchronize()
    t_per_replay = max(time.perf_counter() - c0, 1e-9)
    reps = max(1, math.ceil(PREHEAT_CHUNK_S / t_per_replay))   # ~PREHEAT_CHUNK_S of work per poll

    t0 = time.perf_counter()
    last_clk = None
    while time.perf_counter() - t0 < PREHEAT_MAX_S:
        c0 = time.perf_counter()
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
        t_per_replay = (time.perf_counter() - c0) / reps        # refine at the hot clock
        clk = _sm_clock_mhz()
        if clk is None:                       # no clock telemetry -> can't adapt; one chunk
            break
        if last_clk is not None and abs(clk - last_clk) <= PREHEAT_CLK_TOL * last_clk:
            break                             # settled
        last_clk = clk
    global LAST_PREHEAT_S
    LAST_PREHEAT_S = time.perf_counter() - t0
    return t_per_replay

# --- Cold-DRAM (defeat L2 reuse) configuration ------------------------------
# When COLD_DRAM is True, test_kernel_iter allocates a RING of independent input
# buffers whose combined footprint exceeds the L2 cache by COLD_DRAM_MARGIN, and
# rotates through them. By the time a buffer is reused its data has been evicted
# from L2, so every kernel re-reads its operands from HBM (a "cold" read).
# This adds NO extra kernels (unlike an explicit L2 flush), so it does not
# contaminate the NVML energy / latency measurement with bogus memory traffic.
COLD_DRAM = os.environ.get("COLD_DRAM", "1") not in ("0", "", "false", "False")
COLD_DRAM_MARGIN = float(os.environ.get("COLD_DRAM_MARGIN", "4.0"))  # target ring footprint = MARGIN * L2 size
COLD_DRAM_MAX_RING = 2048   # cap on distinct buffers (bounds graph size / memory)
_L2_BYTES_FALLBACK = 50 * 1024 * 1024  # H100-class default if query fails

def set_cold_dram(enabled: bool, margin: float = None):
    """Toggle cold-DRAM buffer rotation globally (call before running pipelines)."""
    global COLD_DRAM, COLD_DRAM_MARGIN
    COLD_DRAM = enabled
    if margin is not None:
        COLD_DRAM_MARGIN = margin
    print(f"[cold-DRAM] {'ENABLED' if enabled else 'disabled'} (margin x{COLD_DRAM_MARGIN} of L2)")

def _l2_bytes() -> int:
    try:
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        l2 = getattr(p, 'L2_cache_size', 0) or getattr(p, 'l2_cache_size', 0)
        if l2 and l2 > 0:
            return int(l2)
    except Exception:
        pass
    return _L2_BYTES_FALLBACK

def _state_read_bytes(state) -> int:
    """Bytes of operands a single capture() reads. The output buffer (written, not
    read) is excluded so the ring is sized by actual read footprint. Convention:
    setup() returns read tensors first and the output tensor last."""
    tensors = [t for t in state if torch.is_tensor(t)]
    if len(tensors) >= 2:
        tensors = tensors[:-1]  # drop the trailing output buffer
    return max(sum(t.numel() * t.element_size() for t in tensors), 1)

def test_kernel_iter(name: str, setup_func: Callable, capture_func: Callable, iters: int = 100,
                     cold_dram: Optional[bool] = None, ring_override: Optional[int] = None):
    """Profile one captured kernel sequence over a fixed TARGET_SECONDS window at
    thermal steady state (see _preheat_and_probe). `iters` is legacy and ignored --
    the replay count is derived from a per-replay probe so the window hits the target.

    cold_dram:     None -> use the global COLD_DRAM; True/False -> per-call override.
    ring_override: force an exact buffer-ring size, bypassing the cold auto-sizing.
                   Use a SMALL ring (operand footprint < L2) to measure WARM/on-chip
                   reads while still pipelining across iterations (e.g. the softmax half
                   of a fused attention stage whose scores stay resident). The buffers
                   come from setup_func (real `randn` values), so reads hit L2 instead of
                   HBM. Still >1 so consecutive iterations don't false-serialize."""

    global handle
    cold = COLD_DRAM if cold_dram is None else cold_dram

    # Build the buffer ring. ring==1 reproduces the legacy single-buffer behaviour.
    if ring_override is not None and ring_override > 0:
        ring = ring_override                       # explicit (warm) ring
        states = [setup_func() for _ in range(ring)]
    elif cold:
        probe = setup_func()
        target = int(COLD_DRAM_MARGIN * _l2_bytes())
        ring = max(1, math.ceil(target / _state_read_bytes(probe)))
        ring = min(ring, COLD_DRAM_MAX_RING)
        states = [probe] + [setup_func() for _ in range(ring - 1)]
    else:
        ring = 1
        states = [setup_func()]

    # Unroll at least one full sweep of the ring so the captured graph touches
    # every buffer (otherwise only the first UNROLL_FACTOR buffers stay hot/cold).
    unroll = max(UNROLL_FACTOR, ring)

    # Minimal warmup so the capture is safe (allocations/JIT settled).
    for i in range(max(10, ring)):
        capture_func(states[i % ring])
    torch.cuda.synchronize()

    # CUDA Graph Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(unroll):
            capture_func(states[i % ring])

    # Pre-heat to thermal/clock steady state, and probe per-replay time. `iters` (legacy
    # fixed-count arg) is ignored: the timed window is sized to TARGET_SECONDS instead.
    t_per_replay = _preheat_and_probe(g)
    replays = max(1, math.ceil(TARGET_SECONDS / max(t_per_replay, 1e-9)))  # >= one sweep

    # Timing setup
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    e0_mJ = nvmlDeviceGetTotalEnergyConsumption(handle) # NVML energy before (mJ)
    t0 = time.perf_counter()  # wall clock start

    start_evt.record()
    for _ in range(replays):
        g.replay()
    end_evt.record()
    torch.cuda.synchronize()

    iters = unroll * replays  # actual iterations executed (multiple of unroll)

    t1 = time.perf_counter()  # wall clock end
    e1_mJ = nvmlDeviceGetTotalEnergyConsumption(handle)
    total_energy_J = (e1_mJ - e0_mJ) / 1000.0
    avg_energy_J = total_energy_J / iters
    avg_power_W = total_energy_J / (t1 - t0)

    total_latency_ms = start_evt.elapsed_time(end_evt)  # in milliseconds
    avg_latency_ms = total_latency_ms / iters 

    return {
        'name': name,
        'iters': iters,
        'avg_latency_ms': avg_latency_ms,
        'avg_energy_J': avg_energy_J,
        'avg_power_W': avg_power_W
    }

def test_matmul_iter(name: str, M:int, K:int, N:int, datatype:torch.dtype=torch.float16, iters:int=100):
    def setup():
        input1 = torch.randn(M, K, dtype=datatype, device='cuda')
        input2 = torch.randn(K, N, dtype=datatype, device='cuda')
        output = torch.empty(M, N, dtype=datatype, device='cuda')
        return input1, input2, output

    def capture(state):
        input1, input2, output = state
        torch.matmul(input1, input2, out=output)
        return output

    res = test_kernel_iter(name, setup, capture, iters)
    res['out_elems'] = M * N          # output activation element count (for P2P)
    return res

def test_softmax_iter(name: str,N:int, M:int, datatype:torch.dtype=torch.float16, iters:int=100,
                      ring_override: Optional[int] = None):
    def setup():
        input_tensor = torch.randn(N, M, dtype=datatype, device='cuda')
        output_tensor = torch.empty_like(input_tensor)
        return input_tensor, output_tensor

    def capture(state):
        input_tensor, output_tensor = state
        torch.softmax(input_tensor, dim=-1, out=output_tensor)
        return output_tensor

    res = test_kernel_iter(name, setup, capture, iters, ring_override=ring_override)
    res['out_elems'] = N * M          # output activation element count (for P2P)
    return res


# warm-ring tuning for the fused-attn softmax half (see test_fused_qk_softmax_iter)
FUSED_WARM_MAX = 32          # cap warm-ring buffers (enough for pipelining)
FUSED_WARM_MIN = 4           # need at least this many to pipeline; else fall back to cold
FUSED_WARM_L2_FRAC = 0.5     # keep warm working set under this fraction of L2

def test_fused_qk_softmax_iter(name: str, rows: int, head_dim: int, ctx: int,
                               datatype: torch.dtype = torch.float16, iters: int = 100):
    """Attention 'scores' stage = Q@K^T (attn_qk) then softmax, with the scores matrix
    treated as ON-CHIP when it fits. Measured as TWO separate, well-pipelined captured
    graphs whose latencies/energies are summed -- NOT one fused graph (a single stream
    of alternating matmul+softmax kernels schedules ~2x worse than homogeneous streams,
    an artifact, not physics):

      - attn_qk : standard COLD matmul (Q,K legitimately come from DRAM).
      - softmax : measured with a WARM ring sized so the scores working set stays in L2,
                  so it reads scores on-chip (hot) -- removing the spurious cold-DRAM
                  round trip the old `cold softmax` charged. The ring still has several
                  buffers so iterations pipeline. If the scores are too large to keep a
                  few buffers resident (e.g. long-context PREFILL), it FALLS BACK to a
                  cold softmax -- which is physically correct: scores that don't fit
                  on-chip really do spill to HBM.

    On measured hardware (RTX PRO 6000, 2 MB decode scores): warm softmax ~7 us vs cold
    ~10.6 us -> ~25% off the stage. Returns one result (summed) named `name`, with
    out_elems = rows*ctx (softmax output handed to attn_v)."""
    bytes_per_score = rows * ctx * DTYPE_BYTES     # fp16 scores [rows, ctx]
    # softmax setup allocates input+output -> 2x the scores footprint per ring buffer
    fit = int(FUSED_WARM_L2_FRAC * _l2_bytes() / max(1, 2 * bytes_per_score))
    warm = min(FUSED_WARM_MAX, fit)
    use_warm = warm >= FUSED_WARM_MIN          # else scores spill -> cold softmax

    qk = test_matmul_iter(f"{name}__qk", M=rows, K=head_dim, N=ctx, datatype=datatype, iters=iters)
    sm = test_softmax_iter(f"{name}__softmax", N=rows, M=ctx, datatype=datatype, iters=iters,
                           ring_override=(warm if use_warm else None))

    res = {
        'name': name,
        'iters': min(qk['iters'], sm['iters']),
        'avg_latency_ms': qk['avg_latency_ms'] + sm['avg_latency_ms'],   # serial on one chiplet
        'avg_energy_J':   qk['avg_energy_J'] + sm['avg_energy_J'],
        'avg_power_W':    (qk['avg_energy_J'] + sm['avg_energy_J']) /
                          max(1e-12, qk['avg_latency_ms'] + sm['avg_latency_ms']) * 1e3,
        'out_elems': rows * ctx,          # softmax output handed downstream (for P2P)
        'scores_on_chip': use_warm,       # False -> scores spilled to HBM (cold)
    }
    return res


def test_conv_iter(name: str, N, C, P, Q, M, G, R, S, HS:int, WS:int, datatype:torch.dtype=torch.float16, iters:int=100):
    """
    Test a 2D conv using Pytorch FF conv2d.
    - N: batch size
    - C: input channels
    - P: output height
    - Q: output width
    - M: output channels
    - G: groups
    - R: kernel height
    - S: kernel width
    - HS: height stride
    - WS: width stride
    """
    def setup():
        H_in = (P-1) * HS + R # Input height
        W_in = (Q-1) * WS + S # Input width

        C_new = C * G  # Adjust input channels for groups
        M_new = M * G  # Adjust output channels for groups

        input_tensor = torch.randn(N, C_new, H_in, W_in, dtype=datatype, device='cuda')
        weight_tensor = torch.randn(M_new, C_new//G, R, S, dtype=datatype, device='cuda')
        return input_tensor, weight_tensor

    def capture(state):
        input_tensor, weight_tensor = state
        torch.nn.functional.conv2d(input_tensor, weight_tensor, stride=(HS, WS), groups=G)

    res = test_kernel_iter(name, setup, capture, iters)
    res['out_elems'] = N * (M * G) * P * Q   # batch × out_channels × P × Q (for P2P)
    return res


def run_phase(res_list: List[dict], func: Callable, *args, **kwargs):
    res_list.append(func(*args, **kwargs))
    print(GREEN_DOT, end='', flush=True)

def combine_dag(results: List[dict], levels: List = None, edges: List[Tuple] = None,
                pj_per_bit: float = DEFAULT_PJ_PER_BIT,
                link_model: Optional[LinkModel] = None,
                idle_power_W: Optional[float] = None,
                transfer_curve: Optional["TransferCurve"] = None) -> dict:
    """Pure (no-CUDA) combination of per-op profiling results into pipeline metrics.
    Split out of run_pipeline so the latency/energy bookkeeping is unit-testable on
    CPU with synthetic `results`. Each result needs 'name', 'avg_latency_ms',
    'avg_energy_J', and (for P2P/transfer) 'out_elems'.

    `levels` is the operator-DAG stage schedule (topological order); each stage maps
    to one chiplet (group). A stage is either:
      - a list of op names              -> 'parallel': ops run concurrently on
                                           separate chiplets; stage latency = slowest.
      - {'ops':[...], 'mode':'serial'}  -> fused: ops run back-to-back on ONE chiplet;
                                           stage latency = sum; intermediate stays
                                           on-chip (no DRAM, no cross-chiplet transfer).

    THREE hardware resources are modelled separately (see the four-step pipeline
    note: ① read DRAM + ② compute + ③ write DRAM are bundled inside each measured
    op latency -- the GPU overlaps them within the kernel -- while ④ inter-card
    transfer uses the copy engine and is modelled here via `link_model`):
      - compute critical path   = Σ stage latency (single sample, transfer excluded)
      - compute bottleneck       = max stage latency (throughput, transfer excluded)
      - active energy            = Σ per-op compute energy (additive)
      - idle energy              = (N_chiplets·T_bn − Σ op_latency)·P_idle
      - P2P energy               = Σ over cross-chiplet edges of bytes·8·pj_per_bit
      - transfer latency (NEW)   = per cross-chiplet edge, link_model.latency_ms(bytes).
            * critical path WITH transfer = compute critical path + Σ per-stage
              incoming transfer (transfer is on the single-sample dependency chain --
              it is NEVER hidden for one sample).
            * bottleneck WITH transfer = max(compute bottleneck, slowest single edge)
              -- a link slower than the slowest compute stage becomes the throughput
              bottleneck, i.e. step ④ is NOT hidden. `transfer_bound` flags this.
      Edges internal to a fused stage (same chiplet) cost nothing -- the intermediate
      never leaves the chip. If link_model is None, transfer latencies are 0 and the
      WITH-transfer metrics equal the compute-only ones (and `link_modelled` is False).

    `transfer_curve` (preferred over link_model/scalar pj_per_bit when given) makes BOTH
    the per-edge bandwidth AND pj/bit SIZE-DEPENDENT (see TransferCurve): each edge's
    energy uses curve.pj_at(bytes) and its latency uses curve.latency_ms(bytes), so a
    small activation hop is charged its true (high) pj/bit and a large one its plateau --
    instead of one scalar that is wrong by up to ~8x across the size range.
    """
    if idle_power_W is None:
        idle_power_W = GPU_IDLE_POWER
    # Prefer a MEASURED pj/bit from the link model over the default assumption (a
    # size-dependent transfer_curve, if given, overrides this per edge below).
    if link_model is not None and link_model.pj_per_bit is not None:
        pj_per_bit = link_model.pj_per_bit
    by_name = {r['name']: r for r in results}
    total_pipeline_energy = sum(r['avg_energy_J'] for r in results)   # active, additive
    sum_op_latency = sum(r['avg_latency_ms'] for r in results)        # every op once

    if not levels:
        levels = [[r['name']] for r in results]  # linear fallback

    # Walk the DAG stages. Each op is its own chiplet, EXCEPT a serial/fused stage
    # whose ops share one chiplet. `chiplet_of` tells on-chip from cross-chip edges;
    # `stage_of` maps each op to the index of its stage (for per-stage transfer).
    total_compute_latency = 0.0   # critical path, compute only
    n_chiplets = 0
    stage_durs = []
    chiplet_of = {}
    stage_of = {}
    for stage in levels:
        if isinstance(stage, dict):
            op_names, mode = stage['ops'], stage.get('mode', 'parallel')
        else:
            op_names, mode = stage, 'parallel'
        ops = [by_name[n] for n in op_names if n in by_name]
        if not ops:
            continue
        sidx = len(stage_durs)
        if mode == 'serial':
            stage_dur = sum(o['avg_latency_ms'] for o in ops)  # back-to-back on one chiplet
            for o in ops:
                chiplet_of[o['name']] = n_chiplets
                stage_of[o['name']] = sidx
            n_chiplets += 1
        else:
            stage_dur = max(o['avg_latency_ms'] for o in ops)  # parallel: one chiplet per op
            for o in ops:
                chiplet_of[o['name']] = n_chiplets
                stage_of[o['name']] = sidx
                n_chiplets += 1
        total_compute_latency += stage_dur
        stage_durs.append(stage_dur)

    # Throughput bottleneck = slowest compute stage. In steady state every chiplet is
    # powered for one bottleneck period per token; active leakage is already in the
    # measured per-op energy, so add only the idle portion (no double count).
    bottleneck_compute = max(stage_durs) if stage_durs else 0.0
    idle_energy = idle_power_W * (n_chiplets * bottleneck_compute - sum_op_latency) / 1000.0
    total_pipeline_energy_with_idle = total_pipeline_energy + idle_energy

    # Cross-chiplet edges -> P2P energy (always) and ④ transfer latency. A transfer_curve
    # makes bw/pj size-dependent per edge; else a scalar pj/bit + (optional) link_model.
    transfer_modelled = transfer_curve is not None or link_model is not None
    p2p_energy = 0.0
    total_xfer_bits = 0
    transfers = []                  # (src, dst, bytes, ms, pj_per_bit) for each cross-chiplet edge
    xfer_in_per_stage = {}          # dst stage idx -> max incoming transfer (ms)
    max_edge_xfer_ms = 0.0
    for edge in (edges or []):
        src, dst = edge[0], edge[1]
        if src not in by_name or dst not in by_name:
            continue
        if chiplet_of.get(src) == chiplet_of.get(dst):
            continue  # same (fused) chiplet — intermediate never leaves the chip
        n_elems = edge[2] if len(edge) > 2 else by_name[src]['out_elems']
        nbytes = n_elems * DTYPE_BYTES
        if transfer_curve is not None:
            edge_pj = transfer_curve.pj_at(nbytes)        # size-dependent energy
            t_ms = transfer_curve.latency_ms(nbytes)      # size-dependent bandwidth
        else:
            edge_pj = pj_per_bit                          # one scalar for every edge
            t_ms = link_model.latency_ms(nbytes) if link_model is not None else 0.0
        p2p_energy += nbytes * 8 * edge_pj * 1e-12
        total_xfer_bits += nbytes * 8
        transfers.append((src, dst, nbytes, t_ms, edge_pj))
        ds = stage_of.get(dst)
        if ds is not None:
            xfer_in_per_stage[ds] = max(xfer_in_per_stage.get(ds, 0.0), t_ms)
        max_edge_xfer_ms = max(max_edge_xfer_ms, t_ms)

    # Effective (bytes-weighted) pj/bit actually charged -- a single scalar with a curve
    # is meaningless per-edge, but this summarizes it for the report.
    eff_pj_per_bit = (p2p_energy * 1e12 / total_xfer_bits) if total_xfer_bits else pj_per_bit
    sum_xfer_ms = sum(xfer_in_per_stage.values())
    total_pipeline_energy_with_idle_and_p2p = total_pipeline_energy_with_idle + p2p_energy

    # WITH-transfer latencies: critical path adds per-stage transfer (on the single-
    # sample dependency chain); bottleneck takes max(compute, slowest link).
    critical_path_with_xfer = total_compute_latency + sum_xfer_ms
    bottleneck_with_xfer = max(bottleneck_compute, max_edge_xfer_ms)
    transfer_bound = max_edge_xfer_ms > bottleneck_compute

    return {
        'total_pipeline_latency': total_compute_latency,      # compute-only critical path (legacy name)
        'bottleneck_latency': bottleneck_compute,             # compute-only bottleneck (legacy name)
        'total_pipeline_energy': total_pipeline_energy,
        'total_pipeline_energy_with_idle': total_pipeline_energy_with_idle,
        'p2p_transfer_energy': p2p_energy,
        'total_pipeline_energy_with_idle_and_p2p': total_pipeline_energy_with_idle_and_p2p,
        'n_chiplets': n_chiplets,
        # --- step ④ transfer (NEW) ---
        'link_modelled': transfer_modelled,
        'curve_modelled': transfer_curve is not None,
        'p2p_effective_pj_per_bit': eff_pj_per_bit,
        'max_transfer_latency_ms': max_edge_xfer_ms,
        'sum_transfer_latency_ms': sum_xfer_ms,
        'total_pipeline_latency_with_xfer': critical_path_with_xfer,
        'bottleneck_latency_with_xfer': bottleneck_with_xfer,
        'transfer_bound': transfer_bound,
        'transfers': transfers,
    }


def write_pipeline_report(output_file: str, pipe_name: str, results: List[dict],
                          m: dict, pj_per_bit: float, link_model: Optional[LinkModel],
                          transfer_curve: Optional["TransferCurve"] = None):
    """Write the human-readable .txt report. Keeps every legacy line verbatim (the
    visualization/ regex parsers depend on them) and appends the new ④-transfer
    section only when a link model or transfer curve was supplied."""
    # With a size-dependent curve there is no single pj/bit -- report the effective
    # (bytes-weighted) value actually charged so the legacy line stays meaningful.
    report_pj = m.get('p2p_effective_pj_per_bit', pj_per_bit) if m.get('curve_modelled') else pj_per_bit
    with open(output_file, 'w') as f:
        f.write(f"Results for {pipe_name}:\n")
        for result in results:
            f.write(f"{result}\n")
        # --- legacy lines (DO NOT rename: parsed by visualization/parse_bench.py) ---
        f.write(f"Critical Path Latency (ms): {m['total_pipeline_latency']}\n")
        f.write(f"Total Pipeline Energy (J): {m['total_pipeline_energy']}\n")
        f.write(f"Bottleneck Latency (ms): {m['bottleneck_latency']}\n")
        f.write(f"Total Pipeline Energy with Idle (J): {m['total_pipeline_energy_with_idle']}\n")
        pj_tag = f"{report_pj:.0f} pJ/bit (size-dependent, eff.)" if m.get('curve_modelled') \
                 else f"{report_pj:.0f} pJ/bit"
        f.write(f"P2P Transfer Energy (J): {m['p2p_transfer_energy']}  [@ {pj_tag}]\n")
        f.write(f"Total Pipeline Energy with Idle + P2P (J): {m['total_pipeline_energy_with_idle_and_p2p']}\n")
        # --- NEW: step ④ inter-card transfer latency ---
        if m['link_modelled']:
            if transfer_curve is not None:
                f.write(f"--- Inter-card transfer (step 4), "
                        f"curve={transfer_curve.kind} size-dependent "
                        f"(pj/bit {min(transfer_curve.pj_per_bit):.0f}-{max(transfer_curve.pj_per_bit):.0f}, "
                        f"bw {min(transfer_curve.bw_GBps):.0f}-{max(transfer_curve.bw_GBps):.0f} GB/s) ---\n")
            else:
                f.write(f"--- Inter-card transfer (step 4), "
                        f"link={link_model.kind} {link_model.bw_GBps:.1f} GB/s "
                        f"{link_model.latency_us:.1f} us ---\n")
            f.write(f"Max Single-Edge Transfer Latency (ms): {m['max_transfer_latency_ms']}\n")
            f.write(f"Sum On-Path Transfer Latency (ms): {m['sum_transfer_latency_ms']}\n")
            f.write(f"Critical Path Latency w/ Transfer (ms): {m['total_pipeline_latency_with_xfer']}\n")
            f.write(f"Bottleneck Latency w/ Transfer (ms): {m['bottleneck_latency_with_xfer']}\n")
            f.write(f"Transfer-bound (slowest link > slowest compute stage): {m['transfer_bound']}\n")
        else:
            f.write("--- Inter-card transfer (step 4): NOT modelled "
                    "(no link_model/curve; transfer latency assumed 0) ---\n")


def run_pipeline(pipe_name, phase_list:List[Tuple[str, Callable]], output_dir,
                 levels:List[List[str]]=None, edges:List[Tuple]=None,
                 pj_per_bit:float=DEFAULT_PJ_PER_BIT,
                 link_model:Optional[LinkModel]=None,
                 transfer_curve:Optional["TransferCurve"]=None) -> dict:
    """Profile each operator (① read DRAM + ② compute + ③ write DRAM bundled per
    kernel), then combine along the operator DAG via combine_dag(). `link_model`
    adds the ④ inter-card transfer latency to the bottleneck and critical path; when
    None, transfer is energy-only (legacy behaviour). `transfer_curve` (preferred when
    given) makes both bw and pj/bit size-dependent per edge. See combine_dag for the model.
    """
    results = []

    start_time = time.perf_counter()

    print(f"[ Running Pipeline {pipe_name} ]", end=' ')
    for _, lambda_func in phase_list:
        run_phase(results, lambda_func)

    end_time = time.perf_counter()
    print(" " * (60 - len(pipe_name)) + "  [ {:>6.2f}s ]".format(end_time - start_time))

    global GPU_IDLE_POWER
    m = combine_dag(results, levels=levels, edges=edges, pj_per_bit=pj_per_bit,
                    link_model=link_model, idle_power_W=GPU_IDLE_POWER,
                    transfer_curve=transfer_curve)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{pipe_name}.txt")
    write_pipeline_report(output_file, pipe_name, results, m, pj_per_bit, link_model,
                          transfer_curve=transfer_curve)

    return {
        'name': pipe_name,
        'results': results,
        'layer_results': results,
        **m,
    }