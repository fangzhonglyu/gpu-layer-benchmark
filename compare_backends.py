"""Per-operator backend comparison: PyTorch (cuBLAS/cuDNN eager) vs Triton (Inductor).

Every backend is timed with the SAME method: warmup, then CUDA-graph capture +
replay (falls back to a plain replay loop if capture fails -- the method used is
printed so the comparison stays transparent). This mirrors how kernels.py times the
real benchmark, so numbers are comparable.

For GEMM/conv, two Triton columns are produced:
  - triton  : Inductor forced to use ONLY its Triton template (max_autotune_*_backends="TRITON")
  - best    : Inductor free to pick max(Triton, ATen/cuBLAS) -- never worse than cuBLAS.
For softmax: eager (a CUDA reduction kernel) vs Inductor-generated Triton softmax.

CUTLASS is handled separately (GEMM only, needs a source-built profiler on sm_120);
this script merges benchmarks/cutlass_gemm.json if present.

    module load cuda/12.8.1 && python compare_backends.py
"""
import json
import os

import torch
import torch._dynamo
import torch._inductor.config as ind

DTYPE = torch.float16
torch._dynamo.config.cache_size_limit = 512
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Unified timer (same as kernels.test_kernel_iter: graph capture + replay).
# --------------------------------------------------------------------------- #
def graph_time_ms(run, state, iters=300, unroll=50, warmup=25):
    for _ in range(warmup):
        run(state)
    torch.cuda.synchronize()

    method, g = "graph", None
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(unroll):
                run(state)
    except Exception as e:                      # compiled fn not capturable -> loop
        method, g = "loop(" + type(e).__name__ + ")", None

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    replays = max(1, iters // unroll)
    torch.cuda.synchronize()
    start.record()
    if g is not None:
        for _ in range(replays):
            g.replay()
    else:
        for _ in range(replays):
            for _ in range(unroll):
                run(state)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (replays * unroll), method


def compiled(fn, gemm_backend=None, conv_backend=None):
    """Fresh Inductor compile of fn under the given autotune backend restriction."""
    torch._dynamo.reset()
    if gemm_backend is not None:
        ind.max_autotune_gemm_backends = gemm_backend
    if conv_backend is not None:
        ind.max_autotune_conv_backends = conv_backend
    return torch.compile(fn, mode="max-autotune-no-cudagraphs", dynamic=False)


def fmt(ms, method):
    tag = "" if method == "graph" else f" [{method}]"
    return f"{ms:9.4f}ms{tag}"


# --------------------------------------------------------------------------- #
# GEMM
# --------------------------------------------------------------------------- #
GEMM_SHAPES = [
    (4096, 4096, 4096),    # big square
    (4096, 4096, 11008),   # llama FFN
    (2048, 12288, 4096),   # fused qkv-ish
    (32, 4096, 4096),      # decode / skinny  (cuBLAS heuristics weak here)
    (128, 8192, 8192),     # short-batch prefill
]


def bench_gemm():
    print("\n=== GEMM (fp16)  metric: TFLOP/s ===")
    print(f"{'M':>5} {'K':>6} {'N':>6} | {'pytorch':>11} {'TF':>6} | "
          f"{'triton':>11} {'TF':>6} | {'best':>11} {'TF':>6} | winner")
    print("-" * 96)
    rows = []
    for (M, K, N) in GEMM_SHAPES:
        flop = 2.0 * M * K * N

        def st():
            A = torch.randn(M, K, dtype=DTYPE, device="cuda")
            B = torch.randn(K, N, dtype=DTYPE, device="cuda")
            return (A, B)

        s = st()
        eager = lambda x: torch.matmul(x[0], x[1])
        p_ms, p_m = graph_time_ms(eager, s)

        tri = compiled(lambda a, b: a @ b, gemm_backend="TRITON")
        t_ms, t_m = graph_time_ms(lambda x: tri(x[0], x[1]), st())

        bst = compiled(lambda a, b: a @ b, gemm_backend="ATEN,TRITON")
        b_ms, b_m = graph_time_ms(lambda x: bst(x[0], x[1]), st())

        ptf, ttf, btf = (flop / (m * 1e-3) / 1e12 for m in (p_ms, t_ms, b_ms))
        win = "triton" if t_ms < p_ms * 0.98 else ("tie" if t_ms < p_ms * 1.02 else "pytorch")
        print(f"{M:>5} {K:>6} {N:>6} | {fmt(p_ms,p_m)} {ptf:6.1f} | "
              f"{fmt(t_ms,t_m)} {ttf:6.1f} | {fmt(b_ms,b_m)} {btf:6.1f} | {win}")
        rows.append(dict(op="gemm", M=M, K=K, N=N, pytorch_ms=p_ms, triton_ms=t_ms,
                         best_ms=b_ms, pytorch_tflops=ptf, triton_tflops=ttf,
                         best_tflops=btf, winner=win))
    return rows


# --------------------------------------------------------------------------- #
# Softmax  (memory bound; metric GB/s over read+write)
# --------------------------------------------------------------------------- #
SOFTMAX_SHAPES = [(4096, 4096), (8192, 8192), (2048, 2048)]


def bench_softmax():
    print("\n=== Softmax (fp16, dim=-1)  metric: GB/s ===")
    print(f"{'rows':>6} {'cols':>6} | {'pytorch':>11} {'GB/s':>6} | "
          f"{'triton':>11} {'GB/s':>6} | winner")
    print("-" * 70)
    rows = []
    for (R, C) in SOFTMAX_SHAPES:
        gb = R * C * 2 * 2 / 1e9      # fp16 read + write

        def st():
            return (torch.randn(R, C, dtype=DTYPE, device="cuda"),)

        p_ms, p_m = graph_time_ms(lambda x: torch.softmax(x[0], dim=-1), st())
        tri = compiled(lambda x: torch.softmax(x, dim=-1))
        t_ms, t_m = graph_time_ms(lambda x: tri(x[0]), st())

        pbw, tbw = (gb / (m * 1e-3) for m in (p_ms, t_ms))
        win = "triton" if t_ms < p_ms * 0.98 else ("tie" if t_ms < p_ms * 1.02 else "pytorch")
        print(f"{R:>6} {C:>6} | {fmt(p_ms,p_m)} {pbw:6.0f} | "
              f"{fmt(t_ms,t_m)} {tbw:6.0f} | {win}")
        rows.append(dict(op="softmax", R=R, C=C, pytorch_ms=p_ms, triton_ms=t_ms,
                         pytorch_gbps=pbw, triton_gbps=tbw, winner=win))
    return rows


# --------------------------------------------------------------------------- #
# Conv2d  (cuDNN vs Triton conv;  metric TFLOP/s)
# --------------------------------------------------------------------------- #
# (N, C, H, W, M, R, S, stride, pad)
CONV_SHAPES = [
    (32, 256, 56, 56, 256, 3, 3, 1, 1),    # ResNet-ish 3x3
    (32, 128, 112, 112, 128, 3, 3, 1, 1),  # earlier stage
    (16, 512, 28, 28, 512, 3, 3, 1, 1),    # deeper stage
]


def bench_conv():
    print("\n=== Conv2d (fp16)  metric: TFLOP/s ===")
    print(f"{'N':>3} {'C':>4} {'HxW':>8} {'M':>4} {'k':>3} | {'pytorch':>11} {'TF':>6} | "
          f"{'triton':>11} {'TF':>6} | {'best':>11} {'TF':>6} | winner")
    print("-" * 100)
    rows = []
    for (N, C, H, W, M, R, S, stride, pad) in CONV_SHAPES:
        P = (H + 2 * pad - R) // stride + 1
        Q = (W + 2 * pad - S) // stride + 1
        flop = 2.0 * N * M * C * R * S * P * Q

        def st():
            x = torch.randn(N, C, H, W, dtype=DTYPE, device="cuda")
            w = torch.randn(M, C, R, S, dtype=DTYPE, device="cuda")
            return (x, w)

        f = lambda x, w: torch.nn.functional.conv2d(x, w, stride=stride, padding=pad)
        p_ms, p_m = graph_time_ms(lambda x: f(x[0], x[1]), st())

        tri = compiled(f, conv_backend="TRITON")
        t_ms, t_m = graph_time_ms(lambda x: tri(x[0], x[1]), st())

        bst = compiled(f, conv_backend="ATEN,TRITON")
        b_ms, b_m = graph_time_ms(lambda x: bst(x[0], x[1]), st())

        ptf, ttf, btf = (flop / (m * 1e-3) / 1e12 for m in (p_ms, t_ms, b_ms))
        win = "triton" if t_ms < p_ms * 0.98 else ("tie" if t_ms < p_ms * 1.02 else "pytorch")
        print(f"{N:>3} {C:>4} {H:>3}x{W:<4} {M:>4} {R:>3} | {fmt(p_ms,p_m)} {ptf:6.1f} | "
              f"{fmt(t_ms,t_m)} {ttf:6.1f} | {fmt(b_ms,b_m)} {btf:6.1f} | {win}")
        rows.append(dict(op="conv", N=N, C=C, H=H, W=W, M=M, R=R, stride=stride, pad=pad,
                         pytorch_ms=p_ms, triton_ms=t_ms, best_ms=b_ms,
                         pytorch_tflops=ptf, triton_tflops=ttf, best_tflops=btf, winner=win))
    return rows


def main():
    assert torch.cuda.is_available(), "run on a GPU node"
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__}")
    try:
        import triton
        print(f"triton {triton.__version__}")
    except Exception as e:
        print(f"triton import failed: {e}")

    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    out["gemm"] = bench_gemm()
    out["softmax"] = bench_softmax()
    out["conv"] = bench_conv()

    path = os.path.join(HERE, "benchmarks", "backend_compare.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
