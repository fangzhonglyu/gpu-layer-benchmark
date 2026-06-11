"""Quick cuBLAS (torch.matmul) vs CUTLASS GEMM comparison.

Run on a GPU node (needs CUDA + `pip install nvidia-cutlass`). Times each shape
with the SAME CUDA-graph + cuda-Event method kernels.py uses, so the numbers are
comparable to the main benchmark. Reports TFLOP/s and the CUTLASS/cuBLAS speedup.

    python try_cutlass.py                  # default representative shapes
    python try_cutlass.py 4096,4096,4096   # add custom M,K,N triples
"""
import sys
import time
import torch

try:
    import cutlass                   # CUTLASS 3.x python interface
    HAVE_CUTLASS = True
except Exception:
    try:
        import cutlass_cppgen as cutlass   # CUTLASS 4.x renamed the same API
        HAVE_CUTLASS = True
    except Exception as e:           # not installed / arch unsupported
        HAVE_CUTLASS = False
        _CUTLASS_ERR = e

DTYPE = torch.float16
ITERS = 200
UNROLL = 50                          # kernels in one captured graph (amortize launch)

# Representative shapes. Edit / pass on CLI to match your model_benchmarks GEMMs.
DEFAULT_SHAPES = [
    (4096, 4096, 4096),   # big square (cuBLAS usually already peak here)
    (8192, 8192, 8192),
    (4096, 4096, 11008),  # llama-ish FFN
    (32, 4096, 4096),     # skinny / decode (cuBLAS heuristics often weak here)
    (128, 8192, 8192),
    (2048, 12288, 4096),
]


def graph_time_ms(run_fn, state, iters=ITERS, unroll=UNROLL):
    """CUDA-graph capture `unroll` calls of run_fn(state), replay, return ms/call."""
    for _ in range(10):                      # warmup (also triggers any JIT compile)
        run_fn(state)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(unroll):
            run_fn(state)

    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    replays = max(1, iters // unroll)
    torch.cuda.synchronize()
    start.record()
    for _ in range(replays):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (replays * unroll)


def make_cublas(M, K, N):
    A = torch.randn(M, K, dtype=DTYPE, device='cuda')
    B = torch.randn(K, N, dtype=DTYPE, device='cuda')
    D = torch.empty(M, N, dtype=DTYPE, device='cuda')
    return (lambda s: torch.matmul(s[0], s[1], out=s[2]), (A, B, D))


def make_cutlass(M, K, N):
    A = torch.randn(M, K, dtype=DTYPE, device='cuda')
    B = torch.randn(K, N, dtype=DTYPE, device='cuda')
    D = torch.empty(M, N, dtype=DTYPE, device='cuda')
    plan = cutlass.op.Gemm(element=DTYPE,
                           layout=cutlass.LayoutType.RowMajor,
                           element_accumulator=torch.float32)
    return (lambda s: plan.run(s[0], s[1], s[2], s[2]), (A, B, D))


def main():
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device. Run this on a GPU node.")
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__}")
    if HAVE_CUTLASS:
        print(f"CUTLASS python: {getattr(cutlass, '__version__', '?')}")
    else:
        print(f"CUTLASS NOT available ({_CUTLASS_ERR}) -> only cuBLAS will run.\n"
              f"  pip install nvidia-cutlass")

    shapes = list(DEFAULT_SHAPES)
    for arg in sys.argv[1:]:
        M, K, N = (int(x) for x in arg.split(','))
        shapes.append((M, K, N))

    print(f"\n{'M':>6} {'K':>6} {'N':>6} | {'cuBLAS ms':>10} {'TFLOP/s':>8} |"
          f" {'CUTLASS ms':>11} {'TFLOP/s':>8} | {'speedup':>8}")
    print("-" * 78)
    for (M, K, N) in shapes:
        flop = 2.0 * M * K * N
        fn, st = make_cublas(M, K, N)
        cub = graph_time_ms(fn, st)
        cub_tf = flop / (cub * 1e-3) / 1e12

        if HAVE_CUTLASS:
            try:
                fn, st = make_cutlass(M, K, N)
                cut = graph_time_ms(fn, st)
                cut_tf = flop / (cut * 1e-3) / 1e12
                spd = f"{cub / cut:6.2f}x"
                cut_s = f"{cut:11.4f} {cut_tf:8.1f}"
            except Exception as e:
                cut_s, spd = f"  FAILED: {str(e)[:30]}", "   -"
        else:
            cut_s, spd = f"{'--':>11} {'--':>8}", "   -"

        print(f"{M:>6} {K:>6} {N:>6} | {cub:10.4f} {cub_tf:8.1f} | {cut_s} | {spd:>8}")

    print("\nspeedup > 1.0  -> CUTLASS faster on that shape (worth integrating).")


if __name__ == "__main__":
    main()
