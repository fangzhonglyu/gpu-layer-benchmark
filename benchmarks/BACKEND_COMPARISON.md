# Backend comparison: PyTorch (cuBLAS/cuDNN) vs Triton vs CUTLASS

GPU: **NVIDIA RTX PRO 6000 Blackwell (sm_120)**, gl1808 · torch 2.10.0+cu128 · fp16
Timing: CUDA-graph capture + replay (same method as `kernels.test_kernel_iter`).
Script: `compare_backends.py` → raw numbers in `benchmarks/backend_compare.json`.

## Headline

On this **brand-new sm_120** card, for **large/regular fp16 GEMM** all three backends are
close, ranked **CUTLASS(tuned) ≳ Triton > cuBLAS** — both compilers beat cuBLAS by ~18–24%.
For **softmax and conv**, Triton beats cuDNN handily. cuBLAS only wins on **small/skinny**
GEMM (memory/latency bound). All at **identical fp16 precision** (verified, see below).

CUTLASS **does** run fp16 dense GEMM on sm_120 — but only via the **CuTe DSL**
(`examples/.../blackwell_geforce/kernel/dense_gemm/dense_gemm.py`), NOT the high-level
`nvidia-cutlass` API or the library/profiler (those are blockscaled FP4/6/8 only). It also
needs **manual tile tuning** — the example's default tile (128,256,64) is ~6× too slow.

## GEMM (fp16, TFLOP/s — higher better)

| M | K | N | cuBLAS | Triton | CUTLASS(best tile) | best tile |
|---|---|---|---|---|---|---|
| 4096 | 4096 | 4096 | 291.8 | 343.6 | **355.4** | 128,128,64 |
| 4096 | 4096 | 11008 | 306.7 | 376.5 | **381.3** | 128,128,64 |
| 2048 | 12288 | 4096 | 295.6 | 356.5 | **361.8** | 128,128,64 |
| 32 | 4096 | 4096 | **100.2** | 84.9 | 52.3 | 64,64,64 |
| 128 | 8192 | 8192 | **223.0** | 182.2 | 205.6 | 64,128,64 |

CUTLASS = CuTe DSL `blackwell_geforce` dense GEMM, fp16 in / fp32 acc, ref-check PASS,
best of the 6 valid tile shapes per shape (CUDA-event timed by the example's `testing.benchmark`).
On big regular shapes CUTLASS edges out Triton; on skinny/short shapes it trails (its MMA
tile wastes rows when M≪tile). The `best` Inductor column (max(Triton,cuBLAS)) was 360/377/358
TFLOP/s on the three big shapes — i.e. CUTLASS, Triton-best, and Inductor-best all converge ~355–380.

## Softmax (fp16, dim=-1, GB/s — higher better)

| rows | cols | pytorch | triton | winner |
|---|---|---|---|---|
| 4096 | 4096 | 1364 | **2516** | triton +84% |
| 8192 | 8192 | 1172 | **1467** | triton +25% |
| 2048 | 2048 | 1934 | **2242** | triton +16% |

## Conv2d (fp16, 3x3, TFLOP/s — higher better)

| shape | pytorch(cuDNN) | triton | best | winner |
|---|---|---|---|---|
| N32 C256 56x56 M256 | 168.0 | **241.2** | 243.2 | triton +44% |
| N32 C128 112x112 M128 | 134.4 | **208.9** | 221.0 | triton +55% |
| N16 C512 28x28 M512 | 181.1 | **193.7** | 195.3 | triton +7% |

`best` = Inductor free to pick max(Triton, ATen/cuBLAS); always ≥ pytorch.
`triton` = Inductor forced to use ONLY its own Triton kernel (`max_autotune_*_backends="TRITON"`).

## Correctness / precision (why the win is real, not a cheat)

Same compile, compared against an fp32 reference:

| op | triton vs fp32-gold | vendor vs fp32-gold | triton-vs-vendor |
|---|---|---|---|
| GEMM 4096³ | rel 0.0004 | rel 0.0004 | **rel 0.0000** |
| softmax 4096² | rel 1e-5 | rel 1e-5 | — |
| conv 32×256×56² | rel 0.0002 | rel 0.0002 | — |

- `torch.backends.cuda.matmul.allow_tf32 = False`, fp16 inputs / fp32 accumulate on both paths.
- Triton and cuBLAS GEMM outputs are **bitwise-comparable** (rel diff 0.0000) → no precision
  trade, no tf32 leak. The speedup is genuine.
- Independent corroboration: Inductor's own autotune log ranks its Triton template above the
  ATen/cuBLAS choice (e.g. conv 0.328 ms Triton vs 0.342 ms ATen).

## CUTLASS on sm_120 — works via CuTe DSL only

Three CUTLASS entry points, only the last gives fp16 dense GEMM on sm_120:
- **High-level `nvidia-cutlass`** (`cutlass.op.Gemm`, 4.2.0.0): `SharedMemPerCC` caps at cc 100;
  sm_120 → `KeyError: 120` / `cuFuncSetAttribute invalid argument`. ✗ no sm_120.
- **Library / `cutlass_profiler`** (source build, `-DCUTLASS_NVCC_ARCHS=120a`): of 74 generated
  sm_120 GEMM kernels, **0 are plain fp16** — all blockscaled FP4/6/8 (the `GenerateSM120_*`
  generator funcs are all `*_block_scaled` / `*_blockwise` / `fp4` / `mixed_8bits`). ✗ no fp16 dense.
- **CuTe DSL** (`nvidia-cutlass-dsl` 4.x, `cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py`):
  ✓ **fp16 dense GEMM, fp32 accumulate, ref-check PASS.** This is the only fp16 path and it is
  fully competitive (table above). Caveats: it's a kernel-authoring DSL (not a one-line tuned
  call); only 6 tile shapes are exposed and the **default tile is catastrophically slow** (53 vs
  355 TFLOP/s) — you must sweep tiles. Numbers above are best-of-6 per shape.

## Why (interpretation)

sm_120 (GeForce/workstation Blackwell) is very new; cuBLAS/cuDNN heuristics and kernel
selection are not yet well-tuned for it, while Inductor **autotunes** an arch-appropriate
Triton kernel per shape. On mature datacenter GPUs (A100/H100) the gap is usually the other
way; here it favors the compiler.

## End-to-end pipeline (Llama-3.1-8B decoder layer) — eager vs compiled backend

`kernels.py` now has a `COMPILE_BACKEND` switch: when set, every op's kernel becomes a
`torch.compile(mode="max-autotune")` callable (Inductor picks max(cuBLAS/cuDNN, Triton) per
shape), still measured through `test_kernel_iter` (CUDA-graph + NVML energy). Driver:
`run_e2e_compare.py` (`COMPILE_BACKEND=1 python run_e2e_compare.py`; then `--diff`).

| pipeline | metric | eager | compiled | gain |
|---|---|---|---|---|
| **decode** b1 kv2048 | crit-path latency (ms) | 0.349 | 0.222 | **1.58×** |
| | bottleneck (ms) | 0.112 | 0.079 | 1.41× |
| | energy w/ idle+p2p (J) | 0.198 | 0.124 | **1.60×** |
| **prefill** b1 s2048 | crit-path latency (ms) | 2.903 | 2.576 | 1.13× |
| | bottleneck (ms) | 0.833 | 0.716 | 1.17× |
| | energy w/ idle+p2p (J) | 3.198 | 2.968 | 1.08× |

- **Decode wins big (1.6× latency & energy)** because it is GEMV-shaped (M=1): cuBLAS picks a
  pathological kernel for 1×4096×4096 (q_proj/o_proj 0.062 ms → 0.026 ms, **2.4×**), Inductor's
  autotuned Triton kernel is far better. (Note: at M=1 Triton wins; at M=32 cuBLAS won — the
  crossover is in the skinny regime, and max-autotune picks correctly on both.)
- **Prefill wins modestly (1.1×)** — large regular GEMMs, matching the ~18–23% per-op GEMM gains
  diluted by ops that don't improve. A few small ops go slightly slower compiled (wrapper
  overhead at µs scale), but they are off the critical path.
- Energy tracks latency (same kernels, higher throughput → fewer joules); NVML-measured, real.

## Caveat for the chiplet benchmark methodology

These wins are **single-kernel** (no cross-op fusion), so they are compatible with the per-op
DAG model in `kernels.py` — i.e. you could swap `torch.matmul`/`conv2d`/`softmax` for an
Inductor-compiled callable and keep the same per-op latency/energy attribution. The skinny/
decode GEMM (M=32) stays faster on cuBLAS, so a per-shape pick (`mode="max-autotune"` with
ATEN+TRITON, which auto-selects the faster) gives the strict upper bound.
