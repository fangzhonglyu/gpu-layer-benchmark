"""Re-run the two largest models (llama3.1-70B, qwen3-235B-A22B) under the optimized
backend in bf16, skipping prefill configs that OOM the compiled path.

The b16/s4096 prefill attn_scores QK output is [HEADS*b*seq, ctx] = 4194304x4096 = 34 GB.
Inductor max-autotune benchmarks several candidate kernels and holds ~2-3x that buffer
at once, which exceeds the 95 GB card. (Eager fit it: no autotune buffers, in-place
softmax.) Decode attention is tiny (seq=1), so its full grid runs. The dropped configs
are printed so the gap is explicit, not silent.

    BIG_MODEL=llama70b  BENCH_DEVICE=0 COMPILE_BACKEND=1 BENCH_DTYPE=bf16 \
        BENCH_OUT_SUFFIX=_compiled_bf16 python run_big_bf16.py
"""
import os
import sys
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "model_benchmarks"))

from pipeline_benchmark import pipeline_benchmark
from kernels import transfer_curve_from_env, link_model_from_env, device_from_env

SKIP_PREFILL = {(16, 4096)}   # (batch, seq) too big for compiled autotune on 95 GB

MODEL = os.environ.get("BIG_MODEL", "llama70b")
if MODEL == "llama70b":
    from llama3_1_70b_benchmark import (llama3_1_70b_pipeline as PIPE,
                                        PREFILL_ITERS, DECODE_ITERS)
    PRE, DEC = "benchmarks/llama3.1_70b_prefill", "benchmarks/llama3.1_70b_decode"
elif MODEL == "qwen235b":
    from qwen3_235b_a22b_benchmark import (qwen3_235b_pipeline as PIPE,
                                           PREFILL_ITERS, DECODE_ITERS)
    PRE, DEC = "benchmarks/qwen3_235b_a22b_prefill", "benchmarks/qwen3_235b_a22b_decode"
else:
    raise SystemExit(f"unknown BIG_MODEL={MODEL!r}")

CURVE = transfer_curve_from_env(default_measured=True)
LINK = link_model_from_env()
DEV = device_from_env()

B = [1, 4, 8, 16]
S = [512, 1024, 2048, 4096]

prefill = [PIPE(b, s, s, PREFILL_ITERS) for b, s in product(B, S) if (b, s) not in SKIP_PREFILL]
decode = [PIPE(b, 1, kv, DECODE_ITERS) for b, kv in product(B, S)]

dropped = [(b, s) for b, s in product(B, S) if (b, s) in SKIP_PREFILL]
print(f"[{MODEL}] prefill: {len(prefill)} configs ({len(dropped)} dropped for compiled OOM: {dropped}); "
      f"decode: {len(decode)} configs")

pipeline_benchmark(output_dir=PRE, pipelines=prefill, device_index=DEV, link_model=LINK, transfer_curve=CURVE)
pipeline_benchmark(output_dir=DEC, pipelines=decode, device_index=DEV, link_model=LINK, transfer_curve=CURVE)
