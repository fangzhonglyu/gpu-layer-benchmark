"""Time ONE representative pipeline per model class to extrapolate total runtime of
all 7 benchmarks under the new time-target (1.5s) + pre-heat path."""
import time
from kernels import set_device, run_pipeline

import llama3_1_8b_benchmark as l8
import llama3_1_70b_benchmark as l70
import qwen3_30b_a3b_benchmark as q30
import qwen3_235b_a22b_benchmark as q235
import vit_benchmark as vit
import mobilenet_benchmark as mob
import replknet_benchmark as rep

set_device(0)
OUT = "/tmp/est_runtime"

# (label, single pipeline, number of pipelines that benchmark actually runs in __main__)
cases = [
    ("llama3.1_8b   decode", l8.llama3_1_8b_pipeline(8, 1, 2048, l8.DECODE_ITERS),   16),
    ("llama3.1_70b  decode", l70.llama3_1_70b_pipeline(8, 1, 2048, l70.DECODE_ITERS), 16),
    ("qwen3_30b_a3b decode", q30.qwen3_30b_pipeline(8, 1, 2048, q30.DECODE_ITERS),    16),
    ("qwen3_235b    decode", q235.qwen3_235b_pipeline(8, 1, 2048, q235.DECODE_ITERS), 16),
    ("vit_l16",              vit.vit_pipeline('vit_l16_s197', 8),                     12),
    ("mobilenet",            mob.mobilenet_v3_small_pipeline(8),                       4),
    ("replknet",             rep.replknet_31b_pipeline(8),                             4),
]

total = 0.0
print(f"{'benchmark':22s} {'1 pipeline':>12s} {'x N':>5s} {'subtotal':>10s}")
for label, pipe, n in cases:
    name, phases, *rest = pipe
    levels = rest[0] if rest else None
    edges = rest[1] if len(rest) > 1 else None
    t0 = time.perf_counter()
    run_pipeline(name, phases, OUT, levels=levels, edges=edges)
    dt = time.perf_counter() - t0
    sub = dt * n
    total += sub
    print(f"{label:22s} {dt:10.1f}s  {n:>4d}  {sub:9.1f}s")
print(f"{'TOTAL (7 benchmarks)':22s} {'':12s} {'':5s} {total/60:8.1f} min")
