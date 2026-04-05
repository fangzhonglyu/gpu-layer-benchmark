from typing import List, Tuple, Callable
from torch import float16

from kernels import test_matmul_iter, test_conv_iter
from pipeline_benchmark import pipeline_benchmark

ITERS = [50000] * 13


def mobilenet_v3_small_pipeline(N: int) -> Tuple[str, List[Tuple[str, Callable]]]:
    phases = []
    # --- Block: features_2 (early, P=56→28) ---
    phases.append(("layer06_feat2_expand",    lambda: test_conv_iter("layer06_feat2_expand",    C=16, G=1,   M=72,  N=N, P=56, Q=56, R=1, S=1, HS=1, WS=1, datatype=float16, iters=ITERS[0])))
    phases.append(("layer07_feat2_dw",        lambda: test_conv_iter("layer07_feat2_dw",        C=1,  G=72,  M=1,   N=N, P=28, Q=28, R=3, S=3, HS=2, WS=2, datatype=float16, iters=ITERS[1])))
    phases.append(("layer08_feat2_project",   lambda: test_conv_iter("layer08_feat2_project",   C=72, G=1,   M=24,  N=N, P=28, Q=28, R=1, S=1, HS=1, WS=1, datatype=float16, iters=ITERS[2])))

    # --- Block: features_5 (middle, P=14, with SE) ---
    phases.append(("layer17_feat5_expand",    lambda: test_conv_iter("layer17_feat5_expand",    C=40,  G=1,   M=240, N=N, P=14, Q=14, R=1, S=1, HS=1, WS=1, datatype=float16, iters=ITERS[3])))
    phases.append(("layer18_feat5_dw",        lambda: test_conv_iter("layer18_feat5_dw",        C=1,   G=240, M=1,   N=N, P=14, Q=14, R=5, S=5, HS=1, WS=1, datatype=float16, iters=ITERS[4])))
    phases.append(("layer19_feat5_se_fc1",    lambda: test_matmul_iter("layer19_feat5_se_fc1",  M=64,  K=240, N=N, datatype=float16, iters=ITERS[5])))
    phases.append(("layer20_feat5_se_fc2",    lambda: test_matmul_iter("layer20_feat5_se_fc2",  M=240, K=64,  N=N, datatype=float16, iters=ITERS[6])))
    phases.append(("layer21_feat5_project",   lambda: test_conv_iter("layer21_feat5_project",   C=240, G=1,   M=40,  N=N, P=14, Q=14, R=1, S=1, HS=1, WS=1, datatype=float16, iters=ITERS[7])))

    # --- Block: features_10 (late, P=7, with SE) ---
    phases.append(("layer42_feat10_expand",   lambda: test_conv_iter("layer42_feat10_expand",   C=96,  G=1,   M=576, N=N, P=7, Q=7, R=1, S=1, HS=1, WS=1, datatype=float16, iters=ITERS[8])))
    phases.append(("layer43_feat10_dw",       lambda: test_conv_iter("layer43_feat10_dw",       C=1,   G=576, M=1,   N=N, P=7, Q=7, R=5, S=5, HS=1, WS=1, datatype=float16, iters=ITERS[9])))
    phases.append(("layer44_feat10_se_fc1",   lambda: test_matmul_iter("layer44_feat10_se_fc1", M=144, K=576, N=N, datatype=float16, iters=ITERS[10])))
    phases.append(("layer45_feat10_se_fc2",   lambda: test_matmul_iter("layer45_feat10_se_fc2", M=576, K=144, N=N, datatype=float16, iters=ITERS[11])))
    phases.append(("layer46_feat10_project",  lambda: test_conv_iter("layer46_feat10_project",  C=576, G=1,   M=96,  N=N, P=7, Q=7, R=1, S=1, HS=1, WS=1, datatype=float16, iters=ITERS[12])))

    name = f"mobilenet_v3_small_b{N}"
    return name, phases


B = [1, 4, 8, 16]
pipelines = [mobilenet_v3_small_pipeline(n) for n in B]

pipeline_benchmark(output_dir="benchmarks/mobilenet_v3_small", pipelines=pipelines, device_index=0)