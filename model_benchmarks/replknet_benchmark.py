from typing import List, Tuple, Callable
from torch import float16

from kernels import test_conv_iter
from pipeline_benchmark import pipeline_benchmark

ITERS = [50000] * 12
ITERS[1] = 500   # layer2  large kernel 31x31
ITERS[7] = 500   # layer14 large kernel 29x29


def replknet_31b_pipeline(N: int) -> Tuple[str, List[Tuple[str, Callable]]]:
    phases = []
    # --- Stage 0, Block 0 (P=56) ---
    phases.append(("layer01_s0b0_pw1",        lambda: test_conv_iter("layer01_s0b0_pw1",        C=128, G=1,   M=128, N=N, P=56, Q=56, R=1,  S=1,  HS=1, WS=1, datatype=float16, iters=ITERS[0])))
    phases.append(("layer02_s0b0_lk31",       lambda: test_conv_iter("layer02_s0b0_lk31",       C=1,   G=128, M=1,   N=N, P=56, Q=56, R=31, S=31, HS=1, WS=1, datatype=float16, iters=ITERS[1])))
    phases.append(("layer03_s0b0_sk5",        lambda: test_conv_iter("layer03_s0b0_sk5",        C=1,   G=128, M=1,   N=N, P=56, Q=56, R=5,  S=5,  HS=1, WS=1, datatype=float16, iters=ITERS[2])))
    phases.append(("layer04_s0b0_pw2",        lambda: test_conv_iter("layer04_s0b0_pw2",        C=128, G=1,   M=128, N=N, P=56, Q=56, R=1,  S=1,  HS=1, WS=1, datatype=float16, iters=ITERS[3])))

    # --- Stage 0, Block 1 (FFN-like, P=56) ---
    phases.append(("layer05_s0b1_pw1",        lambda: test_conv_iter("layer05_s0b1_pw1",        C=128, G=1,   M=512, N=N, P=56, Q=56, R=1,  S=1,  HS=1, WS=1, datatype=float16, iters=ITERS[4])))
    phases.append(("layer06_s0b1_pw2",        lambda: test_conv_iter("layer06_s0b1_pw2",        C=512, G=1,   M=128, N=N, P=56, Q=56, R=1,  S=1,  HS=1, WS=1, datatype=float16, iters=ITERS[5])))

    # --- Stage 1, Block 0 (P=28) ---
    phases.append(("layer13_s1b0_pw1",        lambda: test_conv_iter("layer13_s1b0_pw1",        C=256, G=1,   M=256,  N=N, P=28, Q=28, R=1,  S=1,  HS=1, WS=1, datatype=float16, iters=ITERS[6])))
    phases.append(("layer14_s1b0_lk29",       lambda: test_conv_iter("layer14_s1b0_lk29",       C=1,   G=256, M=1,    N=N, P=28, Q=28, R=29, S=29, HS=1, WS=1, datatype=float16, iters=ITERS[7])))
    phases.append(("layer15_s1b0_sk5",        lambda: test_conv_iter("layer15_s1b0_sk5",        C=1,   G=256, M=1,    N=N, P=28, Q=28, R=5,  S=5,  HS=1, WS=1, datatype=float16, iters=ITERS[8])))
    phases.append(("layer16_s1b0_pw2",        lambda: test_conv_iter("layer16_s1b0_pw2",        C=256, G=1,   M=256,  N=N, P=28, Q=28, R=1,  S=1,  HS=1, WS=1, datatype=float16, iters=ITERS[9])))

    # --- Stage 1, Block 1 (FFN-like, P=28) ---
    phases.append(("layer17_s1b1_pw1",        lambda: test_conv_iter("layer17_s1b1_pw1",        C=256,  G=1,  M=1024, N=N, P=28, Q=28, R=1,  S=1,  HS=1, WS=1, datatype=float16, iters=ITERS[10])))
    phases.append(("layer18_s1b1_pw2",        lambda: test_conv_iter("layer18_s1b1_pw2",        C=1024, G=1,  M=256,  N=N, P=28, Q=28, R=1,  S=1,  HS=1, WS=1, datatype=float16, iters=ITERS[11])))

    name = f"replknet31b_b{N}"
    return name, phases


B = [1, 4, 8, 16]
pipelines = [replknet_31b_pipeline(n) for n in B]

pipeline_benchmark(output_dir="benchmarks/replknet_31b", pipelines=pipelines, device_index=0)