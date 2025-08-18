from typing import List, Tuple, Callable
from torch import float16

from kernels import test_matmul_iter, test_conv_iter
from pipeline_benchmark import pipeline_benchmark

REPL_ITERS = [50000] * 11
REPL_ITERS[5] = 500


def replknet_31b_pipeline(N:int) -> List[Tuple[str, Callable]]:
    phases = []
    phases.append(("layer01_stem_0_conv",      lambda: test_conv_iter("layer01_stem_0_conv",     C=3,   G=1,    M=128,  N=N, P=112, Q=112, R=3,  S=3,  HS=2, WS=2, datatype=float16, iters=REPL_ITERS[0])))
    phases.append(("layer02_stem_1_conv",      lambda: test_conv_iter("layer02_stem_1_conv",     C=1,   G=128,  M=1,    N=N, P=112, Q=112, R=3,  S=3,  HS=1, WS=1, datatype=float16, iters=REPL_ITERS[1])))
    phases.append(("layer03_stem_2_conv",      lambda: test_conv_iter("layer03_stem_2_conv",     C=128, G=1,    M=128,  N=N, P=112, Q=112, R=1,  S=1,  HS=1, WS=1, datatype=float16, iters=REPL_ITERS[2])))
    phases.append(("layer04_stem_3_conv",      lambda: test_conv_iter("layer04_stem_3_conv",     C=1,   G=128,  M=1,    N=N, P=56,  Q=56,  R=3,  S=3,  HS=2, WS=2, datatype=float16, iters=REPL_ITERS[3])))
    phases.append(("layer05_s0b0_pw1_conv",    lambda: test_conv_iter("layer05_s0b0_pw1_conv",   C=128, G=1,    M=128,  N=N, P=56,  Q=56,  R=1,  S=1,  HS=1, WS=1, datatype=float16, iters=REPL_ITERS[4])))
    phases.append(("layer06_s0b0_lkb_conv",    lambda: test_conv_iter("layer06_s0b0_lkb_conv",   C=1,   G=128,  M=1,    N=N, P=56,  Q=56,  R=31, S=31, HS=1, WS=1, datatype=float16, iters=REPL_ITERS[5])))
    phases.append(("layer07_s0b0_small_conv",  lambda: test_conv_iter("layer07_s0b0_small_conv", C=1,   G=128,  M=1,    N=N, P=56,  Q=56,  R=5,  S=5,  HS=1, WS=1, datatype=float16, iters=REPL_ITERS[6])))
    phases.append(("layer08_s0b0_pw2_conv",    lambda: test_conv_iter("layer08_s0b0_pw2_conv",   C=128, G=1,    M=128,  N=N, P=56,  Q=56,  R=1,  S=1,  HS=1, WS=1, datatype=float16, iters=REPL_ITERS[7])))

    phases.append(("layer147_s3b3_pw1_conv",   lambda: test_conv_iter("layer147_s3b3_pw1_conv", C=1021, G=1,    M=4096, N=N, P=7,   Q=7,   R=1,  S=1,  HS=1, WS=1, datatype=float16, iters=REPL_ITERS[8])))
    phases.append(("layer148_s3b3_pw2_conv",   lambda: test_conv_iter("layer148_s3b3_pw2_conv", C=4096, G=1,    M=1024, N=N, P=7,   Q=7,   R=1,  S=1,  HS=1, WS=1, datatype=float16, iters=REPL_ITERS[9])))
    
    phases.append(("layer149_head",          lambda: test_matmul_iter("layer149_head",          M=1000, K=1024, N=N, datatype=float16, iters=REPL_ITERS[10])))

    name = f"replknet31b_b{N}_seq1"
    return name, phases

B = [1, 32]  # Batch Sizes
replknet_31b_pipelines = [replknet_31b_pipeline(n) for n in B]

pipeline_benchmark(output_dir="benchmarks/replknet_31b", pipelines=replknet_31b_pipelines, device_index=0)