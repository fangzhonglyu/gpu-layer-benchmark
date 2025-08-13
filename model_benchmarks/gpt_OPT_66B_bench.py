from typing import List, Tuple
from itertools import product
from torch import float16

from kernels import test_matmul_iter, test_softmax_iter
from pipeline_benchmark import pipeline_benchmark

def prefill_pipelines() -> List[Tuple]:
    B = [1,2,4,8]               # Batch Sizes
    P = [256, 512, 1024, 2048]  # Lengths
    # M is the same as P
    O = [9216]                  # Original input embedding dimension
    D = [9216]                  # Output embedding dimension after projection
    H = [72]                    # Number of heads
    E = [128]                   # Per head embedding dimension
    F = [128]                   # Feature dim per head
    I = [36864]                 # 4× hidden dim, common in FFN

    prod = product(B, P, O, D, H, E, F, I)
    return [ gpt_OPT_66B_pipeline(b, p, p, o, d, h, e, f, i, PREFILL_ITERS) for b, p, o, d, h, e, f, i in prod ]

def decode_pipelines() -> List[Tuple]:
    B = [1,2,4,8]               # Batch Sizes
    P = [1]                     # Lengths
    M = [2048]                  # M is the output dimension for QKT
    O = [9216]                  # Original input embedding dimension
    D = [9216]                  # Output embedding dimension after projection
    H = [72]                    # Number of heads
    E = [128]                   # Per head embedding dimension
    F = [128]                   # Feature dim per head
    I = [36864]                 # 4× hidden dim, common in FFN

    prod = product(B, P, M, O, D, H, E, F, I)
    return [ gpt_OPT_66B_pipeline(b, p, m, o, d, h, e, f, i, DECODE_ITERS) for b, p, m, o, d, h, e, f, i in prod ]

DECODE_ITERS = [3000, 3000, 3000, 50000, 50000, 50000, 1000, 500, 500]
PREFILL_ITERS = [1000, 1000, 1000, 5000, 5000, 5000, 500, 500, 500]

def gpt_OPT_66B_pipeline(b, p, m, o, d, h, e, f, i, iters) -> List:
    phases = []
    phases.append(('Q-proj',        lambda: test_matmul_iter("Q-proj", b * p, o, d, float16, iters=iters[0])))
    phases.append(('K-proj',        lambda: test_matmul_iter("K-proj", b * p, o, d, float16, iters=iters[1])))
    phases.append(('V-proj',        lambda: test_matmul_iter("V-proj", b * p, o, d, float16, iters=iters[2])))
    phases.append(('QKT',           lambda: test_matmul_iter("QKT", h * b * p, e, m, float16, iters=iters[3])))
    phases.append(('Softmax',       lambda: test_softmax_iter("Softmax", h * b * p, m, float16, iters=iters[4])))
    phases.append(('AV',            lambda: test_matmul_iter("AV", h * b * p, m, f, float16, iters=iters[5])))
    phases.append(('Output-proj',   lambda: test_matmul_iter("Output-proj", b * p, h * f, o, float16, iters=iters[6])))
    phases.append(('FFN-1',         lambda: test_matmul_iter("FFN-1", b * p, o, i, float16, iters=iters[7])))
    phases.append(('Down-proj',     lambda: test_matmul_iter("Down-proj", b * p, i, o, float16, iters=iters[8])))
    name = f"B{b}_P{p}_M{m}"
    return name, phases

pipeline_benchmark(output_dir="benchmarks/gpt_OPT_66B_decode", pipelines=decode_pipelines(), device_index=0)
pipeline_benchmark(output_dir="benchmarks/gpt_OPT_66B_prefill", pipelines=prefill_pipelines(), device_index=0)
