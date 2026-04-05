from typing import List, Tuple
from itertools import product
from torch import float16

from kernels import test_matmul_iter, test_softmax_iter
from pipeline_benchmark import pipeline_benchmark

# Llama 3.1 8B architecture
HIDDEN   = 4096
HEADS    = 32
KV_HEADS = 8
HEAD_DIM = 128
Q_DIM    = HEADS * HEAD_DIM      # 4096
KV_DIM   = KV_HEADS * HEAD_DIM   # 1024
INTER    = 14336

#                Q-proj  K-proj  V-proj  QKT     Softmax AV      O-proj  Gate    Up      Down
PREFILL_ITERS = [20000,  20000,  20000,  50000,  50000,  50000,  20000,  8000,   8000,   8000]
DECODE_ITERS  = [20000,  20000,  20000,  50000,  50000,  50000,  20000,  8000,   8000,   8000]


def llama3_1_8b_pipeline(b, seq, ctx, iters) -> Tuple[str, List]:
    """
    b:   batch size
    seq: query sequence length (S for prefill, 1 for decode)
    ctx: key/value context length (= seq for prefill, = KV cache len for decode)
    """
    p = b * seq   # flattened batch × seq for linear projections
    phases = []
    phases.append(('Q-proj',      lambda: test_matmul_iter("Q-proj",      M=p,            K=HIDDEN, N=Q_DIM,  datatype=float16, iters=iters[0])))
    phases.append(('K-proj',      lambda: test_matmul_iter("K-proj",      M=p,            K=HIDDEN, N=KV_DIM, datatype=float16, iters=iters[1])))
    phases.append(('V-proj',      lambda: test_matmul_iter("V-proj",      M=p,            K=HIDDEN, N=KV_DIM, datatype=float16, iters=iters[2])))
    phases.append(('QKT',         lambda: test_matmul_iter("QKT",         M=HEADS*b*seq,  K=HEAD_DIM, N=ctx,  datatype=float16, iters=iters[3])))
    phases.append(('Softmax',     lambda: test_softmax_iter("Softmax",    N=HEADS*b*seq,  M=ctx,             datatype=float16, iters=iters[4])))
    phases.append(('AV',          lambda: test_matmul_iter("AV",          M=HEADS*b*seq,  K=ctx,    N=HEAD_DIM, datatype=float16, iters=iters[5])))
    phases.append(('O-proj',      lambda: test_matmul_iter("O-proj",      M=p,            K=Q_DIM,  N=HIDDEN, datatype=float16, iters=iters[6])))
    phases.append(('Gate-proj',   lambda: test_matmul_iter("Gate-proj",   M=p,            K=HIDDEN, N=INTER,  datatype=float16, iters=iters[7])))
    phases.append(('Up-proj',     lambda: test_matmul_iter("Up-proj",     M=p,            K=HIDDEN, N=INTER,  datatype=float16, iters=iters[8])))
    phases.append(('Down-proj',   lambda: test_matmul_iter("Down-proj",   M=p,            K=INTER,  N=HIDDEN, datatype=float16, iters=iters[9])))

    if seq == 1:
        name = f"llama3.1_8b_decode_b{b}_kv{ctx}"
    else:
        name = f"llama3.1_8b_prefill_b{b}_s{seq}"
    return name, phases


def prefill_pipelines() -> List[Tuple]:
    B = [1, 4, 8, 16]
    S = [512, 1024, 2048, 4096]
    return [llama3_1_8b_pipeline(b, s, s, PREFILL_ITERS) for b, s in product(B, S)]

def decode_pipelines() -> List[Tuple]:
    B = [1, 4, 8, 16]
    KV = [512, 1024, 2048, 4096]
    return [llama3_1_8b_pipeline(b, 1, kv, DECODE_ITERS) for b, kv in product(B, KV)]


pipeline_benchmark(output_dir="benchmarks/llama3.1_8b_prefill", pipelines=prefill_pipelines(), device_index=0)
pipeline_benchmark(output_dir="benchmarks/llama3.1_8b_decode",  pipelines=decode_pipelines(),  device_index=0)