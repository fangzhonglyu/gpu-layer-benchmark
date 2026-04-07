import math
from typing import List, Tuple
from itertools import product
from torch import float16

from kernels import test_matmul_iter, test_softmax_iter
from pipeline_benchmark import pipeline_benchmark

# Qwen3 235B-A22B (MoE) architecture
HIDDEN     = 4096
HEADS      = 64
KV_HEADS   = 4
HEAD_DIM   = 128
Q_DIM      = HEADS * HEAD_DIM       # 8192
KV_DIM     = KV_HEADS * HEAD_DIM    # 512
# MoE config
NUM_EXPERTS     = 128
ACTIVE_EXPERTS  = 8
MOE_INTER       = 1536  # moe_intermediate_size per expert
NUM_DEVICES     = 8     # expert-parallel GPUs

#                Q-proj  K-proj  V-proj  QKT     Softmax AV      O-proj  Router  Gate    Up      Down
PREFILL_ITERS = [10000,  10000,  10000,  30000,  30000,  30000,  10000,  30000,  10000,  10000,  10000]
DECODE_ITERS  = [10000,  10000,  10000,  50000,  50000,  50000,  10000,  50000,  20000,  20000,  20000]


def expert_gemm_params(b, seq):
    """Per-device expert GEMM parameters for MoE.

    Returns (experts_on_device, tokens_per_expert).
    Each expert has its own weight matrix, so they are separate GEMMs.

    Uniform routing: B*S*top_k token-expert pairs across NUM_EXPERTS.
    - total_pairs >= NUM_EXPERTS → all experts active, tpe = total_pairs / NUM_EXPERTS
    - total_pairs <  NUM_EXPERTS → only total_pairs experts active, tpe = 1
    Active experts evenly split across NUM_DEVICES GPUs.
    """
    total_pairs = b * seq * ACTIVE_EXPERTS
    if total_pairs >= NUM_EXPERTS:
        tpe = total_pairs // NUM_EXPERTS
        num_active = NUM_EXPERTS
    else:
        tpe = 1
        num_active = total_pairs
    experts_on_device = math.ceil(num_active / NUM_DEVICES)
    return experts_on_device, tpe


def expert_phase(name, experts_on_device, **kwargs):
    """Profile single-expert GEMM, scale to all experts on device × NUM_DEVICES."""
    result = test_matmul_iter(name, **kwargs)
    result['avg_latency_ms'] *= experts_on_device
    result['avg_energy_J']   *= experts_on_device * NUM_DEVICES
    return result


def qwen3_235b_pipeline(b, seq, ctx, iters) -> Tuple[str, List]:
    """
    b:   batch size
    seq: query sequence length (S for prefill, 1 for decode)
    ctx: key/value context length (= seq for prefill, = KV cache len for decode)
    """
    p     = b * seq
    eod, tpe = expert_gemm_params(b, seq)
    phases = []
    phases.append(('Q-proj',      lambda: test_matmul_iter("Q-proj",      M=p,            K=HIDDEN, N=Q_DIM,       datatype=float16, iters=iters[0])))
    phases.append(('K-proj',      lambda: test_matmul_iter("K-proj",      M=p,            K=HIDDEN, N=KV_DIM,      datatype=float16, iters=iters[1])))
    phases.append(('V-proj',      lambda: test_matmul_iter("V-proj",      M=p,            K=HIDDEN, N=KV_DIM,      datatype=float16, iters=iters[2])))
    phases.append(('QKT',         lambda: test_matmul_iter("QKT",         M=HEADS*b*seq,  K=HEAD_DIM, N=ctx,       datatype=float16, iters=iters[3])))
    phases.append(('Softmax',     lambda: test_softmax_iter("Softmax",    N=HEADS*b*seq,  M=ctx,                   datatype=float16, iters=iters[4])))
    phases.append(('AV',          lambda: test_matmul_iter("AV",          M=HEADS*b*seq,  K=ctx,    N=HEAD_DIM,    datatype=float16, iters=iters[5])))
    phases.append(('O-proj',      lambda: test_matmul_iter("O-proj",      M=p,            K=Q_DIM,  N=HIDDEN,      datatype=float16, iters=iters[6])))
    phases.append(('Router',      lambda: test_matmul_iter("Router",      M=p,            K=HIDDEN, N=NUM_EXPERTS, datatype=float16, iters=iters[7])))
    phases.append(('Gate-proj',   lambda: expert_phase("Gate-proj",       experts_on_device=eod, M=tpe, K=HIDDEN, N=MOE_INTER,   datatype=float16, iters=iters[8])))
    phases.append(('Up-proj',     lambda: expert_phase("Up-proj",         experts_on_device=eod, M=tpe, K=HIDDEN, N=MOE_INTER,   datatype=float16, iters=iters[9])))
    phases.append(('Down-proj',   lambda: expert_phase("Down-proj",       experts_on_device=eod, M=tpe, K=MOE_INTER, N=HIDDEN,   datatype=float16, iters=iters[10])))

    if seq == 1:
        name = f"qwen3_235b_a22b_decode_b{b}_kv{ctx}"
    else:
        name = f"qwen3_235b_a22b_prefill_b{b}_s{seq}"
    return name, phases


def prefill_pipelines() -> List[Tuple]:
    B = [1, 4, 8, 16]
    S = [512, 1024, 2048, 4096]
    return [qwen3_235b_pipeline(b, s, s, PREFILL_ITERS) for b, s in product(B, S)]

def decode_pipelines() -> List[Tuple]:
    B = [1, 4, 8, 16]
    KV = [512, 1024, 2048, 4096]
    return [qwen3_235b_pipeline(b, 1, kv, DECODE_ITERS) for b, kv in product(B, KV)]


# pipeline_benchmark(output_dir="benchmarks/qwen3_235b_a22b_prefill", pipelines=prefill_pipelines(), device_index=0)
pipeline_benchmark(output_dir="benchmarks/qwen3_235b_a22b_decode",  pipelines=decode_pipelines(),  device_index=0)