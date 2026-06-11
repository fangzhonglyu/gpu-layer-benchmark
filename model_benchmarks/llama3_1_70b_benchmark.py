from typing import List, Tuple
from itertools import product
from torch import float16

from kernels import test_matmul_iter, test_fused_qk_softmax_iter, link_model_from_env, transfer_curve_from_env
from pipeline_benchmark import pipeline_benchmark

# Llama 3.1 70B architecture
HIDDEN   = 8192
HEADS    = 64
KV_HEADS = 8
HEAD_DIM = 128
Q_DIM    = HEADS * HEAD_DIM      # 8192
KV_DIM   = KV_HEADS * HEAD_DIM   # 1024
INTER    = 28672

# Canonical decoder-layer operators. attn_qk + softmax fused into one 'attn_scores'
# op (scores stay on-chip; test_fused_qk_softmax_iter). lm_head excluded.
#                q_proj  k_proj  v_proj  attn_scores  attn_v  o_proj  gate  up   down
PREFILL_ITERS = [10000,  10000,  10000,  50000,       50000,  10000,  500,  500, 500]
DECODE_ITERS  = [10000,  10000,  10000,  50000,       50000,  10000,  500,  500, 500]

# Operator DAG (Llama decoder layer). qk+softmax fusion now lives inside the
# 'attn_scores' measurement, so it is a single op/stage (no 'serial' stage).
#   {q,k,v}             parallel (v_proj off-critical-path, overlaps attention)
#   attn_scores         fused Q@K^T + softmax, scores on-chip
#   attn_v -> o_proj
#   {gate,up} -> down   parallel SwiGLU branches
LEVELS = [
    ['layer0_q_proj', 'layer0_k_proj', 'layer0_v_proj'],
    ['layer0_attn_scores'],
    ['layer0_attn_v'],
    ['layer0_o_proj'],
    ['layer0_gate_proj', 'layer0_up_proj'],
    ['layer0_down_proj'],
]

# DAG edges for inter-chiplet P2P energy + step-④ transfer latency. q/k feed the
# fused scores op; its softmax output feeds attn_v; down_proj wraps to next q/k/v.
EDGES = [
    ('layer0_down_proj',   'layer0_q_proj'),   # wrap: hidden -> next-layer q/k/v
    ('layer0_down_proj',   'layer0_k_proj'),
    ('layer0_down_proj',   'layer0_v_proj'),
    ('layer0_q_proj',      'layer0_attn_scores'),
    ('layer0_k_proj',      'layer0_attn_scores'),
    ('layer0_attn_scores', 'layer0_attn_v'),
    ('layer0_v_proj',      'layer0_attn_v'),
    ('layer0_attn_v',      'layer0_o_proj'),
    ('layer0_o_proj',      'layer0_gate_proj'),
    ('layer0_o_proj',      'layer0_up_proj'),
    ('layer0_gate_proj',   'layer0_down_proj'),
    ('layer0_up_proj',     'layer0_down_proj'),
]


def llama3_1_70b_pipeline(b, seq, ctx, iters) -> Tuple[str, List, List]:
    """
    b:   batch size
    seq: query sequence length (S for prefill, 1 for decode)
    ctx: key/value context length (= seq for prefill, = KV cache len for decode)
    """
    p = b * seq
    phases = [
        ('layer0_q_proj',      lambda: test_matmul_iter("layer0_q_proj",   M=p, K=HIDDEN, N=Q_DIM,  datatype=float16, iters=iters[0])),
        ('layer0_k_proj',      lambda: test_matmul_iter("layer0_k_proj",   M=p, K=HIDDEN, N=KV_DIM, datatype=float16, iters=iters[1])),
        ('layer0_v_proj',      lambda: test_matmul_iter("layer0_v_proj",   M=p, K=HIDDEN, N=KV_DIM, datatype=float16, iters=iters[2])),
        ('layer0_attn_scores', lambda: test_fused_qk_softmax_iter("layer0_attn_scores", rows=HEADS*b*seq, head_dim=HEAD_DIM, ctx=ctx, datatype=float16, iters=iters[3])),
        ('layer0_attn_v',      lambda: test_matmul_iter("layer0_attn_v",   M=HEADS*b*seq, K=ctx, N=HEAD_DIM, datatype=float16, iters=iters[4])),
        ('layer0_o_proj',      lambda: test_matmul_iter("layer0_o_proj",   M=p, K=Q_DIM,  N=HIDDEN, datatype=float16, iters=iters[5])),
        ('layer0_gate_proj',   lambda: test_matmul_iter("layer0_gate_proj",M=p, K=HIDDEN, N=INTER,  datatype=float16, iters=iters[6])),
        ('layer0_up_proj',     lambda: test_matmul_iter("layer0_up_proj",  M=p, K=HIDDEN, N=INTER,  datatype=float16, iters=iters[7])),
        ('layer0_down_proj',   lambda: test_matmul_iter("layer0_down_proj",M=p, K=INTER,  N=HIDDEN, datatype=float16, iters=iters[8])),
    ]

    if seq == 1:
        name = f"llama3.1_70b_decode_b{b}_kv{ctx}"
    else:
        name = f"llama3.1_70b_prefill_b{b}_s{seq}"
    return name, phases, LEVELS, EDGES


def prefill_pipelines() -> List[Tuple]:
    B = [1, 4, 8, 16]
    S = [512, 1024, 2048, 4096]
    return [llama3_1_70b_pipeline(b, s, s, PREFILL_ITERS) for b, s in product(B, S)]

def decode_pipelines() -> List[Tuple]:
    B = [1, 4, 8, 16]
    KV = [512, 1024, 2048, 4096]
    return [llama3_1_70b_pipeline(b, 1, kv, DECODE_ITERS) for b, kv in product(B, KV)]


if __name__ == "__main__":
    LINK = link_model_from_env()  # PCIE_LINK_JSON=<path> to model step-④ transfer
    CURVE = transfer_curve_from_env(default_measured=True)  # size-dependent bw+pj/bit; PCIE_CURVE_JSON=<path> overrides
    pipeline_benchmark(output_dir="benchmarks/llama3.1_70b_prefill", pipelines=prefill_pipelines(), device_index=0, link_model=LINK, transfer_curve=CURVE)
    pipeline_benchmark(output_dir="benchmarks/llama3.1_70b_decode",  pipelines=decode_pipelines(),  device_index=0, link_model=LINK, transfer_curve=CURVE)
