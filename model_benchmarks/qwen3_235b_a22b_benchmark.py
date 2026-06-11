import math
from typing import List, Tuple
from itertools import product
from torch import float16

from kernels import test_matmul_iter, test_fused_qk_softmax_iter, link_model_from_env, transfer_curve_from_env, device_from_env
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

# Canonical decoder-layer operators (match workloads/<net>/ and unified_database).
# lm_head excluded (global op). expert_up_proj is NOT a separate op: it has the same
# dimensions as expert_gate_proj and is deduplicated to it in the unified DB / cp_spec.
# attn_qk + softmax fused into one 'attn_scores' op (scores stay on-chip).
#                q_proj  k_proj  v_proj  attn_scores  attn_v  o_proj  router  e_gate  e_down
PREFILL_ITERS = [10000,  10000,  10000,  30000,       30000,  10000,  30000,  10000,  10000]
DECODE_ITERS  = [10000,  10000,  10000,  50000,       50000,  10000,  50000,  20000,  20000]

# Operator DAG (Qwen3 MoE decoder layer). Attention: qk+softmax fused into a single
# 'attn_scores' op (scores stay on-chip; test_fused_qk_softmax_iter). MLP is MoE:
# o_proj -> router -> expert_gate -> expert_down (expert_up folded into gate).
LEVELS = [
    ['layer0_q_proj', 'layer0_k_proj', 'layer0_v_proj'],
    ['layer0_attn_scores'],
    ['layer0_attn_v'],
    ['layer0_o_proj'],
    ['router'],
    ['expert_gate_proj'],
    ['expert_down_proj'],
]


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
    """Profile single-expert GEMM, scale to all experts on device × NUM_DEVICES.

    Latency is per-device (experts on one device run serially, devices run in
    parallel); energy is the whole system (all experts on all devices).
    """
    result = test_matmul_iter(name, **kwargs)
    result['avg_latency_ms'] *= experts_on_device
    result['avg_energy_J']   *= experts_on_device * NUM_DEVICES
    return result


def qwen3_235b_pipeline(b, seq, ctx, iters) -> Tuple[str, List, List]:
    """
    b:   batch size
    seq: query sequence length (S for prefill, 1 for decode)
    ctx: key/value context length (= seq for prefill, = KV cache len for decode)
    """
    p     = b * seq
    eod, tpe = expert_gemm_params(b, seq)

    # DAG edges for inter-chiplet P2P (mirrors create_qwen_moe_cp_spec). Attention
    # edges derive bytes from each op's output; MoE edges use explicit whole-batch
    # volumes because the measured expert ops are single-expert GEMMs:
    #   dispatch  o_proj -> experts : every token sent to ACTIVE_EXPERTS experts
    #   expert intermediate gate->down : ACTIVE_EXPERTS pairs per token
    #   combine   expert_down -> next-layer q/k/v (hidden), wrap-around
    disp_elems   = p * ACTIVE_EXPERTS * HIDDEN      # token dispatch to experts
    einter_elems = p * ACTIVE_EXPERTS * MOE_INTER   # expert gate output -> down
    hidden_elems = p * HIDDEN                        # combined hidden activation
    edges = [
        ('expert_down_proj', 'layer0_q_proj', hidden_elems),   # wrap: hidden -> next q/k/v
        ('expert_down_proj', 'layer0_k_proj', hidden_elems),
        ('expert_down_proj', 'layer0_v_proj', hidden_elems),
        ('layer0_q_proj',    'layer0_attn_scores'),
        ('layer0_k_proj',    'layer0_attn_scores'),
        ('layer0_attn_scores', 'layer0_attn_v'),
        ('layer0_v_proj',    'layer0_attn_v'),
        ('layer0_attn_v',    'layer0_o_proj'),
        ('layer0_o_proj',    'router'),
        ('layer0_o_proj',    'expert_gate_proj', disp_elems),  # dispatch hidden tokens
        ('router',           'expert_gate_proj'),              # gating control (router logits)
        ('expert_gate_proj', 'expert_down_proj', einter_elems),
    ]

    phases = [
        ('layer0_q_proj',    lambda: test_matmul_iter("layer0_q_proj",   M=p,           K=HIDDEN,   N=Q_DIM,       datatype=float16, iters=iters[0])),
        ('layer0_k_proj',    lambda: test_matmul_iter("layer0_k_proj",   M=p,           K=HIDDEN,   N=KV_DIM,      datatype=float16, iters=iters[1])),
        ('layer0_v_proj',    lambda: test_matmul_iter("layer0_v_proj",   M=p,           K=HIDDEN,   N=KV_DIM,      datatype=float16, iters=iters[2])),
        ('layer0_attn_scores', lambda: test_fused_qk_softmax_iter("layer0_attn_scores", rows=HEADS*b*seq, head_dim=HEAD_DIM, ctx=ctx, datatype=float16, iters=iters[3])),
        ('layer0_attn_v',    lambda: test_matmul_iter("layer0_attn_v",   M=HEADS*b*seq, K=ctx,      N=HEAD_DIM,    datatype=float16, iters=iters[4])),
        ('layer0_o_proj',    lambda: test_matmul_iter("layer0_o_proj",   M=p,           K=Q_DIM,    N=HIDDEN,      datatype=float16, iters=iters[5])),
        ('router',           lambda: test_matmul_iter("router",          M=p,           K=HIDDEN,   N=NUM_EXPERTS, datatype=float16, iters=iters[6])),
        ('expert_gate_proj', lambda: expert_phase("expert_gate_proj",    experts_on_device=eod, M=tpe, K=HIDDEN,    N=MOE_INTER, datatype=float16, iters=iters[7])),
        ('expert_down_proj', lambda: expert_phase("expert_down_proj",    experts_on_device=eod, M=tpe, K=MOE_INTER, N=HIDDEN,    datatype=float16, iters=iters[8])),
    ]

    if seq == 1:
        name = f"qwen3_235b_a22b_decode_b{b}_kv{ctx}"
    else:
        name = f"qwen3_235b_a22b_prefill_b{b}_s{seq}"
    return name, phases, LEVELS, edges


def prefill_pipelines() -> List[Tuple]:
    B = [1, 4, 8, 16]
    S = [512, 1024, 2048, 4096]
    return [qwen3_235b_pipeline(b, s, s, PREFILL_ITERS) for b, s in product(B, S)]

def decode_pipelines() -> List[Tuple]:
    B = [1, 4, 8, 16]
    KV = [512, 1024, 2048, 4096]
    return [qwen3_235b_pipeline(b, 1, kv, DECODE_ITERS) for b, kv in product(B, KV)]


if __name__ == "__main__":
    LINK = link_model_from_env()  # PCIE_LINK_JSON=<path> to model step-④ transfer
    CURVE = transfer_curve_from_env(default_measured=True)  # size-dependent bw+pj/bit; PCIE_CURVE_JSON=<path> overrides
    DEVICE = device_from_env()  # BENCH_DEVICE=k to pin this run to a card (multi-GPU launch)
    pipeline_benchmark(output_dir="benchmarks/qwen3_235b_a22b_prefill", pipelines=prefill_pipelines(), device_index=DEVICE, link_model=LINK, transfer_curve=CURVE)
    pipeline_benchmark(output_dir="benchmarks/qwen3_235b_a22b_decode",  pipelines=decode_pipelines(),  device_index=DEVICE, link_model=LINK, transfer_curve=CURVE)
