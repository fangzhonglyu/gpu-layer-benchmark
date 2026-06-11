from typing import List, Tuple
from itertools import product
from torch import float16

from kernels import test_matmul_iter, test_fused_qk_softmax_iter, link_model_from_env, transfer_curve_from_env
from pipeline_benchmark import pipeline_benchmark

# Vision Transformer encoder-block baselines. Standard MHA (no GQA), GELU MLP
# (fc1 + fc2, no SwiGLU up_proj). Canonical operators match workloads/vit_*/ and
# unified_database (9 ops; lm_head / classifier head excluded).
#
# Q_DIM = HEADS * HEAD_DIM == HIDDEN for all three configs, so k/v/o all use HIDDEN.
VIT_CONFIGS = {
    # name            HIDDEN  HEADS  HEAD_DIM  INTER  S
    'vit_b16_s197': dict(HIDDEN=768,  HEADS=12, HEAD_DIM=64, INTER=3072, S=197),
    'vit_l16_s197': dict(HIDDEN=1024, HEADS=16, HEAD_DIM=64, INTER=4096, S=197),
    'vit_h14_s257': dict(HIDDEN=1280, HEADS=16, HEAD_DIM=80, INTER=5120, S=257),
}

#         q_proj  k_proj  v_proj  attn_scores  attn_v  o_proj  fc1(gate) fc2(down)
ITERS = [ 20000,  20000,  20000,  50000,       50000,  20000,  10000,    10000]

# Operator DAG (ViT encoder block). GELU MLP has no up_proj:
# o_proj -> gate(fc1) -> down(fc2). qk+softmax fused into a single 'attn_scores' op
# (scores stay on-chip; test_fused_qk_softmax_iter).
LEVELS = [
    ['layer0_q_proj', 'layer0_k_proj', 'layer0_v_proj'],
    ['layer0_attn_scores'],
    ['layer0_attn_v'],
    ['layer0_o_proj'],
    ['layer0_gate_proj'],
    ['layer0_down_proj'],
]

# DAG edges for inter-chiplet P2P energy + step-④ transfer (ViT; GELU MLP, no up_proj).
EDGES = [
    ('layer0_down_proj',   'layer0_q_proj'),
    ('layer0_down_proj',   'layer0_k_proj'),
    ('layer0_down_proj',   'layer0_v_proj'),
    ('layer0_q_proj',      'layer0_attn_scores'),
    ('layer0_k_proj',      'layer0_attn_scores'),
    ('layer0_attn_scores', 'layer0_attn_v'),
    ('layer0_v_proj',      'layer0_attn_v'),
    ('layer0_attn_v',      'layer0_o_proj'),
    ('layer0_o_proj',      'layer0_gate_proj'),
    ('layer0_gate_proj',   'layer0_down_proj'),
]


def vit_pipeline(net_name: str, b: int) -> Tuple[str, List, List]:
    cfg = VIT_CONFIGS[net_name]
    H, NH, D, I, S = cfg['HIDDEN'], cfg['HEADS'], cfg['HEAD_DIM'], cfg['INTER'], cfg['S']
    Q_DIM = NH * D   # == H for standard MHA
    p = b * S
    phases = [
        ('layer0_q_proj',      lambda: test_matmul_iter("layer0_q_proj",   M=p, K=H, N=Q_DIM, datatype=float16, iters=ITERS[0])),
        ('layer0_k_proj',      lambda: test_matmul_iter("layer0_k_proj",   M=p, K=H, N=Q_DIM, datatype=float16, iters=ITERS[1])),
        ('layer0_v_proj',      lambda: test_matmul_iter("layer0_v_proj",   M=p, K=H, N=Q_DIM, datatype=float16, iters=ITERS[2])),
        ('layer0_attn_scores', lambda: test_fused_qk_softmax_iter("layer0_attn_scores", rows=NH*b*S, head_dim=D, ctx=S, datatype=float16, iters=ITERS[3])),
        ('layer0_attn_v',      lambda: test_matmul_iter("layer0_attn_v",   M=NH*b*S, K=S, N=D, datatype=float16, iters=ITERS[4])),
        ('layer0_o_proj',      lambda: test_matmul_iter("layer0_o_proj",   M=p, K=Q_DIM, N=H, datatype=float16, iters=ITERS[5])),
        ('layer0_gate_proj',   lambda: test_matmul_iter("layer0_gate_proj",M=p, K=H, N=I,     datatype=float16, iters=ITERS[6])),
        ('layer0_down_proj',   lambda: test_matmul_iter("layer0_down_proj",M=p, K=I, N=H,     datatype=float16, iters=ITERS[7])),
    ]
    name = f"{net_name}_b{b}"
    return name, phases, LEVELS, EDGES


def vit_pipelines() -> List[Tuple]:
    B = [1, 4, 8, 16]
    return [vit_pipeline(net, b) for net, b in product(VIT_CONFIGS.keys(), B)]


if __name__ == "__main__":
    LINK = link_model_from_env()  # PCIE_LINK_JSON=<path> to model step-④ transfer
    CURVE = transfer_curve_from_env(default_measured=True)  # size-dependent bw+pj/bit; PCIE_CURVE_JSON=<path> overrides
    pipeline_benchmark(output_dir="benchmarks/vit", pipelines=vit_pipelines(), device_index=0, link_model=LINK, transfer_curve=CURVE)
