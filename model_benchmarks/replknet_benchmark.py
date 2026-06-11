from typing import List, Tuple, Callable
from torch import float16

from kernels import test_conv_iter, device_from_env
from pipeline_benchmark import pipeline_benchmark

# Canonical (unique-in-DB) operators for replknet31b — match workloads/replknet31b/
# after alignment to unified_database. Other block operators are duplicates folded
# into the shared DB and live under workloads/replknet31b/additional/.
ITERS = [50000, 500, 50000, 50000]   # layer14 is a 29x29 large kernel -> fewer iters

# Operator DAG: CNN is a linear chain (create_cnn_cp_spec), one op per level,
# in network (layer-index) order.
LEVELS = [
    ['layer1_stages_0_blocks_0_pw1_conv'],
    ['layer14_stages_1_blocks_0_large_kernel_lkb_origin_conv'],
    ['layer15_stages_1_blocks_0_large_kernel_small_conv_conv'],
    ['layer16_stages_1_blocks_0_pw2_conv'],
]

# DAG edges for inter-chiplet P2P: linear chain over the canonical layers (CNN, no
# repeated block / wrap-around). Transfer bytes derive from each layer's output.
EDGES = [
    ('layer1_stages_0_blocks_0_pw1_conv',                      'layer14_stages_1_blocks_0_large_kernel_lkb_origin_conv'),
    ('layer14_stages_1_blocks_0_large_kernel_lkb_origin_conv', 'layer15_stages_1_blocks_0_large_kernel_small_conv_conv'),
    ('layer15_stages_1_blocks_0_large_kernel_small_conv_conv', 'layer16_stages_1_blocks_0_pw2_conv'),
]


def replknet_31b_pipeline(N: int) -> Tuple[str, List[Tuple[str, Callable]], List]:
    phases = [
        # stage0.block0.pw1 — 1x1 conv (128 -> 128), P=56
        ("layer1_stages_0_blocks_0_pw1_conv",                       lambda: test_conv_iter("layer1_stages_0_blocks_0_pw1_conv",                       C=128, G=1,   M=128, N=N, P=56, Q=56, R=1,  S=1,  HS=1, WS=1, datatype=float16, iters=ITERS[0])),
        # stage1.block0.large_kernel — 29x29 depthwise (256 groups), P=28
        ("layer14_stages_1_blocks_0_large_kernel_lkb_origin_conv",  lambda: test_conv_iter("layer14_stages_1_blocks_0_large_kernel_lkb_origin_conv",  C=1,   G=256, M=1,   N=N, P=28, Q=28, R=29, S=29, HS=1, WS=1, datatype=float16, iters=ITERS[1])),
        # stage1.block0.small_kernel — 5x5 depthwise (256 groups), P=28
        ("layer15_stages_1_blocks_0_large_kernel_small_conv_conv",  lambda: test_conv_iter("layer15_stages_1_blocks_0_large_kernel_small_conv_conv",  C=1,   G=256, M=1,   N=N, P=28, Q=28, R=5,  S=5,  HS=1, WS=1, datatype=float16, iters=ITERS[2])),
        # stage1.block0.pw2 — 1x1 conv (256 -> 256), P=28
        ("layer16_stages_1_blocks_0_pw2_conv",                      lambda: test_conv_iter("layer16_stages_1_blocks_0_pw2_conv",                      C=256, G=1,   M=256, N=N, P=28, Q=28, R=1,  S=1,  HS=1, WS=1, datatype=float16, iters=ITERS[3])),
    ]
    name = f"replknet31b_b{N}"
    return name, phases, LEVELS, EDGES


B = [1, 4, 8, 16]
pipelines = [replknet_31b_pipeline(n) for n in B]

pipeline_benchmark(output_dir="benchmarks/replknet_31b", pipelines=pipelines, device_index=device_from_env())
