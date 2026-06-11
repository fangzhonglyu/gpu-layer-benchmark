from typing import List, Tuple, Callable
from torch import float16

from kernels import test_matmul_iter, test_conv_iter, device_from_env
from pipeline_benchmark import pipeline_benchmark

# Canonical (unique-in-DB) operators for mobilenet_v3_small — match workloads/
# mobilenet_v3_small/ after alignment to unified_database. The other block operators
# are duplicates folded into the shared DB and live under workloads/.../additional/.
ITERS = [50000] * 5

# Operator DAG: CNN is a linear chain (create_cnn_cp_spec), one op per level,
# in network (layer-index) order.
LEVELS = [
    ['layer6_features_2_block_0_0'],
    ['layer7_features_2_block_1_0'],
    ['layer8_features_2_block_2_0'],
    ['layer19_features_5_block_2_fc1'],
    ['layer43_features_10_block_1_0'],
]

# DAG edges for inter-chiplet P2P: linear chain over the canonical layers (CNN, no
# repeated block / wrap-around). Transfer bytes derive from each layer's output.
EDGES = [
    ('layer6_features_2_block_0_0',    'layer7_features_2_block_1_0'),
    ('layer7_features_2_block_1_0',    'layer8_features_2_block_2_0'),
    ('layer8_features_2_block_2_0',    'layer19_features_5_block_2_fc1'),
    ('layer19_features_5_block_2_fc1', 'layer43_features_10_block_1_0'),
]


def mobilenet_v3_small_pipeline(N: int) -> Tuple[str, List[Tuple[str, Callable]], List]:
    phases = [
        # features.2.block.0.0 — 1x1 expand conv (16 -> 72), P=56
        ("layer6_features_2_block_0_0",   lambda: test_conv_iter("layer6_features_2_block_0_0",   C=16,  G=1,   M=72, N=N, P=56, Q=56, R=1, S=1, HS=1, WS=1, datatype=float16, iters=ITERS[0])),
        # features.2.block.1.0 — 3x3 depthwise (72 groups), stride 2, P=28
        ("layer7_features_2_block_1_0",   lambda: test_conv_iter("layer7_features_2_block_1_0",   C=1,   G=72,  M=1,  N=N, P=28, Q=28, R=3, S=3, HS=2, WS=2, datatype=float16, iters=ITERS[1])),
        # features.2.block.2.0 — 1x1 project conv (72 -> 24), P=28
        ("layer8_features_2_block_2_0",   lambda: test_conv_iter("layer8_features_2_block_2_0",   C=72,  G=1,   M=24, N=N, P=28, Q=28, R=1, S=1, HS=1, WS=1, datatype=float16, iters=ITERS[2])),
        # features.5.block.2.fc1 — SE squeeze fc1 (240 -> 64)
        ("layer19_features_5_block_2_fc1",lambda: test_matmul_iter("layer19_features_5_block_2_fc1", M=64, K=240, N=N, datatype=float16, iters=ITERS[3])),
        # features.10.block.1.0 — 5x5 depthwise (576 groups), P=7
        ("layer43_features_10_block_1_0", lambda: test_conv_iter("layer43_features_10_block_1_0", C=1,   G=576, M=1,  N=N, P=7, Q=7, R=5, S=5, HS=1, WS=1, datatype=float16, iters=ITERS[4])),
    ]
    name = f"mobilenet_v3_small_b{N}"
    return name, phases, LEVELS, EDGES


B = [1, 4, 8, 16]
pipelines = [mobilenet_v3_small_pipeline(n) for n in B]

pipeline_benchmark(output_dir="benchmarks/mobilenet_v3_small", pipelines=pipelines, device_index=device_from_env())
