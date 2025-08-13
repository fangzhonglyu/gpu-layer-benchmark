from typing import List, Tuple, Callable
from torch import float16

from kernels import test_matmul_iter, test_conv_iter
from pipeline_benchmark import pipeline_benchmark

EFFNETB0_ITERS = [50000] * 12


def efficientnet_b0_pipeline(N:int) -> List[Tuple[str, Callable]]:
    phases = []
    phases.append(("layer01_features_0_0",      lambda: test_conv_iter("layer01_features_0_0",     C=3,  G=1,  M=32,  N=N, P=112, Q=112, R=3, S=3, HS=2, WS=2, datatype=float16, iters=EFFNETB0_ITERS[0])))
    phases.append(("layer02_features_1_b_0_0",  lambda: test_conv_iter("layer02_features_1_b_0_0", C=1,  G=32, M=1,   N=N, P=112, Q=112, R=3, S=3, HS=1, WS=1, datatype=float16, iters=EFFNETB0_ITERS[1])))

    phases.append(("layer03_feat_1_0_b1_fc1", lambda: test_matmul_iter("layer03_feat_1_0_b1_fc1",  M=8,  K=32, N=N, datatype=float16, iters=EFFNETB0_ITERS[2])))
    phases.append(("layer04_feat_1_0_b1_fc2", lambda: test_matmul_iter("layer04_feat_1_0_b1_fc2",  M=32, K=8,  N=N, datatype=float16, iters=EFFNETB0_ITERS[3])))

    phases.append(("layer05_features_1_b_2_0",  lambda: test_conv_iter("layer05_features_1_b_2_0", C=32, G=1,  M=16,  N=N, P=112, Q=112, R=1, S=1, HS=1, WS=1, datatype=float16, iters=EFFNETB0_ITERS[4])))
    phases.append(("layer06_features_2_b_0_0",  lambda: test_conv_iter("layer06_features_2_b_0_0", C=16, G=1,  M=96,  N=N, P=112, Q=112, R=1, S=1, HS=1, WS=1, datatype=float16, iters=EFFNETB0_ITERS[5])))
    phases.append(("layer07_features_2_b_1_0",  lambda: test_conv_iter("layer07_features_2_b_1_0", C=1,  G=96, M=1,   N=N, P=56,  Q=56,  R=3, S=3, HS=2, WS=2, datatype=float16, iters=EFFNETB0_ITERS[6])))

    phases.append(("layer08_feat_2_0_b2_fc1", lambda: test_matmul_iter("layer08_feat_2_0_b2_fc1",  M=4,  K=96, N=N, datatype=float16, iters=EFFNETB0_ITERS[7])))
    phases.append(("layer09_feat_2_0_b2_fc2", lambda: test_matmul_iter("layer09_feat_2_0_b2_fc2",  M=96, K=4,  N=N, datatype=float16, iters=EFFNETB0_ITERS[8])))

    name = f"EfficientNetB0_N{N}"
    return name, phases

B = [1, 32]  # Batch Sizes
efficientnet_b0_pipelines = [efficientnet_b0_pipeline(n) for n in B]

pipeline_benchmark(output_dir="benchmarks/efficientnet_b0", pipelines=efficientnet_b0_pipelines, device_index=0)