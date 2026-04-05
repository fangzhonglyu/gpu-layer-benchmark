from typing import List, Tuple, Callable
from torch import float16

from kernels import test_matmul_iter, test_conv_iter
from pipeline_benchmark import pipeline_benchmark

MBNETV3_ITERS = [50000] * 12


def mobilenet_v3_small_pipeline(N:int) -> List[Tuple[str, Callable]]:
    phases = []
    phases.append(("layer01_features_0_0",      lambda: test_conv_iter("layer01_features_0_0",     C=3,  G=1,  M=16,  N=N, P=112, Q=112, R=3, S=3, HS=2, WS=2, datatype=float16, iters=MBNETV3_ITERS[0])))
    phases.append(("layer02_features_1_b_0_0",  lambda: test_conv_iter("layer02_features_1_b_0_0", C=1,  G=16, M=1,   N=N, P=56,  Q=56,  R=3, S=3, HS=2, WS=2, datatype=float16, iters=MBNETV3_ITERS[1])))
    
    phases.append(("layer03_feat_1_b1_fc1",     lambda: test_matmul_iter("layer03_feat_1_b1_fc1",  M=8,  K=16, N=N, datatype=float16, iters=MBNETV3_ITERS[2])))
    phases.append(("layer04_feat_1_b1_fc2",     lambda: test_matmul_iter("layer04_feat_1_b1_fc2",  M=16, K=8,  N=N, datatype=float16, iters=MBNETV3_ITERS[3])))
    
    phases.append(("layer05_features_1_b_2_0",  lambda: test_conv_iter("layer05_features_1_b_2_0", C=16, G=1,  M=16,  N=N, P=56,  Q=56,  R=1, S=1, HS=1, WS=1, datatype=float16, iters=MBNETV3_ITERS[4])))
    phases.append(("layer052_features_12_0",    lambda: test_conv_iter("layer052_features_12_0",   C=96, G=1,  M=576, N=N, P=7,   Q=7,   R=1, S=1, HS=1, WS=1, datatype=float16, iters=MBNETV3_ITERS[5])))
    
    phases.append(("layer053_classifier_0",     lambda: test_matmul_iter("layer053_classifier_0",  M=1024, K=576,  N=N, datatype=float16, iters=MBNETV3_ITERS[6])))
    phases.append(("layer054_classifier_1",     lambda: test_matmul_iter("layer054_classifier_1",  M=1000, K=1024, N=N, datatype=float16, iters=MBNETV3_ITERS[7])))

    phases.append(("layer06_features_2_b_0_0",  lambda: test_conv_iter("layer06_features_2_b_0_0", C=16, G=1,  M=72,  N=N, P=56, Q=56, R=1, S=1, HS=1, WS=1, datatype=float16, iters=MBNETV3_ITERS[5])))
    phases.append(("layer07_features_2_b_1_0",  lambda: test_conv_iter("layer07_features_2_b_1_0", C=1,  G=72, M=1,   N=N, P=28, Q=28, R=3, S=3, HS=2, WS=2, datatype=float16, iters=MBNETV3_ITERS[6])))
    phases.append(("layer08_features_2_b_2_0",  lambda: test_conv_iter("layer08_features_2_b_2_0", C=72, G=1,  M=24,  N=N, P=28, Q=28, R=1, S=1, HS=1, WS=1, datatype=float16, iters=MBNETV3_ITERS[7])))
    phases.append(("layer09_features_3_b_0_0",  lambda: test_conv_iter("layer09_features_3_b_0_0", C=24, G=1,  M=88,  N=N, P=28, Q=28, R=1, S=1, HS=1, WS=1, datatype=float16, iters=MBNETV3_ITERS[8])))
    phases.append(("layer10_features_3_b_1_0",  lambda: test_conv_iter("layer10_features_3_b_1_0", C=1,  G=88, M=1,   N=N, P=28, Q=28, R=3, S=3, HS=1, WS=1, datatype=float16, iters=MBNETV3_ITERS[9])))
    phases.append(("layer11_features_3_b_2_0",  lambda: test_conv_iter("layer11_features_3_b_2_0", C=88, G=1,  M=24,  N=N, P=28, Q=28, R=1, S=1, HS=1, WS=1, datatype=float16, iters=MBNETV3_ITERS[10])))

    name = f"mobilenet_v3_small_b{N}_seq1"
    return name, phases

B = [1, 32]  # Batch Sizes
mobilenet_v3_pipelines = [mobilenet_v3_small_pipeline(n) for n in B]

pipeline_benchmark(output_dir="benchmarks/mobilenet_v3_small", pipelines=mobilenet_v3_pipelines, device_index=0)