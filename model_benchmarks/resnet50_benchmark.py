from typing import List, Tuple, Callable
from torch import float16

from kernels import test_matmul_iter, test_conv_iter
from pipeline_benchmark import pipeline_benchmark

RESNET50_ITERS = [50000] * 12


def resnet50_pipeline(N:int) -> List[Tuple[str, Callable]]:
    phases = []
    phases.append(("layer1_conv1",              lambda: test_conv_iter("layer1_conv1",              C=3,    G=1, M=64,   N=N, P=112, Q=112, R=7, S=7, HS=2, WS=2, datatype=float16, iters=RESNET50_ITERS[0])))
    phases.append(("layer2_layer1_0_conv1",     lambda: test_conv_iter("layer2_layer1_0_conv1",     C=64,   G=1, M=64,   N=N, P=56,  Q=56,  R=1, S=1, HS=1, WS=1, datatype=float16, iters=RESNET50_ITERS[1])))
    phases.append(("layer3_layer1_0_conv2",     lambda: test_conv_iter("layer3_layer1_0_conv2",     C=64,   G=1, M=64,   N=N, P=56,  Q=56,  R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET50_ITERS[2])))
    phases.append(("layer4_layer1_0_conv3",     lambda: test_conv_iter("layer4_layer1_0_conv3",     C=64,   G=1, M=256,  N=N, P=56,  Q=56,  R=1, S=1, HS=1, WS=1, datatype=float16, iters=RESNET50_ITERS[3])))
    phases.append(("layer51_layer4_2_conv1",    lambda: test_conv_iter("layer51_layer4_2_conv1",    C=2048, G=1, M=512,  N=N, P=7,   Q=7,   R=1, S=1, HS=1, WS=1, datatype=float16, iters=RESNET50_ITERS[4])))
    phases.append(("layer52_layer4_2_conv2",    lambda: test_conv_iter("layer52_layer4_2_conv2",    C=512,  G=1, M=512,  N=N, P=7,   Q=7,   R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET50_ITERS[5])))
    phases.append(("layer53_layer4_2_conv3",    lambda: test_conv_iter("layer53_layer4_2_conv3",    C=512,  G=1, M=2048, N=N, P=7,   Q=7,   R=1, S=1, HS=1, WS=1, datatype=float16, iters=RESNET50_ITERS[6])))

    phases.append(("layer54_fc",                lambda: test_matmul_iter("layer54_fc",              M=1000, K=2048, N=N,                                          datatype=float16, iters=RESNET50_ITERS[7])))
    
    phases.append(("layer5_layer1_0_dnsample",  lambda: test_conv_iter("layer5_layer1_0_dnsample",  C=64,   G=1, M=256,  N=N, P=56,  Q=56,  R=1, S=1, HS=1, WS=1, datatype=float16, iters=RESNET50_ITERS[8])))
    phases.append(("layer6_layer1_1_conv1",     lambda: test_conv_iter("layer6_layer1_1_conv1",     C=256,  G=1, M=64,   N=N, P=56,  Q=56,  R=1, S=1, HS=1, WS=1, datatype=float16, iters=RESNET50_ITERS[9])))
    phases.append(("layer7_layer1_1_conv2",     lambda: test_conv_iter("layer7_layer1_1_conv2",     C=64,   G=1, M=64,   N=N, P=56,  Q=56,  R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET50_ITERS[10])))
    phases.append(("layer8_layer1_1_conv3",     lambda: test_conv_iter("layer8_layer1_1_conv3",     C=64,   G=1, M=256,  N=N, P=56,  Q=56,  R=1, S=1, HS=1, WS=1, datatype=float16, iters=RESNET50_ITERS[11])))

    name = f"ResNet50_N{N}"
    return name, phases

B = [1, 32]  # Batch Sizes
resnet50_pipelines = [resnet50_pipeline(n) for n in B]

pipeline_benchmark(output_dir="benchmarks/resnet50", pipelines=resnet50_pipelines, device_index=0)