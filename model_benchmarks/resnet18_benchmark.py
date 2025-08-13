from typing import List, Tuple, Callable
from torch import float16

from kernels import test_matmul_iter, test_conv_iter
from pipeline_benchmark import pipeline_benchmark

RESNET18_ITERS = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]


def resnet18_pipeline(N:int) -> List[Tuple[str, Callable]]:
    phases = []
    phases.append(("layer01_conv1",             lambda: test_conv_iter("layer01_conv1",             C=3,    G=1, M=64,   N=N, P=112, Q=112, R=7, S=7, HS=2, WS=2, datatype=float16, iters=RESNET18_ITERS[0])))
    phases.append(("layer02_layer1_0_conv1",    lambda: test_conv_iter("layer02_layer1_0_conv1",    C=64,   G=1, M=64,   N=N, P=56,  Q=56,  R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET18_ITERS[1])))
    phases.append(("layer03_layer1_0_conv2",    lambda: test_conv_iter("layer03_layer1_0_conv2",    C=64,   G=1, M=64,   N=N, P=56,  Q=56,  R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET18_ITERS[2])))
    phases.append(("layer04_layer1_1_conv3",    lambda: test_conv_iter("layer04_layer1_1_conv3",    C=64,   G=1, M=64,   N=N, P=56,  Q=56,  R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET18_ITERS[3])))
    phases.append(("layer05_layer1_1_conv1",    lambda: test_conv_iter("layer05_layer1_1_conv1",    C=64,   G=1, M=64,   N=N, P=56,  Q=56,  R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET18_ITERS[4])))
    phases.append(("layer06_layer2_0_conv1",    lambda: test_conv_iter("layer06_layer2_0_conv1",    C=64,   G=1, M=128,  N=N, P=28,  Q=28,  R=3, S=3, HS=2, WS=2, datatype=float16, iters=RESNET18_ITERS[5])))
    phases.append(("layer07_layer2_0_conv2",    lambda: test_conv_iter("layer07_layer2_0_conv2",    C=128,  G=1, M=128,  N=N, P=28,  Q=28,  R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET18_ITERS[6])))
    phases.append(("layer08_layer2_0_dn_samp",  lambda: test_conv_iter("layer08_layer2_0_dn_samp",  C=64,   G=1, M=128,  N=N, P=28,  Q=28,  R=1, S=1, HS=2, WS=2, datatype=float16, iters=RESNET18_ITERS[7])))
    phases.append(("layer09_layer2_1_conv1",    lambda: test_conv_iter("layer09_layer2_1_conv1",    C=128,  G=1, M=128,  N=N, P=28,  Q=28,  R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET18_ITERS[8])))
    phases.append(("layer10_layer2_1_conv2",    lambda: test_conv_iter("layer10_layer2_1_conv2",    C=128,  G=1, M=128,  N=N, P=28,  Q=28,  R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET18_ITERS[0])))
    phases.append(("layer11_layer3_0_conv1",    lambda: test_conv_iter("layer11_layer3_0_conv1",    C=128,  G=1, M=256,  N=N, P=14,  Q=14,  R=3, S=3, HS=2, WS=2, datatype=float16, iters=RESNET18_ITERS[1])))
    phases.append(("layer12_layer3_0_conv2",    lambda: test_conv_iter("layer12_layer3_0_conv2",    C=256,  G=1, M=256,  N=N, P=14,  Q=14,  R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET18_ITERS[2])))
    phases.append(("layer13_layer3_0_dn_samp",  lambda: test_conv_iter("layer13_layer3_0_dn_samp",  C=128,  G=1, M=256,  N=N, P=14,  Q=14,  R=1, S=1, HS=2, WS=2, datatype=float16, iters=RESNET18_ITERS[3])))
    phases.append(("layer14_layer3_1_conv1",    lambda: test_conv_iter("layer14_layer3_1_conv1",    C=256,  G=1, M=256,  N=N, P=14,  Q=14,  R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET18_ITERS[4])))
    phases.append(("layer15_layer3_1_conv2",    lambda: test_conv_iter("layer15_layer3_1_conv2",    C=256,  G=1, M=256,  N=N, P=14,  Q=14,  R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET18_ITERS[5])))
    phases.append(("layer16_layer4_0_conv1",    lambda: test_conv_iter("layer16_layer4_0_conv1",    C=256,  G=1, M=512,  N=N, P=7,   Q=7,   R=3, S=3, HS=2, WS=2, datatype=float16, iters=RESNET18_ITERS[6])))
    phases.append(("layer17_layer4_0_conv2",    lambda: test_conv_iter("layer17_layer4_0_conv2",    C=512,  G=1, M=512,  N=N, P=7,   Q=7,   R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET18_ITERS[7])))
    phases.append(("layer18_layer4_0_dn_samp",  lambda: test_conv_iter("layer18_layer4_0_dn_samp",  C=256,  G=1, M=512,  N=N, P=7,   Q=7,   R=1, S=1, HS=2, WS=2, datatype=float16, iters=RESNET18_ITERS[8])))
    phases.append(("layer19_layer4_1_conv1",    lambda: test_conv_iter("layer19_layer4_1_conv1",    C=512,  G=1, M=512,  N=N, P=7,   Q=7,   R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET18_ITERS[9])))
    phases.append(("layer20_layer4_1_conv2",    lambda: test_conv_iter("layer20_layer4_1_conv2",    C=512,  G=1, M=512,  N=N, P=7,   Q=7,   R=3, S=3, HS=1, WS=1, datatype=float16, iters=RESNET18_ITERS[10])))
    
    phases.append(("layer21_fc",                lambda: test_matmul_iter("layer21_fc",              M=1000, K=512, N=N,                                           datatype=float16, iters=RESNET18_ITERS[11])))
    
    name = f"ResNet18_N{N}"
    return name, phases

B = [1, 32]  # Batch Sizes
resnet18_pipelines = [resnet18_pipeline(n) for n in B]

pipeline_benchmark(output_dir="benchmarks/resnet18", pipelines=resnet18_pipelines, device_index=0)