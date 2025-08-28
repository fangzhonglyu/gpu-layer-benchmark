# Per Layer GPU Energy & Latency Benchmark Tool

Mini GPU benchmarking helper for ML that measures per-layer latency and energy. It uses CUDA Graphs to minimize launch overhead and NVML total energy counters for power/energy.

- Included model pipelines: ResNet18/50, MobileNetV3-Small, EfficientNet-B0, RepLKNet-31B, OPT-66B (prefill & decode), ViT
- Profiles small kernels by unrolling and capturing them in CUDA graphs to get amortized energy and latency metrics over many iterations with minimal kernel launch overhead.
- Measures: average latency in ms, average power in W, per-layer avg energy in J, pipeline latency + energy and pipeline energy assuming full pipeline parallelism.

## Requirements

- NVIDIA GPU + with properly installed driver
- Pytorch with CUDA 12.8
- Python 3.9

## Custom Kernels

`test_kernel_iter(name, setup_func, capture_func, iters)` from `kernels.py` is a functional wrapper that supports arbitrary kernels. 
Custom kernels can be defined by providing custom `setup_func` and `capture_func` functions.

- Allocate input and output tensors in `setup_func()` on CUDA
- Execute kernel in `capture_func()`

Example: elementwise Add + ReLU kernel

```python
import torch
from kernels import test_kernel_iter

def test_add_relu_kernel(name="add_relu", M=4096, N=4096, dtype=torch.float16, iters=20000):
    def setup():
        a = torch.randn(M, N, dtype=dtype, device="cuda")
        b = torch.randn(M, N, dtype=dtype, device="cuda")
        out = torch.empty(M, N, dtype=dtype, device="cuda")
        return a, b, out, tmp

    def capture(state):
        a, b, out = state
        torch.add(a, b, out=out)         # no new allocation
        torch.relu(out, inplace=True)    
        return out # return is ignored

    return test_kernel_iter(name, setup, capture, iters)
```

Notes:
- `test_kernel_iter` internally captures `UNROLL_FACTOR` (100) steps per replay; `iters` will be rounded down to a multiple of 100.
- To include idle-energy accounting in pipeline totals, use pipelines or call `profile_device_idle_power()` before running.

## Custom Pipelines

Create a pipeline by composing phases and calling `pipeline_benchmark`:

```python
from typing import List, Tuple, Callable
from torch import float16
from kernels import test_conv_iter, test_matmul_iter
from pipeline_benchmark import pipeline_benchmark

MY_ITERS = [10000, 20000]

def my_pipeline(N:int) -> Tuple[str, List[Tuple[str, Callable]]]:
    phases = []
    phases.append(("conv3x3", lambda: test_conv_iter("conv3x3", C=64, G=1, M=64, N=N, P=56, Q=56, R=3, S=3, HS=1, WS=1, datatype=float16, iters=MY_ITERS[0])))
    phases.append(("fc",      lambda: test_matmul_iter("fc", M=1000, K=2048, N=N, datatype=float16, iters=MY_ITERS[1])))
    name = f"MyPipeline_N{N}"
    return name, phases

pipeline_benchmark(output_dir="benchmarks/my_pipeline", pipelines=[my_pipeline(1)], device_index=0)
```

- `pipeline_benchmark()` takes a list of pipelines, making it easier to benchmark models with different parameters (e.g. batch size, data type, etc)
- Each pipeline contains a list of phases (`List[Tuple[str,Callable]]`), each of which tests a specific layer in isolation.
- A run log is generated for each pipeline, and all pipeline results are gathered in `summary.csv`
