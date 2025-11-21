# Day 35: Profiling & Optimization - Theory & Implementation

> **Phase**: 4 - Advanced Topics
> **Week**: 7 - Modern AI
> **Topic**: PyTorch Profiler, Nsight Systems, and Bottleneck Analysis

## 1. Theoretical Foundation: The Three Bottlenecks

1.  **Compute Bound**: GPU is 100% utilized. Math is the limit. (Good).
2.  **Memory Bandwidth Bound**: GPU is waiting for data from HBM. (Optimize with Fusion/Triton).
3.  **Overhead Bound (CPU Bound)**: GPU is idle waiting for CPU to launch kernels or load data. (Optimize Dataloader/Python).

## 2. PyTorch Profiler

Built-in tool to visualize execution trace.

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = MyModel().cuda()
inputs = torch.randn(1, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
```

*   **Chrome Trace**: Open `chrome://tracing` and load `trace.json`.
*   See timeline of CPU ops vs GPU kernels.

## 3. Optimization Techniques

1.  **Dataloader**: `num_workers > 0`, `pin_memory=True`.
2.  **Mixed Precision (AMP)**: `torch.autocast`. 2x speedup.
3.  **Channels Last**: `memory_format=torch.channels_last`. Optimized for Tensor Cores (NHWC).
4.  **Graph Mode**: `torch.compile`.

## 4. NVIDIA Nsight Systems

System-wide profiler.
Shows OS scheduling, PCIe transfers, and CUDA kernels.
Command: `nsys profile python train.py`.
Best for diagnosing Dataloader/CPU bottlenecks.
