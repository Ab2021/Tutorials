# Day 9: Debugging - Deep Dive

> **Phase**: 1 - Foundations
> **Week**: 2 - Data & Workflow
> **Topic**: PyTorch Profiler, Anomaly Detection, and CUDA Sync

## 1. PyTorch Profiler (Kineto)

The profiler gives a timeline trace (Chrome Tracing format).
It reveals:
*   **Kernel Launch Overhead**: Too many small ops?
*   **Memory Transfers**: Too much H2D/D2H copy?
*   **DataLoader Wait**: GPU waiting for CPU?

**Chrome Trace**:
`prof.export_chrome_trace("trace.json")`. Open `chrome://tracing` and load it.
You see blocks for CPU execution and GPU kernels. Gaps indicate idle time.

## 2. CUDA Synchronization

GPU execution is asynchronous.
`t1 = time.time(); model(x); t2 = time.time()` measures nothing!
The CPU launches kernels and returns immediately.
**Correct Timing**:
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
model(x)
end.record()

torch.cuda.synchronize() # Wait for GPU
print(start.elapsed_time(end))
```

## 3. Anomaly Detection

`torch.autograd.detect_anomaly()`
*   Adds runtime checks to every operation.
*   If a NaN is produced, it halts and prints the **Stack Trace** of the forward operation that caused it.
*   *Cost*: Extremely slow. Use only for debugging.

## 4. Overfitting on One Batch

The ultimate sanity check.
Take a single batch of data. Train on it for 1000 epochs.
*   **Loss should go to 0**.
*   **Accuracy should go to 100%**.
If not, your model code or optimizer is broken.
If yes, your model is capable of learning; the issue is likely generalization or data.

## 5. Gradient Checking

Verifying your custom Autograd function.
`torch.autograd.gradcheck` compares your analytical gradient (backward) with numerical gradient (finite difference).

```python
from torch.autograd import gradcheck
input = (torch.randn(20, 20, dtype=torch.double, requires_grad=True),)
test = gradcheck(model, input, eps=1e-6, atol=1e-4)
```
