# Day 19: Deployment & Serving - Theory & Implementation

> **Phase**: 2 - Computer Vision
> **Week**: 4 - Generative & Deployment
> **Topic**: TorchScript, ONNX, and Production Inference

## 1. Theoretical Foundation: The Production Gap

Research code (Python/PyTorch) is slow and heavy.
Production environments (C++, Mobile, Edge) need:
1.  **No Python Dependency**: Python interpreter is overhead.
2.  **Optimization**: Fusion, Quantization, Memory planning.
3.  **Portability**: Run on NVIDIA GPU, Intel CPU, ARM, or FPGA.

### Solutions
1.  **TorchScript**: PyTorch's own intermediate representation (IR). Runs in C++ LibTorch.
2.  **ONNX (Open Neural Network Exchange)**: Universal format.
3.  **TensorRT**: NVIDIA's optimizer for GPUs.

## 2. TorchScript (Tracing vs Scripting)

### Tracing
Runs the model with dummy input and records operations.
*   *Pros*: Works with almost any PyTorch code.
*   *Cons*: Static graph. Fails with control flow (`if`, `for`) that depends on data.

### Scripting
Compiles Python source code into IR.
*   *Pros*: Supports control flow.
*   *Cons*: Requires type hints and subset of Python.

```python
import torch

# 1. Tracing
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model_traced.pt")

# 2. Scripting
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Load in C++
# auto module = torch::jit::load("model_traced.pt");
```

## 3. ONNX Export

The bridge to other frameworks.

```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}}, # Variable batch size
    opset_version=13
)
```

## 4. TensorRT (NVIDIA)

Optimizes ONNX graph for specific GPU.
1.  **Layer Fusion**: Combines Conv + Bias + ReLU into one kernel.
2.  **Precision Calibration**: FP32 $\to$ FP16/INT8.
3.  **Kernel Tuning**: Selects best CUDA kernel for the specific hardware (A100 vs T4).

Result: 2x - 5x speedup.
