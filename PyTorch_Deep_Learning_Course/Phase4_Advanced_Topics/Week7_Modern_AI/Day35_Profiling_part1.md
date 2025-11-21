# Day 35: Profiling - Deep Dive

> **Phase**: 4 - Advanced Topics
> **Week**: 7 - Modern AI
> **Topic**: Channels Last, Quantization, and Pruning

## 1. Channels Last (NHWC)

PyTorch default: NCHW (Batch, Channel, Height, Width).
Tensor Cores prefer: NHWC (Batch, Height, Width, Channel).
*   Why? Dot product is along Channel dimension. Contiguous memory access.
*   Speedup: 20-30% on ResNet50.

```python
model = model.to(memory_format=torch.channels_last)
input = input.to(memory_format=torch.channels_last)
```

## 2. Post-Training Quantization (PTQ)

Convert FP32 weights to INT8 *after* training.
*   **Dynamic Quantization**: Weights INT8, Activations FP32 (quantized on fly). Good for LSTM/BERT.
*   **Static Quantization**: Weights & Activations INT8. Requires calibration. Good for CNNs.

## 3. Pruning

Removing unimportant weights (set to 0).
*   **Unstructured**: Random zeros. Requires sparse hardware support.
*   **Structured**: Remove whole channels/filters. Real speedup.

## 4. Roofline Model

Visualizing performance.
*   X-axis: Arithmetic Intensity (FLOPS / Byte).
*   Y-axis: Performance (GFLOPS).
*   Helps identify if you are Compute Bound or Memory Bound.

## 5. Checkpoint Activation (Gradient Checkpointing)

Trade Compute for Memory.
Instead of storing activations for backward pass, recompute them.
Allows fitting 4x larger batch size.
`torch.utils.checkpoint.checkpoint(module, input)`.
