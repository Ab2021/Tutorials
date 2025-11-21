# Day 19: Deployment - Deep Dive

> **Phase**: 2 - Computer Vision
> **Week**: 4 - Generative & Deployment
> **Topic**: Triton, Dynamic Batching, and Quantization

## 1. Triton Inference Server (NVIDIA)

Serving a model isn't just `model(x)`. You need:
*   **HTTP/gRPC API**: To accept requests.
*   **Dynamic Batching**: Aggregate concurrent requests into a batch to saturate GPU.
*   **Model Management**: Hot-swap models without downtime.
*   **Backend Support**: Run PyTorch, ONNX, TensorRT, and Python backends simultaneously.

**Config (`config.pbtxt`)**:
```protobuf
name: "resnet50"
platform: "onnxruntime_onnx"
max_batch_size: 8
dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 100
}
```

## 2. Post-Training Quantization (PTQ)

Converting FP32 weights to INT8.
*   **Weights**: Easy. Just round them.
*   **Activations**: Hard. Range depends on input.
*   **Calibration**: Run a small dataset through the model to measure activation ranges (Min/Max).

**Static vs Dynamic Quantization**:
*   **Dynamic**: Quantize weights to INT8. Quantize activations on the fly (overhead). Good for LSTM/BERT.
*   **Static**: Quantize weights and activations. Pre-compute ranges. Good for CNNs.

## 3. Quantization Aware Training (QAT)

PTQ drops accuracy. QAT fixes it.
*   Insert "Fake Quantize" nodes during training.
*   Simulates rounding errors during forward pass.
*   Model learns to be robust to quantization noise.
*   Result: INT8 model with FP32 accuracy.

## 4. TorchServe

PyTorch's default serving tool (Java-based).
*   Wraps model in a `.mar` (Model Archive).
*   Handles workers, logging, and metrics.
*   Easier than Triton for pure PyTorch, but less performant.
