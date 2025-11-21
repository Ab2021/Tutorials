# Day 19: Deployment - Interview Questions

> **Phase**: 2 - Computer Vision
> **Week**: 4 - Generative & Deployment
> **Topic**: Inference Optimization, Formats, and Serving

### 1. What is the difference between Tracing and Scripting in TorchScript?
**Answer:**
*   **Tracing**: Records operations by running an example input. Fails on data-dependent control flow.
*   **Scripting**: Compiles the Python source code. Supports control flow but requires type hints.

### 2. Why use ONNX?
**Answer:**
*   **Interoperability**: Train in PyTorch, deploy in C++ (ONNX Runtime), C# (Unity), or JavaScript (Web).
*   **Optimization**: Hardware vendors (Intel OpenVINO, NVIDIA TensorRT) optimize heavily for ONNX graph.

### 3. What is "Dynamic Batching"?
**Answer:**
*   A server-side technique.
*   Instead of processing requests one by one (Batch 1), the server waits a few milliseconds to collect multiple requests and forms a larger batch (Batch 8).
*   Increases Throughput significantly at slight Latency cost.

### 4. What is "TensorRT"?
**Answer:**
*   NVIDIA's inference optimizer.
*   Performs Layer Fusion (Conv+BN+ReLU), Kernel Auto-tuning, and Precision Calibration (FP16/INT8).
*   Delivers lowest latency on NVIDIA GPUs.

### 5. Explain "Quantization".
**Answer:**
*   Reducing the precision of weights/activations (e.g., FP32 $\to$ INT8).
*   **Benefits**: 4x smaller model, 2-4x faster compute (INT8 Tensor Cores), lower memory bandwidth.
*   **Risk**: Accuracy drop.

### 6. What is "Layer Fusion"?
**Answer:**
*   Merging multiple operations into a single CUDA kernel.
*   Example: `Conv -> Bias -> ReLU`.
*   Reduces memory access overhead (reading/writing intermediate results to VRAM).

### 7. What is "Model Pruning"?
**Answer:**
*   Removing unimportant weights (setting them to zero).
*   **Unstructured**: Random zeros. Hard to accelerate without sparse hardware.
*   **Structured**: Removing entire channels or filters. Directly speeds up dense compute.

### 8. What is "Triton Inference Server"?
**Answer:**
*   NVIDIA's production serving system.
*   Supports multiple backends (PyTorch, TensorRT, ONNX).
*   Handles dynamic batching, model versioning, and concurrent execution.

### 9. Why is Python bad for production inference?
**Answer:**
*   **GIL**: Limits concurrency.
*   **Overhead**: Interpreter is slow.
*   **Dependency Hell**: Managing Python environments on edge devices is painful.

### 10. What is "Calibration" in Quantization?
**Answer:**
*   Running a representative dataset through the model to determine the dynamic range (Min, Max) of activations.
*   Used to map FP32 values to INT8 integers.

### 11. What is "QAT" (Quantization Aware Training)?
**Answer:**
*   Simulating quantization errors during training.
*   Allows the network to adapt its weights to minimize the loss *given* the quantization constraints.
*   Yields higher accuracy than Post-Training Quantization.

### 12. How do you handle variable input sizes in ONNX?
**Answer:**
*   Specify `dynamic_axes` during export.
*   `dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'}}`.
*   Allows the runtime to accept any shape.

### 13. What is "FP16" vs "BF16"?
**Answer:**
*   **FP16**: Half precision. 5 bits exponent, 10 bits mantissa. Prone to overflow/underflow.
*   **BF16**: Brain Float. 8 bits exponent (same as FP32), 7 bits mantissa. Less precision, but same dynamic range as FP32. More stable.

### 14. What is "TorchServe"?
**Answer:**
*   Official PyTorch model serving framework.
*   Wraps PyTorch models in a REST API.
*   Good for simple deployments, but less optimized than Triton.

### 15. What is "OpenVINO"?
**Answer:**
*   Intel's toolkit for optimizing models on Intel CPUs, iGPUs, and VPUs.
*   Converts ONNX/PyTorch to Intermediate Representation (IR).

### 16. What is "Knowledge Distillation" for deployment?
**Answer:**
*   Training a small student model (MobileNet) to mimic a large teacher model (ResNet-152).
*   Deploy the student.

### 17. How do you profile a deployed model?
**Answer:**
*   Measure **Latency** (Time per request) and **Throughput** (Requests per second).
*   Use tools like `nsys` (Nsight Systems) to see GPU utilization.

### 18. What is "Eager Mode" vs "Graph Mode"?
**Answer:**
*   **Eager**: PyTorch default. Op-by-op execution. Debuggable. Slow.
*   **Graph**: TorchScript/TensorFlow Graph. Whole program optimization. Fast. Hard to debug.

### 19. What is "Zero-Copy" in inference?
**Answer:**
*   Passing pointers to data (e.g., from Camera buffer to GPU memory) without copying bytes.
*   Crucial for low-latency video processing.

### 20. Why might `torch.jit.trace` produce a wrong model?
**Answer:**
*   If the model logic depends on the *value* of the input (e.g., `if input.sum() > 0:`).
*   Tracing records the path taken by the *dummy input* and freezes it.
*   Future inputs will follow the same path regardless of their values.
