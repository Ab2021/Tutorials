# Day 25 (Part 1): Advanced Model Serving

> **Phase**: 6 - Deep Dive
> **Topic**: Inference at Scale
> **Focus**: Triton, TensorRT, and Microservices
> **Reading Time**: 60 mins

---

## 1. Triton Inference Server (NVIDIA)

Why use Triton over Flask?

### 1.1 Features
*   **Dynamic Batching**: Groups requests.
*   **Concurrent Model Execution**: Run ResNet and BERT on the same GPU simultaneously.
*   **Backend Support**: Runs PyTorch, TensorFlow, ONNX, TensorRT backends.

### 1.2 Ensemble Models
*   **DAG**: Define a pipeline (Preprocessing -> ResNet -> Postprocessing) entirely within Triton.
*   **Benefit**: No network overhead between steps. Data stays in GPU memory.

---

## 2. TensorRT Optimization

### 2.1 Layer Fusion
*   Combines `Conv + Bias + ReLU` into a single kernel. Reduces memory access.

### 2.2 Precision Calibration
*   Converts FP32 to INT8.
*   **Calibration**: Runs a sample dataset to determine the dynamic range of activations to minimize quantization error.

---

## 3. Tricky Interview Questions

### Q1: Why is Python (Flask/FastAPI) bad for high-throughput serving?
> **Answer**:
> 1.  **GIL**: Only one request processed at a time per process (CPU bound).
> 2.  **Overhead**: HTTP parsing, JSON serialization/deserialization in Python is slow.
> 3.  **Fix**: Use C++ servers (Triton/TF Serving) or run Python with `uvicorn` (Asynchronous) and many workers.

### Q2: Explain the "Sidecar" pattern in K8s serving.
> **Answer**:
> *   Main Container: Your Application (Business Logic).
> *   Sidecar Container: Model Server (Triton).
> *   **Comm**: App talks to Sidecar via localhost.
> *   **Pros**: Decouples App code from Model runtime. Can upgrade Model Server independently.

### Q3: How to handle "Hot Spots" (one model getting 90% traffic)?
> **Answer**:
> *   **Replica Scaling**: K8s HPA (Horizontal Pod Autoscaler) based on `custom_metric_qps`.
> *   **Model Routing**: Route specific models to specific hardware (e.g., Heavy models to A100, Light models to T4).

---

## 4. Practical Edge Case: Memory Fragmentation
*   **Problem**: Long running server OOMs despite low usage.
*   **Reason**: PyTorch caching allocator fragmentation.
*   **Fix**: `torch.cuda.empty_cache()` (Slow). Better: Set `max_split_size_mb` env var to reduce fragmentation.

