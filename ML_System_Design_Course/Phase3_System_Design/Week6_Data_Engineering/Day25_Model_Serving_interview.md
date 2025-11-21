# Day 25: Model Serving - Interview Questions

> **Topic**: Inference at Scale
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What are the main challenges in Model Serving?
**Answer:**
*   Latency (SLA).
*   Throughput (QPS).
*   Cost (GPU/CPU).
*   Version Management.

### 2. Explain TorchServe / TensorFlow Serving.
**Answer:**
*   Specialized servers for ML models.
*   Features: Batching, Multi-model serving, Metrics, API endpoints (gRPC/REST).

### 3. What is "Dynamic Batching"?
**Answer:**
*   Server waits a few ms (e.g., 5ms) to collect multiple incoming requests.
*   Stacks them into a batch. Runs inference once.
*   Improves Throughput (GPU utilization) at slight Latency cost.

### 4. What is Quantization?
**Answer:**
*   Reducing precision of weights (FP32 -> INT8).
*   4x smaller model. Faster math.
*   Slight accuracy drop.

### 5. What is Pruning for inference?
**Answer:**
*   Removing zero/small weights.
*   Makes model sparse.
*   Requires specialized hardware/kernels to see speedup.

### 6. What is Knowledge Distillation?
**Answer:**
*   Train a small "Student" model to mimic a large "Teacher" model.
*   Student learns from Teacher's soft probabilities (Dark Knowledge).
*   Deploy Student.

### 7. Explain the Sidecar Pattern (K8s).
**Answer:**
*   Deploy model container alongside app container in same Pod.
*   Low latency communication (localhost).
*   Decouples app logic from model logic.

### 8. What is Triton Inference Server?
**Answer:**
*   NVIDIA's high-performance server.
*   Supports TensorRT, PyTorch, ONNX.
*   Optimized for GPU.

### 9. What is TensorRT?
**Answer:**
*   Optimizer and Runtime for NVIDIA GPUs.
*   Fuses layers, selects best kernels, calibrates for INT8.
*   Huge speedup.

### 10. How do you handle Model Versioning in production?
**Answer:**
*   Model Registry (MLflow).
*   Deploy `v2` to new endpoint or shadow `v1`.
*   Never overwrite `latest`.

### 11. What is A/B Testing for models?
**Answer:**
*   Route 50% traffic to Model A, 50% to B.
*   Compare business metrics.

### 12. What is Multi-Armed Bandit (MAB) for serving?
**Answer:**
*   Adaptive A/B testing.
*   Thompson Sampling / UCB.
*   Automatically routes more traffic to the winning model. Minimizes regret.

### 13. How do you scale serving? (K8s HPA).
**Answer:**
*   **Horizontal Pod Autoscaler**.
*   Scale replicas based on CPU/GPU utilization or Custom Metric (Queue Depth / Latency).

### 14. What is ONNX?
**Answer:**
*   Open Neural Network Exchange.
*   Standard format to represent models.
*   Train in PyTorch -> Export ONNX -> Run in ONNX Runtime (C++).

### 15. What is "Cold Start" in Serverless Inference (Lambda)?
**Answer:**
*   Time to spin up container and load model into memory.
*   Can be seconds. Bad for real-time.
*   **Fix**: Provisioned Concurrency.

### 16. How do you optimize Python for serving?
**Answer:**
*   Python is slow (GIL).
*   Use C++ backends (TorchScript).
*   Use AsyncIO.
*   Use specialized servers (FastAPI + Uvicorn).

### 17. What is Edge Deployment (TFLite)?
**Answer:**
*   Running model on device (Phone/IoT).
*   Privacy, Zero Latency, Offline.
*   Constraints: Battery, Memory, Compute.

### 18. What is Federated Learning?
**Answer:**
*   Train on device. Send gradients (not data) to server.
*   Aggregates updates.
*   Privacy-preserving.

### 19. How do you monitor inference latency?
**Answer:**
*   P99 and P95 latency (Tail latency).
*   Average is misleading.

### 20. What is Request/Response Logging?
**Answer:**
*   Log inputs and predictions for monitoring drift and debugging.
*   Sample (1%) if volume is too high.
