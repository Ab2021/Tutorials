# Day 25: Model Serving & Deployment

> **Phase**: 3 - System Design
> **Week**: 6 - Data Engineering
> **Focus**: Putting Models in Production
> **Reading Time**: 50 mins

---

## 1. Serving Patterns

### 1.1 Real-time (Online)
*   **REST API (FastAPI/Flask)**: Standard. JSON payload. Easy to debug. High overhead (text parsing).
*   **gRPC (Protobuf)**: Binary protocol. Low latency. Strongly typed. Best for internal microservices.
*   **Tools**: TorchServe, TensorFlow Serving, Triton Inference Server (NVIDIA).

### 1.2 Batch Serving
*   **Scenario**: Churn prediction, Weekly recommendations.
*   **Architecture**: Airflow job reads DB -> Runs Inference -> Writes to DB.
*   **Pros**: High throughput. No latency constraints.

### 1.3 Asynchronous (Queue-Based)
*   **Scenario**: Image generation, Video processing.
*   **Architecture**: User Request -> Kafka/SQS -> Worker GPU -> S3 -> Notify User.
*   **Pros**: Decouples user from processing. Handles bursts.

---

## 2. Advanced Deployment Strategies

### 2.1 Canary Deployment
*   Roll out new model to 1% of traffic. Monitor errors/latency. Gradually increase to 10%, 50%, 100%.
*   **Safety**: If it crashes, only 1% of users are affected.

### 2.2 Shadow Deployment
*   Run new model alongside old model.
*   **Serve**: Old model's prediction to user.
*   **Log**: New model's prediction.
*   **Compare**: Analyze performance offline without risking user experience.

### 2.3 A/B Testing
*   Split traffic 50/50. Compare business metrics (CTR, Conversion).

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Dependency Hell (Again)
**Scenario**: Model requires `pytorch==1.9`. Server has `pytorch==1.10`. Crash.
**Solution**:
*   **Containerization (Docker)**: Bake the environment into the image.
*   **Model Formats**: ONNX (Open Neural Network Exchange). Convert PyTorch/TF models to a standard graph format that runs on a lightweight runtime (ONNX Runtime).

### Challenge 2: CPU vs GPU Inference
**Scenario**: GPU is expensive. CPU is slow.
**Solution**:
*   **Batching**: On GPU, processing 1 item takes 10ms. Processing 32 items takes 12ms. Dynamic Batching (in Triton/Ray) waits 5ms to collect requests and sends them as a batch. Huge throughput gain.
*   **Quantization**: Convert weights to INT8. 4x smaller, 2-3x faster on CPU.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: Why use gRPC over REST for model serving?**
> **Answer**:
> 1.  **Performance**: gRPC uses Protobuf (binary), which is smaller and faster to serialize/deserialize than JSON.
> 2.  **HTTP/2**: Allows multiplexing (multiple requests over one connection).
> 3.  **Streaming**: Native support for streaming responses (e.g., LLM token generation).

**Q2: What is "Dynamic Batching"?**
> **Answer**: A serving optimization. Instead of processing requests immediately, the server waits a tiny window (e.g., 5ms) to group incoming requests into a single tensor batch. This leverages the GPU's parallelism, significantly increasing throughput with minimal impact on latency.

**Q3: Explain the difference between Canary and Shadow deployment.**
> **Answer**:
> *   **Canary**: The new model serves *real* traffic to a subset of users. Bad model = Bad user experience for that subset.
> *   **Shadow**: The new model processes traffic but its output is *ignored* (logged only). Zero risk to user.

---

## 5. Further Reading
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)
- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html)
