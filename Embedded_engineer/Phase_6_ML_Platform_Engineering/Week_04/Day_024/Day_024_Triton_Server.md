# Day 24: Triton Inference Server
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 4: TensorRT & Inference Optimization

---

> **🎯 Focus Area:** Master NVIDIA Triton Inference Server to deploy AI models as robust, scalable, and high-performance microservices with Dynamic Batching and Multi-Framework support.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Understand** the architecture of Triton (Model Repository, Scheduler, Backends).
2.  **Structure** a Model Repository for production deployment.
3.  **Configure** Dynamic Batching to maximize GPU throughput without sacrificing latency.
4.  **Deploy** models from different frameworks (PyTorch, TensorRT, ONNX) simultaneously.
5.  **Query** the inference server using gRPC and HTTP Python clients.
6.  **Analyze** performance using `perf_analyzer`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU
- Linux Environment (Docker is the standard distribution method for Triton).

### Software Environment
```bash
# Pull Triton Docker Image
docker pull nvcr.io/nvidia/tritonserver:23.10-py3

# Install Client Libraries
pip install tritonclient[all] gevent
```

### Prior Knowledge
- Day 22: Building TensorRT engines.
- Basic Docker usage.
- Networking concepts (HTTP vs gRPC).

---

## 📖 Theoretical Foundation

### 1. Scaling Inference

Running `python inference.py` works for demos. In production, you face:
*   **Concurrency:** Multiple users hitting the endpoint.
*   **Utilization:** Sending 1 image to an A100 is a waste. You want to send 32 at once.
*   **Diversity:** You might have a ResNet (TensorRT) and a BERT (ONNX) running on one GPU.

**Triton Inference Server** solves this. It manages memory, schedules kernels, and batches requests automatically.

### 2. Architecture

*   **Model Repository:** A filesystem directory organizing models and versions.
*   **Dynamic Batcher:** Collects individual requests within a time window (e.g., 5ms) and executes them as a single large batch interaction with the GPU.
*   **Backends:**
    *   `tensorrt_plan`: For TRT engines.
    *   `onnxruntime_onnx`: For ONNX files.
    *   `python`: For pre/post-processing logic.

### 3. The `config.pbtxt`

Every model has a Protocol Buffer configuration file defining:
*   Inputs/Outputs (Shape, Datatype).
*   Scheduling policies (Dynamic Batching settings).
*   Instance Groups (How many copies of the model to load).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Setting up the Model Repository

We will set up a repo for the TensorRT engine we built in Day 22.

**Directory Structure:**
```
model_repository/
  simple_resnet/
    config.pbtxt
    1/
      model.plan    <-- The TRT Engine from Day 22
```

#### 📁 `model_repository/simple_resnet/config.pbtxt`
```text
name: "simple_resnet"
platform: "tensorrt_plan"
max_batch_size: 16

# Input Configuration
# dims does NOT include batch size
input [
  {
    name: "input"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 32, 32 ]
  }
]

# Output Configuration
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }
]

# Dynamic Batching
# Combines requests to improve throughput
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 1000  # Wait up to 1ms to form a batch
}

# Instance Group
# Run 2 instances of this model on GPU 0
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

### 👨‍💻 Infrastructure: Running Docker

To start the server (assuming you are in the directory containing `model_repository`):

```bash
docker run --gpus=all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models --strict-model-config=false
```

*   **Port 8000:** HTTP
*   **Port 8001:** gRPC (Faster, binary)
*   **Port 8002:** Metrics (Prometheus)

### 👨‍💻 Client Implementation: Python gRPC

This client mimics a frontend service sending requests to the inference server.

#### 📁 `src/triton_client.py`
```python
#!/usr/bin/env python3
"""
Day 24: Triton gRPC Client
Phase 6: Platform Engineering
"""

import tritonclient.grpc as grpcclient
import numpy as np
import time

def run_inference():
    # 1. Connection
    url = "localhost:8001"
    try:
        triton_client = grpcclient.InferenceServerClient(url=url)
    except Exception as e:
        print(f"Failed to connect to Triton at {url}")
        return

    # Check server status
    if not triton_client.is_server_live():
        print("Server is NOT live")
        return

    model_name = "simple_resnet"
    
    # 2. Data Prep
    # Batch size 1 for the client request (Server will batch dynamically)
    batch_size = 1
    # Shape must match config (Batch, C, H, W)
    input_data = np.random.randn(batch_size, 3, 32, 32).astype(np.float32)

    # 3. Create Inputs/Outputs Config
    inputs = []
    # Input name matches config.pbtxt
    inputs.append(grpcclient.InferInput('input', input_data.shape, "FP32"))
    
    # Initialize data
    inputs[0].set_data_from_numpy(input_data)

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('output'))

    # 4. Inference Call
    print("Sending Request...")
    start = time.time()
    
    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs
    )
    
    end = time.time()
    
    # 5. Result Parsing
    output_data = results.as_numpy('output')
    print(f"Inference Time (Network included): {(end-start)*1000:.2f} ms")
    print(f"Output Shape: {output_data.shape}")
    print(f"Output Data (first 5): {output_data[0][:5]}")

    # Statistics
    stats = triton_client.get_inference_statistics(model_name=model_name)
    print("\nServer Stats:")
    print(stats)

if __name__ == "__main__":
    run_inference()
```

---

## 🔬 Lab Exercise: "Performance Analyzer"

### Lab Objectives
1.  Use `perf_analyzer` (shipped with Triton SDK) to stress test the model.
2.  Compare throughput with and without **Dynamic Batching**.

### Steps
1.  **Baseline (No Dynamic Batching):**
    Modify `config.pbtxt` to remove the `dynamic_batching { ... }` block. Restart server.
    ```bash
    perf_analyzer -m simple_resnet -u localhost:8000 --concurrency-range 1:8
    ```
    *Observe:* Latency might stay low, but Throughput (req/sec) won't scale perfectly linearly.

2.  **With Dynamic Batching:**
    Restore config. Restart server.
    Run `perf_analyzer` with high concurrency (simulating many users).
    *Observe:* Throughput increases significantly because Triton gathers 8 requests and runs 1 GPU Inference (Batch=8) instead of 8 small ones. Latency per request increases slightly (waiting for batch construction), but system efficiency boosts.

### Understanding Output
```text
Examples:
  Concurrency: 4, throughput: 450 infer/sec, latency: 8900 usec
```
This tells you that with 4 simultaneous clients, your server handles 450 FPS.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Decoupling:** Triton separates "Model Deployment" from "Application Logic." The App just sends JSON/Tensors; Triton handles GPU complexities.
2.  **Dynamic Batching:** The single most important feature for cost-saving. It turns an erratic stream of user requests into dense, GPU-friendly batches.
3.  **Standardization:** Using Triton means your infrastructure (K8s, Metrics, Logging) looks the same whether you deploy PyTorch, TensorFlow, or TRT.
4.  **Protocol Buffers:** `config.pbtxt` is the contract. If inputs don't match, Triton rejects the load, preventing runtime crashes.

### API Summary
```python
# Client
client = grpcclient.InferenceServerClient(url)
client.infer(model, inputs, outputs)

# Config (pbtxt)
dynamic_batching { max_queue_delay_microseconds: 100 }
instance_group [ { count: 2, kind: KIND_GPU } ]
```

---

**Day 24 Complete** ✅

*Next: Day 25 - Model Quantization & Compression - Techniques to make models smaller and faster before they reach Triton.*
