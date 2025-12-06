# Day 28: Week 4 Review & The High-Performance Inference Microservice
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 4: TensorRT & Inference Optimization

---

> **🎯 Focus Area:** Consolidate mastery of TensorRT, Triton, and Profiling by building an **End-to-End Inference Microservice** for Semantic Segmentation, achieving <10ms latency.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Synthesize** the full platform engineering workflow: Train -> Export -> Optimize -> Serve -> Monitor.
2.  **Build** an INT8 TensorRT engine for a Segmentation model (U-Net).
3.  **Deploy** the engine to Triton Inference Server with Dynamic Batching.
4.  **Validate** performance gains using `perf_analyzer` and `nsys`.
5.  **Review** key concepts from Week 4 (TRT Architecture, Quantization, DALI, Profiling).

---

## 📚 Week 4 Recap

### Topics Covered

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 22 | TensorRT Architecture | Builder, Network Config, ONNX Parser, Layer Fusion. |
| 23 | Advanced TRT | Dynamic Shapes, INT8 Calibration (PTQ), Plugins. |
| 24 | Triton Server | Model Repository, Dynamic Batching, Concurrent Execution. |
| 25 | Compression | Quantization Aware Training (QAT), Structured Pruning. |
| 26 | DALI | GPU Data Loading, Pipeline Graphs, `DALIGenericIterator`. |
| 27 | Profiling | Nsight Systems (OS Timeline), Nsight Compute (Kernel/Roofline). |

### The "Inference Optimization" Stack
1.  **Data:** DALI (GPU Decode).
2.  **Model:** TensorRT (INT8 Fusion).
3.  **Server:** Triton (Dynamic Batching).
4.  **Monitor:** Prometheus (Metrics) + Nsight (Deep Dive).

---

## 🏗️ Week 4 Project: Real-Time Segmentation Service

### Project Overview
We will deploy a **U-Net** semantic segmentation model. Segmentation is computationally heavy (Outputs a class per pixel), making it a perfect candidate for TensorRT optimization.

**Workflow:**
1.  **Pytorch:** export `unet.onnx`.
2.  **Polygraphy:** Inspect and sanitize.
3.  **TensorRT:** Build `unet_int8.plan` with calibration.
4.  **Triton:** Serve on port 8001.
5.  **Client:** Simulate video stream ingestion.

### Architecture
```
┌────────────────────────────────────────────────────────┐
│               INFERENCE SERVER (Docker)                │
│                                                        │
│  [Dynamic Batcher]                                     │
│         │ (Aggregates reqs)                            │
│         ▼                                              │
│  [TRT Execution Context]                               │
│         │                                              │
│    (UNet INT8 Engine)                                  │
│         │                                              │
│         ▼                                              │
│      [GPU Memory]                                      │
└────────────────────────────────────────────────────────┘
          ▲ gRPC
          │
      [Client App]
```

---

## 💻 Complete Project Implementation

### Step 1: Export ONNX (Host)

#### 📁 `project/1_export_model.py`
```python
import torch
import torch.hub

def export_unet():
    # Load a pretrained segmentation model (using simple unet or similar from methods)
    # Using a hub model for standard reproducibility
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model.cuda().eval()
    
    # Dummy Input (Batch 1, 3 channels, 256, 256)
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    
    print("Exporting U-Net to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        "unet.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13
    )
    print("Export Complete.")

if __name__ == "__main__":
    export_unet()
```

### Step 2: Build INT8 Engine (Host)

#### 📁 `project/2_build_engine.py`
```python
import tensorrt as trt
import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Dummy Calibrator from Day 23 (Simplified for brevity)
class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.d_input = cuda.mem_alloc(1 * 3 * 256 * 256 * 4) # Batch 1 buffer
        self.idx = 0
        
    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.idx < 10: # Calibrate on 10 random batches
            self.idx += 1
            data = np.random.randn(1, 3, 256, 256).astype(np.float32)
            cuda.memcpy_htod(self.d_input, data)
            return [int(self.d_input)]
        return None

    def read_calibration_cache(self):
        return None
        
    def write_calibration_cache(self, cache):
        pass

def build_engine():
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open("unet.onnx", "rb") as model:
        parser.parse(model.read())
        
    # Optimization Profile for Dynamic Batching
    profile = builder.create_optimization_profile()
    # Min=1, Opt=8, Max=16
    profile.set_shape("input", (1,3,256,256), (8,3,256,256), (16,3,256,256))
    config.add_optimization_profile(profile)
    
    # Enable INT8
    if builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = EntropyCalibrator("calib.cache")
        print("INT8 Enabled")
        
    engine = builder.build_serialized_network(network, config)
    
    os.makedirs("model_repository/unet/1", exist_ok=True)
    with open("model_repository/unet/1/model.plan", "wb") as f:
        f.write(engine)
    print("Engine saved to model_repository/unet/1/model.plan")

if __name__ == "__main__":
    build_engine()
```

### Step 3: Triton Config

#### 📁 `project/model_repository/unet/config.pbtxt`
```text
name: "unet"
platform: "tensorrt_plan"
max_batch_size: 16

input [
  {
    name: "input"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 256, 256 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1, 256, 256 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 500
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

### Step 4: Launch Server
*(Instructions only, script cannot execute docker)*
```bash
docker run --gpus=all --rm -p 8000:8000 -p 8001:8001 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models
```

### Step 5: Benchmark Client

#### 📁 `project/3_benchmark.py`
```python
import tritonclient.grpc as grpcclient
import numpy as np
import time
import threading

def client_thread(thread_id):
    client = grpcclient.InferenceServerClient(url="localhost:8001")
    input_data = np.random.randn(1, 3, 256, 256).astype(np.float32)
    inputs = [grpcclient.InferInput('input', input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)
    outputs = [grpcclient.InferRequestedOutput('output')]
    
    start = time.time()
    for i in range(50): # Send 50 requests
        client.infer("unet", inputs, outputs)
    dur = time.time() - start
    print(f"Thread {thread_id}: 50 reqs in {dur:.2f}s (fps={50/dur:.1f})")

def main():
    print("Starting Multi-Threaded Stress Test...")
    threads = []
    # Simulate 8 concurrent users
    for i in range(8):
        t = threading.Thread(target=client_thread, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    print("Test Complete. Check 'dstat' or 'nvtop' for GPU Load.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Profiling Audit

### Task
You have the system running. Now, prove it works efficiently.

1.  **Capture Trace:**
    ```bash
    nsys profile --trace=cuda,osrt --output=triton_test python 3_benchmark.py
    ```
    *Note: Normally you attach nsys to the SERVER process, but profiling the client gives latency info. To profile server, run the docker container with `nsys launch ... tritonserver ...`.*

2.  **Analyze Dynamic Batching:**
    *   Look at the Triton Server timeline.
    *   Do you see `Execute` calls with `batch=1` or `batch=4/8`?
    *   If you see many `batch=1`, increase `max_queue_delay_microseconds` in `config.pbtxt`.

3.  **Analyze INT8 Kernel:**
    *   Open `ncu` on the server.
    *   Check if kernels have names like `hcgemm` (Half Precision) or `imma` (Integer Matrix Multiply Accumulate). `imma` confirms INT8 usage.

---

## 📝 Week 4 Summary

You have moved from "Writing Kernels" (Week 1-2) to "Platform Engineering" (Week 4).
*   **The Kernel Engineer** cares about `threadIdx.x` and Shared Memory banks.
*   **The Platform Engineer** cares about **Throughput (QPS)**, **Latency (ms)**, **Cost ($/hour)**, and **Scalability**.

Techniques like **TensorRT (INT8)** and **dynamic batching** are the primary tools to reduce cloud costs by 4x-10x.

---

**Week 4 Complete** ✅

*Next Week: Week 5 - The Deep Learning Compiler Stack (TVM, MLIR) - What happens below TensorRT?*
