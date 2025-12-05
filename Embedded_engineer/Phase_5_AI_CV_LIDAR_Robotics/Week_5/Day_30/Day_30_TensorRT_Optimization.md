# Day 30: TensorRT & ONNX Runtime
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 5: Edge AI & Optimization

---

> **📝 Content Creator Instructions:**
> PyTorch is great for training (flexible). It is terrible for deployment (slow).
> - **Focus:** ONNX (Interoperability) and TensorRT (NVIDIA's High Performance Inference SDK).
> - **Code:** Converting a PyTorch model to TensorRT Engine and running inference.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Export** a PyTorch model to the ONNX standard format.
2.  **Explain** how TensorRT optimizes a graph (Layer Fusion, Kernel Auto-Tuning, Dynamic Tensor Memory).
3.  **Build** a TensorRT Engine (`.trt`) from an ONNX file.
4.  **Execute** inference using the TensorRT Python API.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU (Required for TensorRT).

### Software Environment
```bash
pip install torch onnx onnxruntime-gpu
pip install tensorrt pycuda
```

### Prior Knowledge
- Computation Graphs (Nodes and Edges).
- GPU Memory (VRAM).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: ONNX (Open Neural Network Exchange)

A universal format.
*   PyTorch -> ONNX -> TensorFlow? Yes.
*   PyTorch -> ONNX -> EdgeTPU? Yes.
*   **Structure:** A static graph of operators (`Conv`, `Relu`, `Add`).
*   **Limitation:** Dynamic Control Flow (if/else loops) is hard in ONNX. Loops must be unrolled.

### 🔹 Part 2: TensorRT Magic

TensorRT takes a static graph and optimizes it for the *specific GPU* you are running on.
1.  **Layer Fusion:** Merges layers. `Conv` + `Bias` + `ReLU` becomes a single CUDA kernel (`CBR`). Reduces memory bandwidth usage.
2.  **Kernel Auto-Tuning:** TensorRT benchmarks dozens of convolution algorithms (Winograd, GEMM, FFT) and picks the fastest one for *your* specific layer dimensions and *your* specific GPU.
3.  ** Precision Calibration:** Automatically casts weights to FP16 or INT8 if requested.

---

## 💻 Implementation: The Pipeline

We will take a ResNet-18 model -> ONNX -> TensorRT.

### 🛠️ Project Structure
```text
day30_tensorrt/
├── models/
│   └── resnet.onnx
├── src/
│   ├── export_onnx.py
│   ├── build_engine.py
│   └── infer_trt.py
└── run_pipeline.py
```

### 👨‍💻 Step 1: Export to ONNX (`src/export_onnx.py`)

```python
import torch
import torchvision

def export():
    model = torchvision.models.resnet18(pretrained=True).cuda().eval()
    
    # Dummy Input (needed to trace the graph)
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        "models/resnet.onnx",
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )
    print("Done.")
```

### 👨‍💻 Step 2: Build TensorRT Engine (`src/build_engine.py`)

```python
import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    # Config parameters
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
    
    # FP16 Mode (Half Precision) - 2x Speedup
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        
    # Build
    print("Building TensorRT Engine... (This may take a while)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        
    print(f"Engine saved to {engine_path}")
```

### 👨‍💻 Step 3: Inference (`src/infer_trt.py`)

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # Allocation
        self.h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()
        
    def infer(self, image_numpy):
        # Flatten image
        np.copyto(self.h_input, image_numpy.ravel())
        
        # Async Copy input to device
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # Execute
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        
        # Async Copy output to host
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        
        # Sync
        self.stream.synchronize()
        
        return self.h_output
```

---

## 🔬 Lab Exercise: Speed Test

### 1. Lab Objectives
- Measure FPS of PyTorch vs ONNX Runtime vs TensorRT.
- Input: Random 224x224 batches.
- **Results Typical:**
    - PyTorch: 6 ms
    - ONNX Runtime: 4 ms
    - TensorRT (FP16): 1.5 ms (**4x Speedup**)

---

## 🚀 Project: "Jetson Nano Deploy"

**Goal:** Run YOLOv8 on a Jetson Nano.
1.  **Problem:** PyTorch YOLOv8 runs at 3 FPS on Nano.
2.  **Solution:** Export to ONNX. Build TensorRT engine *on the Nano* (compilation takes time).
3.  **Result:** 25-30 FPS (Real-time).
4.  **Note:** Engines are *hardware specific*. You cannot copy a `.trt` file from your desktop 3090 to a Jetson. You must rebuild it on the target device.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Dimension Mismatch"
*   **Cause:** ONNX export had fixed batch size `1`, but you tried to run batch size `8`.
*   **Fix:** Use dynamic axes during ONNX export.
    ```python
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    ```

#### 2. "Unsupported Operator"
*   **Symptom:** TensorRT parser fails.
*   **Cause:** You used a weird PyTorch function (e.g., `torch.unique`) that has no TensorRT equivalent.
*   **Fix:** Implement the logic using basic ops, or write a **TensorRT Custom Plugin** (C++).

---

## ⚡ Optimization: Polygraphy

NVIDIA provides `polygraphy`.
*   A tool to debug TensorRT accuracy.
*   Compares the outputs of ONNX vs TensorRT layer-by-layer to find where precision loss occurred (e.g., if FP16 overflowed).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why can't I re-use a TensorRT engine on a different GPU?
    *   **A:** Auto-tuning picks kernels specific to the hardware (SM count, Cache size). A kernel optimal for RTX 3090 might be terrible (or invalid) for a Jetson Orin.
2.  **Q:** What is "Graph Folding"?
    *   **A:** Pre-calculating constant expressions. `x + (3 * 5)` becomes `x + 15` in the graph.
3.  **Q:** What is "FP16"?
    *   **A:** Half-precision float (16 bits: 1 sign, 5 exponent, 10 mantissa). Less range, less precision, but double the throughput on Tensor Cores.

### Challenge Task
> **Task:** Dynamic Shapes.
> 1. Configure TensorRT to handle input images from 224x224 up to 1024x1024.
> 2. Define "Optimization Profiles" (Min, Opt, Max dims).

---

## 📚 Further Reading
- **TensorRT Developer Guide:** NVIDIA Docs.
- **ONNX Model Zoo:** GitHub.

---

**Day 30 Complete**
