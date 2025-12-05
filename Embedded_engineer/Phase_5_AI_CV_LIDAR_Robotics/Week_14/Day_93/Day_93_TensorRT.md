# Day 93: TensorRT Optimization
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 14: Edge AI Deployment

---

> **📝 Content Creator Instructions:**
> PyTorch is for training. TensorRT is for deployment.
> - **Focus:** The TensorRT Pipeline (Network Definition -> Builder -> Engine), FP16/INT8 Quantization, and Profiling.
> - **Code:** A complete pipeline that converts a `torchvision` ResNet model to a TensorRT engine and runs inference 10x faster.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** Graph Optimization: Layer Fusion (Conv+Bias+Relu $\to$ CBR), Kernel Auto-Tuning.
2.  **Convert** models using `torch-tensorrt` or `trtexec` (ONNX route).
3.  **Implement** INT8 Calibration (Entropy Calibrator) to minimize accuracy loss.
4.  **Serialize** and **Deserialize** Engines to disk (`.trt` files).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU (Jetson or Desktop).

### Software Environment
```bash
pip install torch torchvision
pip install tensorrt pycuda
```

### Prior Knowledge
- Neural Networks (CNNs).
- Floating Point Rep (FP32 vs FP16).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why TensorRT?

*   **Training Frameworks (PyTorch):** Dynamic Graphs, flexible, overhead for debugging.
*   **Inference Engines (TensorRT):** Static Graphs, compiled.
    *   **Layer Fusion:** Combines vertical (Conv+BN+ReLU) and horizontal layers.
    *   **Kernel Selection:** benchmark 100s of implementations of `Conv2d` on *your specific GPU* and picks the fastest one.
    *   **Memory Optimization:** Reuses memory for tensors that aren't alive simultaneously.

### 🔹 Part 2: Quantization (FP16/INT8)

*   **FP32 (32-bit float):** Standard. High precision.
*   **FP16 (16-bit float):** Tensor Cores use this. 2x speedup, 0.5x memory. Usually "free" accuracy-wise.
*   **INT8 (8-bit integer):** 4x speedup. Requires **Calibration**.
    *   Since Int8 range is [-128, 127], we must scale the weights and activations.
    *   **Calibration:** Run a few batches of real data to find the dynamic range (min/max) of activations to scale correctly.

### 🔹 Part 3: The Workflow

1.  **Export:** PyTorch $\to$ ONNX.
2.  **Build:** ONNX $\to$ TRT Engine (Optimization Phase).
3.  **Runtime:** Load Engine $\to$ Context $\to$ Execute.

---

## 💻 Implementation: PyTorch to TensorRT (via ONNX)

We will manually build the engine using the Python API for maximum control.

### 🛠️ Project Structure
```text
day93_tensorrt/
├── src/
│   ├── export_onnx.py
│   ├── build_engine.py
│   └── infer_trt.py
└── models/
    ├── resnet.onnx
    └── resnet_fp16.engine
```

### 👨‍💻 Step 1: Export ONNX (`src/export_onnx.py`)

```python
import torch
import torchvision.models as models

def export():
    # 1. Load Model
    model = models.resnet18(pretrained=True).eval().cuda()
    
    # 2. Dummy Input (for tracing)
    dummy = torch.randn(1, 3, 224, 224).cuda()
    
    # 3. Export
    torch.onnx.export(
        model, 
        dummy, 
        "models/resnet.onnx",
        input_names=["input"], 
        output_names=["output"],
        opset_version=13
    )
    print("Exported to models/resnet.onnx")

if __name__ == "__main__":
    export()
```

### 👨‍💻 Step 2: Build Engine (`src/build_engine.py`)

Using `tensorrt` python api.

```python
import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 1. Parse ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    # 2. Config (FP16)
    config.set_flag(trt.BuilderFlag.FP16)
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
    
    # 3. Build Serialized Engine
    serialized_engine = builder.build_serialized_network(network, config)
    
    # 4. Save
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Saved Engine to {engine_path}")

if __name__ == "__main__":
    build_engine("models/resnet.onnx", "models/resnet_fp16.engine")
```

### 👨‍💻 Step 3: Inference (`src/infer_trt.py`)

Using `pycuda` for memory management.

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.input_shape = (1, 3, 224, 224)
        self.output_shape = (1, 1000)
        
        # Host (CPU) memory
        self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)
        
        # Device (GPU) memory
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        
        # Stream
        self.stream = cuda.Stream()

    def infer(self, image):
        # 1. Preprocess
        # Resize to 224x224, Normalize... (omitted for brevity)
        self.h_input[:] = image.flatten()
        
        # 2. Transfer Input Host -> Device
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # 3. Execute
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        
        # 4. Transfer Output Device -> Host
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        
        # 5. Sync
        self.stream.synchronize()
        
        return self.h_output

# Run loop
# Compare fps vs PyTorch
```

---

## 🔬 Lab Exercise: "The Speedster"

### 1. Lab Objectives
- Measure Inference time of PyTorch ResNet (`~15ms` on Jetson Nano for example).
- Measure TensorRT FP16 Engine (`~3ms`).
- **Reduction:** 5x Speedup.
- **Task:** Try `trtexec` command line:
    ```bash
    trtexec --onnx=resnet.onnx --saveEngine=resnet.trt --fp16
    ```
- **Check Accuracy:** Ensure Top-1 prediction hasn't changed.

---

## 🚀 Project: "YOLO-TRT"

**Goal:** Real-time Object Detection.
1.  **Download:** `yolov8n.pt`.
2.  **Export:** Use Ultralytics export tool: `yolo export model=yolov8n.pt format=engine device=0 halve=True`.
3.  **Deploy:** Run the generated `.engine` file in a customized `DeepStream` pipeline or Python wrapper.
4.  **Result:** 30FPS+ detection on Jetson.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "OOM during Build"
*   **Symptom:** Process killed.
*   **Cause:** Building the engine explores thousands of kernels. It uses lots of RAM.
*   **Fix:** Build on a strong Desktop GPU, *but make sure to target the Jetson GPU architecture* (Cross-compilation is hard). Better: Create SWAP file on Jetson or use `trtexec --memPoolSize=...`.

#### 2. "Dimension Mismatch"
*   **Cause:** Dynamic Batch size in PyTorch vs Static in TRT.
*   **Fix:** Use `OptimizationProfile` in TRT builder to allow dynamic shapes, or enforce fixed batch size.

---

## ⚡ Optimization: Graph Surgeon

Sometimes the ONNX export is messy (extra nodes).
*   **ONNX Graph Surgeon:** A tool to explicitly modify the ONNX graph before building.
*   Can fuse custom plugins or remove debug nodes.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can I run a TensorRT engine built on RTX 4090 on a Jetson Orin?
    *   **A:** **NO.** Engines are hardware-specific. You must build the engine on the target device (or exact same GPU).
2.  **Q:** What is "Calibration"?
    *   **A:** Determining the scale factor for INT8 conversion. Finding the range of values so we don't clip important data.
3.  **Q:** Why is FP16 faster?
    *   **A:** Data is half size (bandwidth). Math ops are 2x faster (Tensor Cores).

### Challenge Task
> **Task:** Dynamic Shapes.
> 1. Allow input image to be any size between 224x224 and 640x640.
> 2. Use `IOptimizationProfile`.
> 3. Set Min, Opt, Max dimensions.
> 4. Performance will vary based on input size.

---

## 📚 Further Reading
- **NVIDIA TensorRT Stats:** "Developer Guide".
- **torch2trt:** Simplest wrapper for PyTorch users.

---

**Day 93 Complete**
