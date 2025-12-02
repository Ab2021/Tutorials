# Day 154: AI on the Edge (TensorRT & TFLite)
## Phase 3: Camera Systems & ISP | Week 25: Machine Learning for Camera Systems

---

## 🎯 Learning Objectives
1.  **Differentiate** between Training (Cloud/GPU) and Inference (Edge/Embedded).
2.  **Understand** the constraints of Edge AI: Power, Memory, Thermal, Latency.
3.  **Compare** Inference Engines: NVIDIA TensorRT vs TensorFlow Lite (TFLite) vs ONNX Runtime.
4.  **Convert** a pre-trained model (ResNet/MobileNet) to an optimized format.
5.  **Benchmark** performance (FPS, ms) on the target hardware.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** NVIDIA Jetson (for TensorRT) OR Raspberry Pi (for TFLite).
*   **Software:** PyTorch, ONNX, `trtexec`, `tflite_convert`.
*   **Model:** A standard ResNet-18 or MobileNetV2 trained on ImageNet.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Edge Constraint
*   **Cloud:** 1000W GPU, Unlimited RAM. Goal: Max Accuracy.
*   **Edge:** 5W-15W SoC, 4GB RAM. Goal: Real-time (30fps) with "Good Enough" Accuracy.
*   **Optimization:** We cannot run the training graph (PyTorch `backward()`). We only need the forward pass. We can fuse layers (Conv+BN+ReLU) and reduce precision (FP32 -> FP16 -> INT8).

### 🔹 Part 2: The Frameworks
1.  **TensorRT (NVIDIA):** The king of speed on Jetson/Orin. Uses CUDA and Tensor Cores. Proprietary.
2.  **TFLite (Google):** Runs on CPU, GPU (Delegate), and NPU (Coral/Hexagon). Open Source. Great for Android/Pi.
3.  **ONNX Runtime (Microsoft):** Cross-platform. Good middle ground.

### 🔹 Part 3: The Pipeline
1.  **Train:** PyTorch/TF -> `.pt` / `.h5`.
2.  **Export:** Convert to **ONNX** (Open Neural Network Exchange). The universal format.
3.  **Optimize:** Convert ONNX to **Engine** (TensorRT) or **FlatBuffer** (TFLite).
4.  **Deploy:** C++ or Python runtime loads the optimized file.

---

## 💻 Implementation Examples

### Example 1: PyTorch to ONNX

```python
import torch
import torchvision

# 1. Load Model
model = torchvision.models.resnet18(pretrained=True).cuda()
model.eval()

# 2. Create Dummy Input (Batch, Channels, Height, Width)
dummy_input = torch.randn(1, 3, 224, 224, device='cuda')

# 3. Export
torch.onnx.export(model, dummy_input, "resnet18.onnx",
                  verbose=True,
                  input_names=['input'],
                  output_names=['output'],
                  opset_version=11)
print("Exported to resnet18.onnx")
```

### Example 2: ONNX to TensorRT (CLI)

Using `trtexec` on the Jetson.

```bash
# FP16 Optimization
/usr/src/tensorrt/bin/trtexec \
  --onnx=resnet18.onnx \
  --saveEngine=resnet18_fp16.engine \
  --fp16 \
  --workspace=1024
```

### Example 3: Running Inference (Python TensorRT)

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Load Engine
logger = trt.Logger(trt.Logger.WARNING)
with open("resnet18_fp16.engine", "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# Create Context
context = engine.create_execution_context()

# Allocate Memory
input_shape = (1, 3, 224, 224)
input_size = trt.volume(input_shape) * 4 # FP32 bytes
d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(1000 * 4) # 1000 classes

# Execution
def infer(input_data):
    cuda.memcpy_htod(d_input, input_data)
    context.execute_v2(bindings=[int(d_input), int(d_output)])
    h_output = np.empty(1000, dtype=np.float32)
    cuda.memcpy_dtoh(h_output, d_output)
    return h_output
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: The "Framework Face-Off"

**Objective:** Compare FPS.

**Steps:**
1.  Run ResNet-18 in pure **PyTorch** on CPU. Measure FPS (e.g., 5 fps).
2.  Run ResNet-18 in **PyTorch** on GPU. Measure FPS (e.g., 40 fps).
3.  Run ResNet-18 in **TensorRT (FP16)**. Measure FPS (e.g., 200 fps).
4.  **Conclusion:** TensorRT is 5x faster than native PyTorch GPU.

### Lab 2: TFLite on Raspberry Pi

**Objective:** Low-power inference.

**Steps:**
1.  Convert model to `.tflite`.
2.  Run on Pi 4 CPU.
3.  Add a **Google Coral USB Accelerator** (Edge TPU).
4.  Compile model for Edge TPU (`edgetpu_compiler`).
5.  Run again.
6.  **Observation:** Speed jumps from 10fps to 60fps.

### Lab 3: Latency Jitter

**Objective:** Real-time stability.

**Steps:**
1.  Run inference in a loop.
2.  Log the time for each frame.
3.  **Plot:** Histogram of latencies.
4.  **Analysis:** Is there a "Long Tail"? (Occasional frames taking 100ms due to garbage collection or thermal throttling).

---

## 🐛 Debugging Edge AI

### Debug 1: "Result is Garbage"

**Symptom:** The model predicts "Toaster" for a "Cat".

**Cause:**
*   **Preprocessing Mismatch:** Did you normalize the image?
    *   PyTorch expects `mean=[0.485, ...], std=[0.229, ...]`.
    *   TensorRT expects raw pixels? Or 0-1?
    *   **Fix:** Ensure the preprocessing (Resize, Crop, Normalize) is *identical* to training.

### Debug 2: "Engine Creation Failed"

**Symptom:** `trtexec` crashes.

**Cause:**
*   Unsupported ONNX operator.
*   **Fix:** Upgrade TensorRT. Or implement the missing layer as a "Custom Plugin" in C++.

---

## ⚡ Performance Optimization

### Optimization 1: Batching

*   If you have 4 cameras, don't run inference 4 times with batch=1.
*   Stack the images into a tensor of shape `(4, 3, 224, 224)` and run once with batch=4.
*   GPU utilization improves significantly.

### Optimization 2: Asynchronous Copy

*   While the GPU is crunching the numbers for Frame N, the CPU should be preparing Frame N+1 and copying it to pinned memory.
*   Use `cudaStream` to overlap Compute and Memory Transfer.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "Quantization"?** (Reducing the number of bits to represent weights/activations, e.g., 32-bit float to 8-bit integer).
2.  **Why is FP16 faster than FP32?** (Less memory bandwidth, and Tensor Cores can do $A \times B + C$ twice as fast).
3.  **What is an "Engine" file?** (A compiled binary specific to the exact GPU architecture it was built on. Not portable).

### Practical Challenges

1.  **Write a Dockerfile:** Create a container with PyTorch, ONNX, and TensorRT installed.
2.  **Profile Memory:** Check how much VRAM the model uses. Can you fit two models?

---

## 📚 Further Reading & Resources

### Documentation
*   **NVIDIA TensorRT Developer Guide.**
*   **ONNX Model Zoo.**

---

## 🎓 Summary

Today we covered:
- ✅ **Edge vs Cloud:** Constraints.
- ✅ **TensorRT:** The speed demon.
- ✅ **ONNX:** The bridge.
- ✅ **FP16:** Free speedup.
- ✅ **Preprocessing:** The common trap.

**Next:** Day 155 - Object Detection (YOLO).

---

**Day 154 Complete** | Phase 3: Camera Systems & ISP | Week 25: Machine Learning for Camera Systems


