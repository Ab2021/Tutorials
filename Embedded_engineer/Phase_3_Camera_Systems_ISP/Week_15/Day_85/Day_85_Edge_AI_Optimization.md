# Day 85: Advanced Capstone - Edge AI Optimization (TensorRT)
## Phase 3: Camera Systems & ISP | Week 15: Final Capstone & Career

---

## 🎯 Learning Objectives
1.  **Understand** the TensorRT Optimization Pipeline: Layer Fusion, Kernel Auto-Tuning, Quantization.
2.  **Convert** PyTorch/ONNX models to TensorRT Engines (`.plan` / `.engine`).
3.  **Implement** FP16 and INT8 Quantization for 2x-4x speedup.
4.  **Configure** DeepStream for maximum throughput (Batching, Zero-Copy).
5.  **Profile** the AI pipeline using NVIDIA Nsight Systems.
6.  **Deploy** a multi-stream (4-camera) inference application.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** NVIDIA Jetson Nano / Xavier / Orin.
*   **Software:** TensorRT, CUDA, DeepStream SDK, `trtexec`.
*   **Models:** YOLOv8 (ONNX format).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why TensorRT?
*   **Frameworks (PyTorch/TF):** Optimized for *Training* (Flexibility, Gradients). Slow for Inference.
*   **TensorRT:** Optimized for *Inference* on NVIDIA GPUs.
    *   **Layer Fusion:** Combines multiple layers (Conv + Bias + ReLU) into a single CUDA kernel.
    *   **Kernel Selection:** Tests hundreds of kernels to find the fastest one for *your* specific GPU.
    *   **Memory Optimization:** Reuses memory for intermediate tensors.

### 🔹 Part 2: Precision (FP32 vs FP16 vs INT8)
*   **FP32 (32-bit Float):** Standard. High precision. Slow.
*   **FP16 (16-bit Float):** Half precision. 2x speedup on Tensor Cores. Minimal accuracy loss.
*   **INT8 (8-bit Integer):** 4x speedup. Requires **Calibration** (running sample images to determine the dynamic range of activations).

### 🔹 Part 3: DeepStream Architecture
*   **NVMM (NVIDIA Memory):** Keeps frames in GPU memory.
*   **Batching:** Aggregates frames from multiple cameras into a single "Batch" for the GPU.
*   **Zero-Copy:** The Camera writes to GPU, the ISP processes in GPU, the AI infers in GPU, the Encoder encodes in GPU. CPU is idle.

---

## 💻 Implementation Examples

### Example 1: Converting ONNX to TensorRT (`trtexec`)

The command-line swiss army knife.

```bash
# 1. Basic Conversion (FP32)
/usr/src/tensorrt/bin/trtexec --onnx=yolov8.onnx --saveEngine=yolov8_fp32.engine

# 2. FP16 Optimization (Jetson Default)
/usr/src/tensorrt/bin/trtexec --onnx=yolov8.onnx --saveEngine=yolov8_fp16.engine --fp16

# 3. INT8 Optimization (Needs Calibration Data, simplified here)
/usr/src/tensorrt/bin/trtexec --onnx=yolov8.onnx --saveEngine=yolov8_int8.engine --int8 --calib=calibration.cache
```

### Example 2: DeepStream Config (`config_infer_primary.txt`)

Configuring `nvinfer` to use the engine.

```ini
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-engine-file=yolov8_fp16.engine
labelfile-path=labels.txt
batch-size=4  # Process 4 cameras at once
process-mode=1 # Primary GIE
network-mode=2 # 0=FP32, 1=INT8, 2=FP16
interval=0    # Inference every frame
gie-unique-id=1
```

### Example 3: Python TensorRT Inference (Standalone)

If not using DeepStream.

```python
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

# 1. Load Engine
logger = trt.Logger(trt.Logger.WARNING)
with open("yolov8_fp16.engine", "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# 2. Allocate Buffers
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)

# 3. Inference Loop
def infer(image):
    # Copy to GPU
    cuda.memcpy_htod_async(inputs[0].device, image, stream)
    # Run
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Copy back
    cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
    stream.synchronize()
    return outputs
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Benchmarking with `trtexec`

**Objective:** Measure speedup.

**Steps:**
1.  Run `trtexec` with FP32. Note "Throughput: 45 qps".
2.  Run `trtexec` with FP16. Note "Throughput: 120 qps".
3.  **Observation:** Massive gain on Jetson Orin (which has Tensor Cores).

### Lab 2: Multi-Stream DeepStream

**Objective:** Max out the GPU.

**Steps:**
1.  Edit `source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt` (Sample Config).
2.  Change `num-sources=1` to `num-sources=4`.
3.  Run `deepstream-app -c ...`.
4.  **Monitor:** Run `sudo tegrastats`. Check GPU Load (GR3D).
5.  **Goal:** 4x 1080p streams at 30fps with < 90% GPU load.

### Lab 3: INT8 Calibration

**Objective:** Squeeze the last drop of performance.

**Steps:**
1.  Use the `create_calibration_cache.py` script (from TensorRT samples).
2.  Feed it 1000 representative images from your dataset.
3.  Generate `calib.cache`.
4.  Build INT8 engine.
5.  **Verify:** Check accuracy (mAP). Did it drop? Usually < 1% drop is acceptable.

---

## 🐛 Debugging Optimization Issues

### Debug 1: Accuracy Drop in FP16

**Symptom:** Detections are jittery or missing.

**Cause:**
*   Overflow/Underflow in 16-bit float range.
*   **Fix:** Keep the final output layers in FP32 (`--layerPrecisions=Output:fp32`).

### Debug 2: "Out of Memory" (OOM)

**Symptom:** Application crashes on start.

**Cause:**
*   Batch size too large.
*   Engine workspace too large.
*   **Fix:** Reduce batch size. Increase swap file on Jetson.

---

## ⚡ Performance Optimization

### Optimization 1: Dynamic Batching

*   If you have variable input (e.g., sometimes 1 camera, sometimes 4), use Dynamic Batching.
*   The engine waits a few ms to collect enough frames to fill a batch.

### Optimization 2: DLA (Deep Learning Accelerator)

*   Jetson Xavier/Orin has dedicated DLA cores (separate from GPU).
*   Offload the primary detector to DLA to leave GPU free for other tasks (Tracking, OCR).
*   `trtexec --useDLACore=0 --allowGPUFallback`.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is "Zero-Copy" critical for 4K video?** (Copying 4K frames (24MB) between CPU and GPU at 60fps saturates the memory bandwidth).
2.  **What is "Layer Fusion"?** (Merging `Conv -> BN -> ReLU` into one operation to reduce memory access).
3.  **Difference between `trtexec` and `deepstream-app`?** (`trtexec` benchmarks the model only. `deepstream-app` runs the full pipeline including decoding and display).
4.  **What is a "Tensor Core"?** (Specialized hardware for Matrix Multiplication).

### Practical Challenges

1.  **Profile with Nsight:** Run a trace. Identify if the bottleneck is the "Pre-processing" (CPU) or "Inference" (GPU).
2.  **Optimize a Custom Model:** Take a ResNet-50, prune 50% of filters, retrain, and deploy. Measure speedup.

---

## 📚 Further Reading & Resources

### Documentation
*   **NVIDIA TensorRT Developer Guide.**
*   **DeepStream SDK Plugin Manual.**

---

## 🎓 Summary

Today we covered:
- ✅ **TensorRT:** The speed engine.
- ✅ **Precision:** FP32 vs FP16 vs INT8.
- ✅ **DeepStream:** The video pipeline.
- ✅ **Benchmarking:** `trtexec`.
- ✅ **DLA:** Using all the silicon.

**Next:** Day 86 - System Hardening & Security Audit.

---

**Day 85 Complete** | Phase 3: Camera Systems & ISP | Week 15: Final Capstone & Career
