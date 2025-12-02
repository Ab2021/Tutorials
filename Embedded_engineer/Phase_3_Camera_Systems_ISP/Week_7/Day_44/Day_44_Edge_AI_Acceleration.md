# Day 44: Edge AI Acceleration (TensorRT & TFLite)
## Phase 3: Camera Systems & ISP | Week 7: Machine Vision & Edge AI

---

## 🎯 Learning Objectives
1.  **Understand** the need for optimization on Edge Devices (Latency, Power, Memory).
2.  **Analyze** the TensorRT workflow (Parse -> Build Engine -> Serialize -> Infer).
3.  **Implement** TensorFlow Lite (TFLite) inference on CPU/Edge TPU.
4.  **Apply** Quantization (FP32 -> FP16 -> INT8) to reduce model size and increase speed.
5.  **Compare** different accelerators: GPU (Jetson), TPU (Coral), NPU (Rockchip/Amba).
6.  **Debug** accuracy loss due to quantization.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Nvidia Jetson (for TensorRT) or Raspberry Pi (for TFLite).
*   **Software:** TensorRT SDK, TensorFlow Lite Runtime.
*   **Knowledge:** ONNX (Open Neural Network Exchange) format.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Acceleration?
*   **Desktop GPU:** 300W, Huge Memory. Can run raw PyTorch models.
*   **Edge Device:** 5W-15W, Limited Memory.
*   **Optimization Techniques:**
    *   **Layer Fusion:** Combining Conv + Bias + ReLU into a single kernel to reduce memory access.
    *   **Kernel Auto-Tuning:** Selecting the best algorithm (Winograd, GEMM) for the specific hardware.
    *   **Quantization:** Using 8-bit integers instead of 32-bit floats. 4x smaller, 4x faster math.

### 🔹 Part 2: TensorRT (Nvidia)
*   **Workflow:**
    1.  **Import:** Take an ONNX model.
    2.  **Build:** The "Builder" optimizes the graph for the specific GPU (e.g., Orin Nano). This takes time.
    3.  **Serialize:** Save the optimized "Engine" plan to disk (`.engine` or `.trt`).
    4.  **Runtime:** Load the Engine and run inference (very fast).
*   **Precision:** Supports FP32, FP16, INT8 (requires calibration).

### 🔹 Part 3: TensorFlow Lite (Google)
*   **Workflow:** Convert `.h5` or SavedModel to `.tflite` FlatBuffer.
*   **Interpreters:**
    *   **XNNPACK:** Optimized CPU kernels (ARM NEON).
    *   **GPU Delegate:** Uses OpenCL/OpenGL ES.
    *   **Edge TPU Delegate:** Runs on Coral USB Stick (INT8 only).

---

## 💻 Implementation Examples

### Example 1: TensorRT Inference (C++)

Simplified flow using the `nvinfer1` API.

```cpp
/**
 * @brief TensorRT Inference
 */
#include <NvInfer.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
} logger;

void run_tensorrt(const char* engine_file) {
    // 1. Load Engine
    std::ifstream file(engine_file, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    
    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size, nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    
    // 2. Allocate Buffers (Input/Output)
    void* buffers[2];
    cudaMalloc(&buffers[0], input_size); // Input
    cudaMalloc(&buffers[1], output_size); // Output
    
    // 3. Copy Input to GPU
    cudaMemcpy(buffers[0], input_host, input_size, cudaMemcpyHostToDevice);
    
    // 4. Execute
    context->executeV2(buffers);
    
    // 5. Copy Output to Host
    cudaMemcpy(output_host, buffers[1], output_size, cudaMemcpyDeviceToHost);
    
    // Cleanup...
}
```

### Example 2: TFLite Inference (Python)

```python
import tensorflow.lite as tflite
import numpy as np

def run_tflite(model_path, input_data):
    # 1. Load Interpreter
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 2. Set Input
    # Ensure type matches (e.g., float32 or uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # 3. Invoke
    interpreter.invoke()
    
    # 4. Get Output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data
```

### Example 3: INT8 Calibration (Concept)

To convert to INT8, we need to know the dynamic range of activations to scale them correctly.
1.  **Calibration Dataset:** A set of representative images (e.g., 100 images).
2.  **Run FP32 Inference:** Measure the min/max values of every layer.
3.  **Calculate Scale Factors:** Map [Min, Max] to [-127, 127].
4.  **Generate INT8 Engine:** Hardcode these scales into the model.

---

## 🔬 Hands-On Lab Exercises

### Lab 1: FP32 vs FP16 Speedup

**Objective:** Measure the impact of reduced precision.

**Steps:**
1.  Build a TensorRT engine for YOLOv3 in **FP32** mode. Measure FPS.
2.  Build a TensorRT engine for YOLOv3 in **FP16** mode. Measure FPS.
3.  **Observation:** FP16 should be ~2x faster on Jetson (Tensor Cores).
4.  **Accuracy:** Check if detections are still valid. Usually identical.

### Lab 2: TFLite on Raspberry Pi

**Objective:** Run MobileNet.

**Steps:**
1.  Download `mobilenet_v2_1.0_224.tflite`.
2.  Run inference loop in Python.
3.  **Benchmark:**
    *   1 thread: 10 FPS.
    *   4 threads: 30 FPS.
4.  **Observation:** Multi-threading helps on CPU.

### Lab 3: Converting ONNX to TensorRT

**Objective:** Use `trtexec`.

**Steps:**
1.  Export a PyTorch model to ONNX.
2.  Run command:
    ```bash
    /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
    ```
3.  **Result:** A highly optimized engine file ready for deployment.

---

## 🐛 Debugging Techniques

### Debug 1: Accuracy Drop in INT8

**Symptom:** Model runs fast but detects nothing or garbage.

**Cause:**
*   Calibration dataset was too small or not representative (e.g., calibrated on day images, tested on night images).
*   Sensitive layers (first/last) quantized too aggressively.
*   **Fix:** Use "Quantization Aware Training" (QAT) or keep sensitive layers in FP16.

### Debug 2: "Cuda Error: Out of Memory"

**Symptom:** Crash during engine build.

**Cause:**
*   TensorRT workspace size too small.
*   Model too large for GPU RAM.
*   **Fix:** Increase workspace size or reduce batch size.

---

## ⚡ Performance Optimization

### Optimization 1: Batching

*   Processing 4 images at once is faster than 4 images sequentially.
*   GPU kernels have overhead; batching amortizes this overhead.
*   **Trade-off:** Increases latency for the *first* image, but increases overall Throughput (FPS).

### Optimization 2: Pipelining

*   Copy Image N+1 to GPU *while* GPU is processing Image N.
*   Use CUDA Streams (`cudaStream_t`) to overlap Compute and Memory Transfer.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is INT8 faster than FP32?** (Memory bandwidth + Tensor Core throughput).
2.  **What is the difference between "Post-Training Quantization" (PTQ) and "Quantization Aware Training" (QAT)?**
3.  **Why does TensorRT need to "Build" an engine for a specific GPU?**
4.  **What is an "ONNX Runtime"?**

### Practical Challenges

1.  **Build a "Smart Camera" pipeline:** Capture (Libcamera) -> Pre-process (OpenCV CUDA) -> Inference (TensorRT) -> Display. Measure end-to-end latency.
2.  **Implement "Dynamic Batching":** If images arrive fast, batch them. If slow, run immediately to minimize latency.

---

## 📚 Further Reading & Resources

### Documentation
*   **Nvidia TensorRT Developer Guide.**
*   **TensorFlow Lite Guide.**

### Tools
*   **Netron:** Visualizer for neural network graphs (ONNX, TFLite).

---

## 🎓 Summary

Today we covered:
- ✅ **Acceleration:** Why we need it.
- ✅ **TensorRT:** The gold standard for Nvidia GPUs.
- ✅ **TFLite:** The standard for Mobile/CPU.
- ✅ **Quantization:** FP16/INT8 trade-offs.
- ✅ **Workflow:** ONNX -> Engine -> Inference.

**Next:** Day 45 - Week 7 Review & Edge AI Project.

---

**Day 44 Complete** | Phase 3: Camera Systems & ISP | Week 7: Machine Vision & Edge AI
