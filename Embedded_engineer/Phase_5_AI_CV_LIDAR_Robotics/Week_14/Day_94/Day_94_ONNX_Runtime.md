# Day 94: ONNX Runtime for Robotics
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 14: Edge AI Deployment

---

> **📝 Content Creator Instructions:**
> The Universal Adapter.
> - **Focus:** ONNX Runtime (ORT), Execution Providers (CUDA, TensorRT, OpenVINO, CPU), and why robotics engineers love it (Portability).
> - **Code:** A C++ ROS 2 Node that wraps `onnxruntime` to perform inference on a Lidar Point Cloud network (or Camera).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Understand** the role of ONNX (Open Neural Network Exchange) as the intermediate representation.
2.  **Select** the right Execution Provider (EP): `CUDAExecutionProvider` vs `TensorrtExecutionProvider`.
3.  **Deploy** a model on a CPU-only robot (Raspberry Pi) and a GPU robot (Jetson) using the *exact same code*.
4.  **Write** a robust C++ Wrapper for ORT.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Any (CPU implementation is standard).

### Software Environment
```bash
sudo apt install libonnxruntime-dev # Or download from GitHub
```

### Prior Knowledge
- C++17.
- CMake.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Fragmentation Problem

*   **Training:** PyTorch, TensorFlow, Jax.
*   **Hardware:** Nvidia GPU, Intel CPU, ARM CPU, NPU.
*   **Solution:** ONNX Runtime.
*   **Mechanism:** PyTorch exports `.onnx`. ORT loads `.onnx` and maps operators to the backend (Execution Provider).

### 🔹 Part 2: Execution Providers (EP)

*   **CPU (Default):** Uses MLAS/pthreads. Good for small models.
*   **CUDA:** Uses cuDNN. Fast.
*   **TensorRT:** Uses TRT engine inside ORT. Fastest on Nvidia.
*   **OpenVINO:** Intel CPUs/iGPUs.
*   **CoreML/NNAPI:** iOS/Android.

### 🔹 Part 3: C++ API

Robotics is C++. Python ORT is great for testing, but C++ ORT is needed for low-latency ROS nodes (`ros2_control` loops).
*   **Session:** Holds the model.
*   **Env:** Threading environment.
*   **RunOptions:** Per-inference settings.

---

## 💻 Implementation: C++ ORT Wrapper

We will create a class `YoloOrt` that runs a YOLOv5/v8 model.

### 🛠️ Project Structure
```text
day94_onnx/
├── include/day94_onnx/
│   └── yolo_ort.hpp
├── src/
│   ├── yolo_ort.cpp
│   └── main.cpp
└── CMakeLists.txt
```

### 👨‍💻 Header (`include/day94_onnx/yolo_ort.hpp`)

```cpp
#ifndef YOLO_ORT_HPP
#define YOLO_ORT_HPP

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class YoloOrt {
public:
    YoloOrt(const std::string& model_path, bool use_gpu = true);
    std::vector<int> detect(const cv::Mat& image); // Returns mock boxes for brevity

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    
    // Model metadata
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;
};

#endif
```

### 👨‍💻 Implementation (`src/yolo_ort.cpp`)

```cpp
#include "day94_onnx/yolo_ort.hpp"
#include <iostream>

YoloOrt::YoloOrt(const std::string& model_path, bool use_gpu) 
    : env_(ORT_LOGGING_LEVEL_WARNING, "YoloOrt") 
{
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_gpu) {
        // CUDA Provider Options
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "Using CUDA Execution Provider" << std::endl;
    }

    session_ = Ort::Session(env_, model_path.c_str(), session_options);

    // Get Input Info (Dynamic for generic ONNX)
    size_t num_input_nodes = session_.GetInputCount();
    input_names_.reserve(num_input_nodes);
    
    // NOTE: In modern ORT, getting names is slightly more verbose due to freeing memory
    // Simplified here for clarity (assuming persistent strings)
    for(size_t i=0; i<num_input_nodes; i++){
        // input_names_.push_back(session_.GetInputName(i, allocator_)); // v1.13+ syntax
        // For older versions it returns raw char*
    }
    
    // Hardcoding input shape for example [1, 3, 640, 640]
    input_shape_ = {1, 3, 640, 640}; 
}

std::vector<int> YoloOrt::detect(const cv::Mat& image) {
    // 1. Preprocess (Resize, Normalize, CHW)
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(0,0,0), true, false);
    
    // 2. Wrap Memory (No Copy if possible) -> Tensor
    size_t input_tensor_size = 1 * 3 * 640 * 640;
    std::vector<float> input_tensor_values(blob.begin<float>(), blob.end<float>());
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size, input_shape_.data(), input_shape_.size());

    // 3. Run
    // inputs
    const char* input_names[] = {"images"}; // Check Netron for exact name
    const char* output_names[] = {"output0"};

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr}, 
        input_names, &input_tensor, 1, 
        output_names, 1
    );

    // 4. Output Processing
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    // ... Post Processing (NMS) logic here ...
    
    return {};
}
```

---

## 🔬 Lab Exercise: "Provider Switch"

### 1. Lab Objectives
- **Setup:** Run the C++ code with `use_gpu = false`. Measure FPS.
- **Action:** Switch `use_gpu = true`.
- **Result:** Massive speedup (on Jetson).
- **Test:** Run on Laptop (Intel CPU). It works.
- **Test:** Run on Server (A100). It works.
- **Goal:** Verify "Write Once, Run Anywhere".

---

## 🚀 Project: "ROS 2 ONNX Node"

**Goal:** Integrate into ROS 2.
1.  **Subscriber:** `/camera/image_raw`.
2.  **Inference:** `YoloOrt::detect()`.
3.  **Publisher:** `/detections` (Bounding Boxes).
4.  **Visualize:** Rviz.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Segmentation Fault on GetInputName"
*   **Cause:** ONNX Runtime API changes frequently regarding string lifetime.
*   **Fix:** Ensure you copy the string returned by `GetInputNameAllocated` (v1.13+).

#### 2. "CUDA Provider not available"
*   **Cause:** Installed `onnxruntime` (CPU only) instead of `onnxruntime-gpu`.
*   **Fix:** `pip install onnxruntime-gpu` or link against correct C++ libs. Note: The versions must match CUDA version exactly (e.g., ORT 1.16 requires CUDA 11.8).

---

## ⚡ Optimization: I/O Binding

Standard `Run()` copies data CPU $\to$ GPU.
*   **IO Binding:** Pre-allocate output buffer on GPU. Tell ORT to write directly to GPU memory.
*   **Zero-Copy:** If Input is already on GPU (e.g., from `isaac_ros_image_proc`), pass the GPU pointer directly to ORT.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not just use TensorRT directly?
    *   **A:** TRT takes time to build engines (minutes). ONNX loads instantly. Also, ONNX works on non-Nvidia hardware. TRT is for "Maximum Performance Phase".
2.  **Q:** What is "Opset Version"?
    *   **A:** The "Language Version" of ONNX. PyTorch keeps adding new operators. You need to export with an Opset that your ORT version understands (e.g., Opset 11 is very stable).
3.  **Q:** Dynamic Axes?
    *   **A:** Allowing batch size or image size to change. ORT handles this better than TRT (which requires Profiles).

### Challenge Task
> **Task:** FP16 in ORT.
> 1. Export model as FP16 ONNX?
> 2. OR use `session_options.EnableCpuMemArena(false)`?
> 3. Actually, just adding CUDA Provider usually handles FP16 math if the GPU supports it, but converting the ONNX weights to FP16 reduces model size by 50%.

---

## 📚 Further Reading
- **Netron:** Visualizer for ONNX files.
- **ONNX Model Zoo:** Pre-trained models.

---

**Day 94 Complete**
