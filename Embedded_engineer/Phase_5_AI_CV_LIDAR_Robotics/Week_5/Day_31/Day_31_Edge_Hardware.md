# Day 31: Edge Hardware Acceleration (Jetson/TPU)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 5: Edge AI & Optimization

---

> **📝 Content Creator Instructions:**
> We have the software (TensorRT). Now we look at the silicon.
> - **Focus:** NVIDIA Jetson (Orin/Nano), Google Coral (Edge TPU), and Rockchip (NPU).
> - **Code:** Deployment scripts for TPU and Jetson GPU.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Compare** the architectures of GPGPU (Jetson) vs ASIC (TPU) vs FPGA.
2.  **Deploy** a `.tflite` model to a Google Coral USB Accelerator.
3.  **Utilize** the Deep Learning Accelerator (DLA) on Jetson Xavier/Orin to offload the GPU.
4.  **Profile** power consumption (`tegrostats`) and optimize Performance/Watt.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Optional: Jetson Nano/Orin or Google Coral Stick.
- Sim: We will write code that *would* run there.

### Software Environment
```bash
# For TPU
pip install tflite-runtime
# For Jetson stats
# sudo jtop 
```

### Prior Knowledge
- Quantization (Day 29) - TPUs *require* INT8.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Hardware Zoo

1.  **GPGPU (General Purpose GPU):** (NVIDIA Jetson).
    *   *Pros:* Flexible. Runs CUDA, PyTorch, C++. FP16/FP32 support.
    *   *Cons:* Power hungry (10W-60W).
2.  **ASIC (Application Specific IC):** (Google Coral TPU, Tesla FSD).
    *   *Pros:* Extremely efficient (TOPS/Watt).
    *   *Cons:* Rigid. Only runs specific operations (Matrix Multiply). Requires quantization.
3.  **NPU (Neural Processing Unit):** (Rockchip, Apple Neural Engine).
    *   Similar to ASIC, embedded in the SoC.

### 🔹 Part 2: NVIDIA Deep Learning Accelerator (DLA)

On Jetson Xavier/Orin, there are GPUs *and* DLAs.
*   DLA is a specialized HW block for inference.
*   *Why use it?* Save the GPU for heavy tasks like Mapping (SLAM) or Planning (MPC), while DLA runs Object Detection in the background.

### 🔹 Part 3: Google Coral (Edge TPU)

*   4 TOPS (Trillion Operations Per Second) @ 2 Watts.
*   Requires models to be **Full Integer Quantized**. Not just weights, but activations too.

---

## 💻 Implementation: TPU Deployment

We assume we have a `mobilenet_v2_quant.tflite` model (Day 29).

### 🛠️ Project Structure
```text
day31_hardware/
├── models/
│   └── mobilenet_quant.tflite
├── src/
│   ├── run_tpu.py
│   └── run_jetson_dla.py
└── monitor_power.sh
```

### 👨‍💻 Code Implementation: Google Coral (`src/run_tpu.py`)

```python
import time
import numpy as np
from PIL import Image
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    print("tflite_runtime not installed (only works on ARM/Linux usually)")

def run_inference(model_path, image_path):
    # Load Delegate (The driver that talks to the USB Stick)
    # 'libedgetpu.so.1' is the Coral driver
    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
    )
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess
    img = Image.open(image_path).resize((224, 224))
    input_data = np.expand_dims(img, axis=0) # uint8 [0, 255]
    
    # Run
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    start = time.perf_counter()
    interpreter.invoke() # BLOCKS until TPU finishes
    end = time.perf_counter()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"Inference Time: {(end-start)*1000:.2f} ms")
    return output_data
```

### 👨‍💻 Code Implementation: Jetson DLA via TensorRT (`src/run_jetson_dla.py`)

Using `trtexec` command line allows strictly assigning layers to DLA.

```bash
# Convert ONNX to TensorRT Engine, forcing DLA Core 0
trtexec --onnx=resnet.onnx \
        --saveEngine=resnet_dla.trt \
        --useDLACore=0 \
        --allowGPUFallback \
        --fp16
```

**Note:** `allowGPUFallback` is critical because DLA doesn't support *all* layers. If a layer isn't supported, it falls back to the GPU automatically.

---

## 🔬 Lab Exercise: The Power Challenge

### 1. Lab Objectives
- Measure Power vs FPS.
- **Tools:**
    - Jetson: `tegrastats` (Shows Voltage/Current for GPU rail, CPU rail).
    - USB Meter for Coral.
- **Experiment:**
    1.  Run ResNet on Jetson GPU (Max Pwr Mode). Result: 30ms, 10W. Efficiency: 3 FPS/Watt.
    2.  Run MobileNet on Coral TPU. Result: 10ms, 2W. Efficiency: 50 FPS/Watt.
- **Conclusion:** For pure CNN inference, ASICs destroy GPUs in efficiency. But GPUs are more versatile.

---

## 🚀 Project: "Multi-Cam Pipeline"

**Scenario:** A delivery robot has 4 Cameras.
**Hardware:** Jetson Orin NX.
**Configuration:**
- Cam 1 (Front): Object Detection (Run on **DLA 0**).
- Cam 2 (Rear): Segmentation (Run on **DLA 1**).
- Cam 3/4 (Stereo): Depth Estimation (Run on **GPU**).
- **Result:** All hardware blocks utilized. CPU free for ROS 2 networking.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Delegate not found"
*   **Cause:** Coral USB stick not plugged in, or `libedgetpu` not installed.
*   **Fix:** Check `lsusb`. Install `libedgetpu1-std` (standard speed) or `libedgetpu1-max` (overclocked, gets hot).

#### 2. "DLA Error"
*   **Symptom:** `trtexec` fails saying layer not supported.
*   **Fix:** DLA handles Conv/ReLU/Pooling well. It hates weird stuff like `ArgMax` or `NonMaxSuppression`. Run those on GPU.

---

## ⚡ Optimization: Pipeline Latency

Inferencing is fast (2ms). Copying data is slow.
*   **Zero-Copy:** On Jetson, CPU and GPU share RAM.
*   Do NOT use `cudaMemcpy`. Use **Unified Memory** or `EGLStreams` to pass camera frames directly to TensorRT without CPU buffers.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can I run Training on a Coral TPU?
    *   **A:** No. Inference Only. (Though Google has "Cloud TPUs" for training, Edge TPU is read-only).
2.  **Q:** What is the bottleneck usually?
    *   **A:** Data Transfer. Moving 4K images from Camera -> RAM -> GPU VRAM takes longer than the actual neural network processing.
3.  **Q:** Why use DLA if GPU is faster?
    *   **A:** To run *more* things parallel. Freeing up the GPU allows you to run heavier algorithms (like heavy SLAM) that DLA cannot handle.

### Challenge Task
> **Task:** Thermal Throttling.
> 1. Run the Coral TPU at 100% duty cycle.
> 2. Measure temperature.
> 3. Observe the speed drop when it hits 85°C.
> 4. Design a passive heatsink strategy.

---

## 📚 Further Reading
- **Coral Docs:** coral.ai/docs
- **Jetson DLA:** NVIDIA Developer Blog.

---

**Day 31 Complete**
