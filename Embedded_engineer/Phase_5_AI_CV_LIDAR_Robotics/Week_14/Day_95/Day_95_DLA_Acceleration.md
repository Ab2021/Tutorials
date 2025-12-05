# Day 95: DLA (Deep Learning Accelerator)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 14: Edge AI Deployment

---

> **📝 Content Creator Instructions:**
> The GPU is busy planning. The DLA should handle the Eyes.
> - **Focus:** Understanding the Deep Learning Accelerator (DLA) ASIC, its constraints (Supported Layers), and how to offload models to it using TensorRT.
> - **Code:** A Python script creating a TensorRT engine specifically targeting `DLA Core 0`, leaving the GPU 0% utilized.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Identify** the DLA cores on Jetson Orin/Xavier (Specialized Fixed-Function hardware).
2.  **Determine** layer compatibility (DLA supports Conv2d, but maybe not deformable convolutions).
3.  **Configure** TensorRT builder to target `use_dla_core=0`.
4.  **Design** a pipeline where DLA runs perception (YOLO) and GPU runs Mapping/Planning.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA Jetson Orin or Xavier (DLA not present on Nano/TX2/Desktop).

### Software Environment
```bash
# Standard TensorRT Container
```

### Prior Knowledge
- TensorRT Builder API (Day 93).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is DLA?

DLA is not a GPU. It is an ASIC (Application Specific Integrated Circuit) designed *only* for Convolutional Neural Networks.
*   **Pros:** Extremely power efficient (Energy per inference is low). Frees up the GPU.
*   **Cons:** Not flexible. Supports a subset of layers (Conv, Pool, Scale, Activation). No generic CUDA kernels.
*   **Architecture:** Convolution Core, Planar Data Processor, Active Pointwise.

### 🔹 Part 2: GPU Fallback

What if my model has a layer DLA doesn't support?
*   TensorRT is smart. It splits the graph.
*   Part A (Supported) $\to$ DLA.
*   Part B (Unsupported) $\to$ GPU.
*   **Warning:** Switching between DLA and GPU incurs overhead (Copying memory?).
    *   Ideally, the entire backbone runs on DLA.

### 🔹 Part 3: Asynchronous Compute

On Orin AGX:
*   GPU: 1 Unit.
*   DLA: 2 Units (Core 0, Core 1).
*   **Throughput:** You can run 3 copies of YOLO simultaneously! Or run YOLO on DLA and Depth Anything on GPU.

---

## 💻 Implementation: Target the DLA

We will modify our TensorRT build script to force DLA usage.

### 🛠️ Project Structure
```text
day95_dla/
├── src/
│   ├── build_dla_engine.py
│   └── check_compatibility.sh
└── output/
    └── resnet_dla.engine
```

### 👨‍💻 Build Script (`src/build_dla_engine.py`)

```python
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_dla_engine(onnx_path, engine_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())

    # --- DLA Configuration ---
    
    # 1. Check if DLA is available
    num_dla = builder.num_dla_cores
    print(f"Number of DLA Cores: {num_dla}")
    if num_dla == 0:
        print("No DLA found! Falling back to GPU.")
    else:
        # 2. Enable DLA
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK) # Allow fallback if layer unsupported
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = 0 # Target Core 0
        print("Targeting DLA Core 0")

    # 3. FP16 is mandatory for DLA (usually)
    config.set_flag(trt.BuilderFlag.FP16)
    
    # Build
    serialized_engine = builder.build_serialized_network(network, config)
    
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print("Engine Built.")

if __name__ == "__main__":
    # Ensure you are on a Jetson for this to work properly
    build_dla_engine("resnet50.onnx", "resnet_dla.engine")
```

### 👨‍💻 Compatibility Checker

How do we know which layers failed?

```bash
# Use trtexec with verbose logging
trtexec --onnx=resnet50.onnx --useDLACore=0 --allowGPUFallback --verbose > build_log.txt
```
*   **Inspect Log:** Look for `Layer XXX runs on DLA` vs `Layer YYY runs on GPU`.
*   If too many layers fall back, performance might be worse than GPU-only due to context switching.

---

## 🔬 Lab Exercise: "Resource Monitor"

### 1. Lab Objectives
- **Run:** The DLA engine inference loop.
- **Monitor:** `tegrastats` or `jtop`.
- **Observation:**
    *   `GR3D` (GPU) load should be near 0% (or low).
    *   `DLA0` load should be high.
- **Task:** Run TWO instances of the inference script.
    *   Script 1: Target `config.DLA_core = 0`.
    *   Script 2: Target `config.DLA_core = 1` (If AGX).
    *   Result: Doubled throughput without touching GPU.

---

## 🚀 Project: "The Efficient Scout"

**Goal:** Long-endurance Surveillance.
1.  **Robot:** Moves slowly (Low CPU Nav).
2.  **Vision:** Person Detection (MobileNetV2 or ResNet-Detect).
3.  **Constraint:** Must run fully on DLA.
4.  **Power:** Switch NVPModel to `15W`.
5.  **Metric:** Measure FPS/Watt. DLA should beat GPU significantly.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Layer Not Supported"
*   **Symptom:** Entire model runs on GPU despite request.
*   **Cause:** Input dimensions must often be aligned (e.g., channels multiple of 32) for DLA. Start with standard backbones (ResNet, MobileNet). Custom architectures need care.
*   **Fix:** Read DLA Supported Layers Guide. Use simple layers (Conv, Batchnorm, Relu). Avoid weird Reshapes.

#### 2. "Slower than GPU?"
*   **Cause:** DLA is clocked lower than GPU. It's built for Efficiency, not pure Peak Performance (though usually comparable).
*   **Fix:** Use DLA for background tasks, GPU for latency-critical tasks.

---

## ⚡ Optimization: INT8 on DLA

DLA shines with INT8.
*   **Pipeline:** PTQ (Post Training Quantization).
*   **Calibrator:** Same as GPU (entropy).
*   **Result:** Maximum TOPS efficiency.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can DLA run Transformers (Attention)?
    *   **A:** Traditionally No. Newest DLA (Orin) has some support, but mostly it falls back to GPU. DLA is for CNNs.
2.  **Q:** Does `onnxruntime` support DLA?
    *   **A:** Yes, via the TensorRT Execution Provider. You pass `{device_id: 0, trt_dla_enable: true, trt_dla_core: 0}` options.
3.  **Q:** What is "GPU Fallback"?
    *   **A:** The ability of the builder to say "This layer can't run on DLA, so run it on GPU". Without this flag, build fails if any layer is unsupported.

### Challenge Task
> **Task:** Heterogeneous Pipeline.
> 1. Run Localization (VIO) on GPU.
> 2. Run Object Detection on DLA.
> 3. Verify total system load is balanced.

---

## 📚 Further Reading
- **NVIDIA DLA Open Source:** The DLA compiler itself is open source!
- **Supported Layers Table:** Critical reference.

---

**Day 95 Complete**
