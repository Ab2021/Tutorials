# Day 23: Advanced TensorRT - INT8, Dynamic Shapes, and Plugins
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 4: TensorRT & Inference Optimization

---

> **🎯 Focus Area:** Master critical production capabilities: handling variable input sizes (Dynamic Shapes), executing unsupported layers (Plugins), and achieving 4x speedups with INT8 Quantization.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** Optimization Profiles to handle dynamic batch sizes and resolutions.
2.  **Implement** a custom `IInt8EntropyCalibrator` for Post-Training Quantization (INT8).
3.  **Create** and register a TensorRT Plugin for unsupported operators.
4.  **Debug** precision issues using **Polygraphy**.
5.  **Achieve** maximum throughput using mixed-precision strategies.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU with **INT8 Tensor Cores** (Turing T4, Ampere A100/3090/4090).
- *Note: Pascal (GTX 1080) supports INT8 but with reduced throughput compared to modern architectures.*

### Software Environment
```bash
# Helper tools for TRT
pip install polygraphy onnx-graphsurgeon
```

### Prior Knowledge
- Day 22: Basic TensorRT Workflow.
- Quantization Concepts (FP32 -> INT8 mapping, Scale factors).

---

## 📖 Theoretical Foundation

### 1. Dynamic Shapes & Optimization Profiles

In standard deployment, batch size varies (1 user vs 32 users). TensorRT kernels are auto-tuned for specific shapes. If shapes change, TRT needs a "Range".

**Optimization Profile:** You define three sets of dimensions for every input:
1.  **Min:** Smallest expected input (e.g., Batch=1).
2.  **Opt:** Most common input (e.g., Batch=16). TRT optimizes for this.
3.  **Max:** Largest possible input (e.g., Batch=64). Determines memory allocation.

### 2. INT8 Quantization (PTQ)

Representing weights/activations in 8-bit integers reduces memory bandwidth by 4x and compute by 2-4x.
*   **Challenge:** Loss of dynamic range.
*   **Solution (Calibration):** We pass a "Representative Dataset" (e.g., 100 images) through the network *before* building the engine. TRT observes the histograms of activation values and calculates optimal scale factors to map `float` range `[-A, A]` to `int8` range `[-127, 127]`.

### 3. Plugins

Sometimes an ONNX model contains an operator (e.g., `NonMaxSuppression` or a custom GridSampler) that TensorRT doesn't support natively.
*   **Plugin:** A custom C++/CUDA implementation of a layer that implements `IPluginV2`.
*   **Registry:** You "register" the plugin, and the ONNX Parser calls it when it sees the specific Op node.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: INT8 Calibration

This is the most common advanced task: Quantizing a FP32 model to INT8.

#### 📁 `src/int8_calibration.py`
```python
#!/usr/bin/env python3
"""
Day 23: TensorRT INT8 Calibration
Phase 6: Platform Engineering
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os

# 1. Defined Calibrator Class
class RandomDataLoader:
    """
    Simulates a data loader. In production, load real images here!
    """
    def __init__(self, batch_size, shape, count=10):
        self.index = 0
        self.count = count
        self.batch_size = batch_size
        self.shape = shape
        self.nbytes = batch_size * np.prod(shape) * 4 # FP32 bytes
        self.d_input = cuda.mem_alloc(self.nbytes)
    
    def next_batch(self):
        if self.index >= self.count:
            return None
        
        # Generate fake data (Calibration needs REAL data usually to find correct ranges)
        # Using random for syntax demo
        data = np.random.normal(size=(self.batch_size, *self.shape)).astype(np.float32)
        cuda.memcpy_htod(self.d_input, data)
        self.index += 1
        return [int(self.d_input)]
        
    def reset(self):
        self.index = 0

class MyEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data_loader, cache_file):
        # Initialize base class
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.loader = calibration_data_loader
        self.cache_file = cache_file

    def get_batch_size(self):
        return self.loader.batch_size

    def get_batch(self, names):
        # TensorRT calls this to get data pointers for calibration
        return self.loader.next_batch()

    def read_calibration_cache(self):
        # Load pre-computed scales to avoid re-calibrating
        if os.path.exists(self.cache_file):
            print("Reading calibration cache...")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        # Save scales for future use
        print("Writing calibration cache...")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_int8_engine(onnx_path, engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())
        
    # Check if INT8 supported
    if builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        
        # Create Calibrator
        # Assumes input size (3, 32, 32)
        loader = RandomDataLoader(batch_size=8, shape=(3, 32, 32), count=5)
        calibrator = MyEntropyCalibrator(loader, "calib.cache")
        config.int8_calibrator = calibrator
        print("INT8 Configured with Calibration.")
    else:
        print("INT8 not supported on this platform. Fallback to FP16/FP32.")
        
    # Build
    # Note: Optimization profile needed if dynamic input
    profile = builder.create_optimization_profile()
    profile.set_shape("input0", (1, 3, 32, 32), (8, 3, 32, 32), (16, 3, 32, 32)) 
    config.add_optimization_profile(profile)

    # Note: OnnxParser sets input names. We assume 'input0' for demo.
    # In real code, check `network.get_input(0).name`
    
    engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(engine)
    print("INT8 Build Complete.")

# Dummy Main logic assuming onnx exists
# if __name__ == "__main__":
#     build_int8_engine("model.onnx", "model_int8.trt")
```

### 👨‍💻 Workflow: Polygraphy for Debugging

Polygraphy is NVIDIA's command-line toolkit to inspect models and compare TRT vs ONNX Runtime. Validating that INT8 didn't break accuracy is crucial.

#### 📁 `src/debug_workflow.sh`
```bash
#!/bin/bash
# Day 23: Debugging Workflow
# 1. Generate a dummy ONNX model (using polygraphy's surgeon or torch export)

# Create a sample model
polygraphy run --gen-model-model model.onnx \
    --model-type onnx \
    --input-shapes input:[1,3,224,224]

# 2. Compare TRT FP32 vs ONNX Runtime
# This runs both and checks absolute tolerance
polygraphy run model.onnx \
    --trt --onnxrt \
    --trt-fp16 \
    --atol 1e-3 --rtol 1e-3

# 3. Inspect a TRT Engine
# (Assuming built previously)
polygraphy inspect model model_int8.trt --display-layer-info

# 4. Check precision (INT8 vs FP32)
# Generates Golden values from ONNX Runtime (FP32) and compares against TRT-INT8
polygraphy run model.onnx \
    --trt --int8 \
    --onnxrt \
    --val-range input:[0,1] \
    --atol 0.1 
    # Tolerance is higher for INT8
```

---

## 🔬 Lab Exercise: "Dynamic Shapes in Action"

### Lab Objectives
1.  Train/Create a simple PyTorch model covering variable sequence lengths (e.g., Simple RNN or Transformer logic).
2.  Export with dynamic axes.
3.  Build TRT engine with `add_optimization_profile`.
4.  Run inference with Batch=1 and Batch=8 in loop.

### Key Python Logic
```python
# During Build
profile.set_shape("input", min=(1, 3, 224, 224), opt=(8, 3, 224, 224), max=(32, 3, 224, 224))
config.add_optimization_profile(profile)

# During Runtime
context.set_input_shape("input", (current_batch_size, 3, 224, 224))
# Note: Input shape MUST be between min and max defined in profile!
```

---

## 📝 Daily Summary

### Key Takeaways
1.  **Dynamic Shapes:** Essential for production servers handling varying traffic. Profiling multiple ranges (Opt=1, Opt=32) allows TRT to generate specialized kernels for each size if beneficial.
2.  **Calibration:** INT8 is not magic. It requires representative data. If data distribution drifts (e.g., day vs night images), calibration becomes invalid (Accuracy drops).
3.  **Accuracy vs Speed:** INT8 usually gives $<1\%$ accuracy drop for 3x speedup. If accuracy drops more, investigate "Layer Partial Quantization" (forcing sensitive layers to FP16).
4.  **Polygraphy:** Use it. It turns hours of "Why is my output all zeros?" debugging into a simple CLI report.

### API Summary
```python
# INT8
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = MyCalibrator(...)

# Dynamic Shapes
profile = builder.create_optimization_profile()
profile.set_shape(name, min_shape, opt_shape, max_shape)
config.add_optimization_profile(profile)

# Runtime Resizing
context.set_input_shape(name, shape)
```

---

**Day 23 Complete** ✅

*Next: Day 24 - Triton Inference Server - Scaling from one model to a production microservice!*
