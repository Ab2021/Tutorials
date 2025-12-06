# Day 22: TensorRT Architecture and Workflow
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 4: TensorRT & Inference Optimization

---

> **🎯 Focus Area:** Understand the architecture of NVIDIA TensorRT and master the workflow of compiling PyTorch/ONNX models into highly optimized inference engines.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** how TensorRT optimizes neural networks (Layer Fusion, Kernel Tuning, Precision Calibration).
2.  **Differentiate** between the "Build Phase" (Optimization) and "Runtime Phase" (Inference).
3.  **Build** a TensorRT engine from scratch using the Python Network Definition API.
4.  **Execute** inference using the TensorRT Runtime.
5.  **Compare** latency and throughput against standard PyTorch inference.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU (TensorRT is NVIDIA optimized).
- Tensor Cores strongly recommended for FP16/INT8 demos.

### Software Environment
```bash
# Install TensorRT Python bindings
pip install tensorrt cuda-python

# Install PyTorch for comparison and ONNX export
pip install torch torchvision onnx
```

### Prior Knowledge
- Neural Networks (Layers, Weights, Activations).
- Week 3: cuDNN (TensorRT uses cuDNN and cuBLAS internally but autotunes them).

---

## 📖 Theoretical Foundation

### 1. What is TensorRT?

TensorRT is an SDK for high-performance deep learning inference. It includes:
1.  **Optimizer (The Builder):** Takes a network definition (e.g., ONNX), performs graph optimizations, and outputs a "Plan" (Engine).
2.  **Runtime:** Executes the engine efficiently on the GPU.

### 2. The Optimization Pipeline

When you "build" an engine, TensorRT performs:

*   **Layer Fusion:** Combines layers to reduce kernel launches and memory R/W.
    *   *Vertical Fusion:* `Conv -> Bias -> ReLU` becomes one kernel.
    *   *Horizontal Fusion:* Combining parallel branches with same inputs.
*   **Kernel Auto-Tuning:** TensorRT runs dummy inputs through layers using every available algorithm (cuDNN, cuBLAS, specialized TRT kernels) and selects the fastest one for *your specific GPU* and input shape.
*   **Precision Calibration:** Quantizes weights/activations to FP16 or INT8 while minimizing accuracy loss (Optional).
*   **Dynamic Tensor Memory:** Minimizes memory footprint by creating efficient memory reuse plans.

### 3. Workflow: ONNX is King

The standard workflow for Platform Engineers:
1.  **Train** model in PyTorch/TensorFlow.
2.  **Export** to ONNX (Open Neural Network Exchange).
3.  **Parse** ONNX into TensorRT.
4.  **Build** Engine (`.plan` or `.trt` file).
5.  **Deploy** Engine (C++ or Python Runtime).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The "Hello World" of TensorRT

We will define a simple network (Input -> Linear -> Relu -> Output) directly using the TensorRT Python API, bypassing ONNX for a moment to understand the `INetworkDefinition` structure.

#### 📁 `src/trt_simple_network.py`
```python
#!/usr/bin/env python3
"""
Day 22: TensorRT Manual Network Definition
Phase 6: Platform Engineering
"""

import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# Logger is required
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine():
    print("Building TensorRT Engine...")
    
    # 1. Builder and Network Definition
    builder = trt.Builder(TRT_LOGGER)
    
    # Explicit Batch flag is standard in modern TRT
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Configuration (Memory limits, etc.)
    config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
    
    # 2. Define the Network Structure MANUALLY
    # Input Layer name, data type, shape (Batch, Size)
    input_tensor = network.add_input(name='input0', dtype=trt.float32, shape=(1, 5))
    
    # Weights for Fully Connected Layer
    # Simple identity-like weights for demo
    w_np = np.eye(5, dtype=np.float32).ravel()
    b_np = np.zeros(5, dtype=np.float32)
    
    # Add Fully Connected Layer (Matrix Multiply + Bias)
    fc_layer = network.add_fully_connected(
        input=input_tensor,
        num_outputs=5,
        kernel=w_np,
        bias=b_np
    )
    
    # Add Activation (ReLU)
    relu_layer = network.add_activation(
        input=fc_layer.get_output(0), 
        type=trt.ActivationType.RELU
    )
    
    # Mark Output
    network.mark_output(relu_layer.get_output(0))
    
    # 3. Build Serialized Engine
    serialized_engine = builder.build_serialized_network(network, config)
    
    print("Engine Built successfully.")
    return serialized_engine

def run_inference(serialized_engine):
    print("\nRunning Inference...")
    
    # 1. Deserialize Engine
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    # 2. Create Execution Context
    context = engine.create_execution_context()
    
    # 3. Allocation (Host & Device)
    # Get Input/Output details
    input_name = 'input0'
    # In newer TRT, use get_tensor_shape / get_tensor_mode
    # Assuming single input/output for simplicity
    
    h_input = np.random.normal(size=(1, 5)).astype(np.float32)
    h_output = np.zeros((1, 5), dtype=np.float32)
    
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    # 4. Inference Check
    # Transfer Input
    cuda.memcpy_htod(d_input, h_input)
    
    # Set Tensor Addresses
    context.set_tensor_address(input_name, int(d_input))
    # output name is usually mapped by index 1 if not named explicitly in new API
    # Looking up name of output
    output_name = engine.get_tensor_name(1) 
    context.set_tensor_address(output_name, int(d_output))
    
    # Execute (Async V3 API is preferred, but execute_v2 is common)
    context.execute_async_v3(stream_handle=0)
    
    # Transfer Output
    cuda.memcpy_dtoh(h_output, d_output)
    cuda.Context.synchronize()
    
    print(f"Input:  {h_input}")
    print(f"Output: {h_output}")
    print("Verification: Should match (Identity * ReLU)")

if __name__ == "__main__":
    engine_bytes = build_engine()
    run_inference(engine_bytes)
```

### 👨‍💻 Workflow Implementation: PyTorch to TensorRT (via ONNX)

This is the real-world workflow.

#### 📁 `src/torch_to_trt.py`
```python
#!/usr/bin/env python3
"""
Day 22: PyTorch to TensorRT Workflow
Phase 6: Platform Engineering
"""

import torch
import torch.nn as nn
import torch.onnx
import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 1. Define PyTorch Model
class SimpleResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 32 * 32, 10) # Assuming 32x32 input

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def export_onnx(model, filename="model.onnx"):
    dummy_input = torch.randn(1, 3, 32, 32).cuda()
    model.eval()
    
    print(f"Exporting to {filename}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        filename, 
        opset_version=13,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

def build_engine_from_onnx(onnx_file_path, engine_file_path):
    print("Building TRT Engine from ONNX...")
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Enable FP16 if supported
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 Mode Enabled")
        
    # Parse ONNX
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Build
    # Note: create_optimization_profile required for dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 32, 32), (16, 3, 32, 32), (32, 3, 32, 32))
    config.add_optimization_profile(profile)
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    # Save
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
        
    print(f"Engine saved to {engine_file_path}")

def main():
    # Setup
    model = SimpleResNet().cuda()
    export_onnx(model)
    
    # Build
    build_engine_from_onnx("model.onnx", "model.trt")
    
    # Cleanup
    if os.path.exists("model.onnx"):
        print("Cleaning up ONNX file.")
        
if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Latency Shootout"

### Lab Objectives
1.  Measure the inference time of the `SimpleResNet` using native PyTorch on GPU.
2.  Measure the inference time of the generated TensorRT engine (`model.trt`).
3.  Calculate the speedup.

### Expectation
For small models, overhead dominates, but you should still see 1.5x - 2x. For larger models (ResNet50, BERT), TensorRT often provides **3x to 6x speedups**, especially with FP16 enabled.

### Why is TRT faster?
1.  **Vertical Fusion:** `Conv2d + ReLU` is 1 kernel launch in TRT. In PyTorch, it's 2 (load output of conv, apply relu).
2.  **Kernel Tuning:** PyTorch makes a heuristic guess on which cuDNN algorithm to use. TRT *benchmarks* them all during build time.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Compile Time vs Run Time:** TensorRT introduces a "Compile" step deployment. This takes time (minutes) but makes runtime (milliseconds) blazing fast.
2.  **Engine Specificity:** An engine built on an RTX 3090 **cannot** run on an RTX 4090 or T4. It is tuned for specific hardware SMs. You must build on the target device.
3.  **ONNX Bridge:** Platform engineers rarely write TRT networks by hand. We export from PyTorch/TF to ONNX, then compile ONNX to TRT.
4.  **FP16 is Free Speed:** Enabling `config.set_flag(trt.BuilderFlag.FP16)` often doubles performance on modern GPUs with minimal accuracy loss.

### API Summary
```python
# Create Builder
builder = trt.Builder(logger)
network = builder.create_network(...)
parser = trt.OnnxParser(network, logger)

# Config
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)

# Build
engine = builder.build_serialized_network(network, config)

# Runtime
context.execute_async_v3(stream)
```

---

**Day 22 Complete** ✅

*Next: Day 23 - Advanced TensorRT - Dynamic Shapes, Plugins, and INT8 Calibration!*
