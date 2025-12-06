# Day 25: Model Compression - Quantization Aware Training & Pruning
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 4: TensorRT & Inference Optimization

---

> **🎯 Focus Area:** Techniques to compress models and retain accuracy beyond standard Post-Training Quantization (PTQ), focusing on Quantization Aware Training (QAT) and Structured Pruning.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Distinguish** between Post-Training Quantization (PTQ) and Quantization Aware Training (QAT).
2.  **Implement** QAT using NVIDIA's `pytorch-quantization` toolkit.
3.  **Export** models with explicit Quantize-Dequantize (Q-DQ) nodes to ONNX.
4.  **Understand** the trade-offs of Structured vs. Unstructured Pruning on GPUs.
5.  **Apply** Knowledge Distillation to recover accuracy loss from compression.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU with Tensor Cores (Required for seeing speedup from INT8).

### Software Environment
```bash
# NVIDIA Quantization Toolkit used for QAT
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
pip install onnx onnxruntime
```

### Prior Knowledge
- Day 23: INT8 Calibration (PTQ).
- Basics of Backpropagation (for QAT fine-tuning).

---

## 📖 Theoretical Foundation

### 1. PTQ vs QAT

*   **PTQ (Post-Training Quantization):** Take a pre-trained FP32 model, run calibration data, calculate scales. Fast, but works poorly for MobileNet or small models where every bit of precision counts.
*   **QAT (Quantization Aware Training):** Add "Fake Quantization" nodes during training. The forward pass simulates INT8 rounding errors. The backward pass (Straight Through Estimator) updates FP32 weights to be robust to this noise. Result: **Higher accuracy INT8 models.**

### 2. Q-DQ Workflow

Modern TensorRT (8.0+) prefers **Explicit Quantization**:
1.  PyTorch model has `Quantize` and `Dequantize` nodes inserted.
2.  Export to ONNX preserves these Q-DQ nodes.
3.  TensorRT parses Q-DQ nodes and fuses layers between them into INT8 kernels.

### 3. Pruning on GPU

"Removing connections" sounds good, but GPUs hate irregularity.
*   **Unstructured Pruning:** Setting random weights to 0. Result: Sparse matrices. Hard to accelerate without Ampere Sparse Tensor Cores (2:4 sparsity).
*   **Structured Pruning:** Removing entire filters/channels. Result: A smaller dense matrix. **Universally faster.**

---

## 💻 Implementation

### 👨‍💻 Core Implementation: QAT with NVIDIA Toolkit

We will take a simple model, insert QAT modules, "fine-tune" (simulated), and export.

#### 📁 `src/qat_workflow.py`
```python
#!/usr/bin/env python3
"""
Day 25: Quantization Aware Training (QAT)
Phase 6: Platform Engineering
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

# 1. Define Model (Standard PyTorch)
# Notice: usage of quant_nn modules instead of torch.nn for layers we want quantizable
# OR: We can Monkey-Patch torch.nn
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Use Quantized Conv2d from nvidia toolkit
        self.conv1 = quant_nn.QuantConv2d(1, 32, 3, 1)
        self.relu = nn.ReLU()
        self.conv2 = quant_nn.QuantConv2d(32, 64, 3, 1)
        self.fc = quant_nn.QuantLinear(64 * 12 * 12, 10) # sizes approx
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def main():
    print("Initializing QAT Workflow...")
    
    # 2. Configuration
    # Set descriptor to use INT8
    # We can choose per-cannel or per-tensor
    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    
    # 3. Create Model
    model = SimpleCNN().cuda()
    
    # 4. Calibration Phase (Different from PTQ)
    # In QAT, we first run some data to initialize statistics (calib),
    # THEN we train.
    print("Collecting statistics (Calibration)...")
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable_calib()
            module.disable_quant() # Just collect stats first

    # Feed dummy data
    dummy_input = torch.randn(8, 1, 28, 28).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

    # Finalize calibration
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.load_calib_amd_save_params()
            module.enable_quant() # Enable Fake Quantization simulation
            module.disable_calib()
            
    print("Calibration done. Model is now simulating INT8 noise.")

    # 5. Fine-Tuning (Training Loop)
    # Standard PyTorch training...
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Fine-tuning for 1 step (Demo)...")
    output = model(dummy_input)
    loss = criterion(output, torch.tensor([1]*8).cuda())
    loss.backward()
    optimizer.step()
    
    # 6. Export to ONNX with Q-DQ Nodes
    print("Exporting ONNX...")
    model.eval()
    # Explicitly enable ONNX export compatibility
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    
    torch.onnx.export(
        model, 
        dummy_input, 
        "qat_model.onnx", 
        opset_version=13,
        input_names=['input'], 
        output_names=['output']
    )
    print("Exported qat_model.onnx with Q-DQ nodes.")
    print("This model can be compiled by TensorRT 8.0+ directly to INT8 engine.")

if __name__ == "__main__":
    main()
```

### 👨‍💻 Advanced: Structured Pruning (Channel Pruning)

A simple script to prune channels with the smallest L1 norm using `torch.nn.utils.prune`.

#### 📁 `src/pruning_demo.py`
```python
#!/usr/bin/env python3
"""
Day 25: Structured Pruning
Phase 6: Platform Engineering
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class VGGBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))

def run_pruning():
    model = VGGBlock()
    print("Original Weights Norm (Conv1):", torch.norm(model.conv1.weight))
    
    # Pruning 50% of channels (Structured) based on L1 Norm
    # dim=0 removes filters (output channels)
    prune.ln_structured(model.conv1, name="weight", amount=0.5, n=1, dim=0)
    
    # Pruning creates a mask, but weight tensor size is standard.
    # To truly shrink the model for inference, we need to remove the re-parameterization.
    prune.remove(model.conv1, 'weight')
    
    # Note: In PyTorch, 'remove' makes the pruning permanent in the tensor values (zeros them out),
    # but the Tensor shape (64, 64, 3, 3) remains 64!
    # A true speedup requires rewriting the model definition to have size (32, 64, 3, 3).
    # This is "Physical Pruning".
    
    # Check sparsity
    zero_count = torch.sum(model.conv1.weight == 0)
    total_count = model.conv1.weight.nelement()
    print(f"Sparsity: {zero_count/total_count:.2f}")
    
    print("\nWarning: PyTorch Pruning creates zeroed weights. ")
    print("To get speedup, you must physically reconstruct the model with fewer channels.")
    
    # Example of Physical Pruning reconstruction logic (Conceptual)
    # 1. Identify indices of non-zero filters.
    # 2. Create new Conv2d(32, 64, ...).
    # 3. Copy weights.
    
if __name__ == "__main__":
    run_pruning()
```

---

## 🔬 Lab Exercise: "Sparse Tensor Cores"

### Lab Objectives
1.  Target NVIDIA Ampere (A100/3090) 2:4 Sparsity.
2.  Use the `apex.contrib.sparsity` or TensorRT sparsity flags.

Ampere GPUs have a special mode where if every block of 4 elements has 2 zeros (50% sparse), it runs 2x faster.
**Exercise:**
*   Load a ResNet50.
*   Apply **2:4 Pruning** (Prune 2 smallest values in every group of 4).
*   Fine-tune.
*   Export to TensorRT.
*   Set flag `config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)`.
*   Benchmark.

---

## 📝 Daily Summary

### Key Takeaways
1.  **QAT > PTQ:** If you have the training data and pipeline, always use QAT for production INT8. The accuracy stability is superior.
2.  **Explicit Quantization:** Modern TRT relies on the ONNX graph telling it where to quantize (via Q/DQ nodes), giving the developer granular control.
3.  **Physical Pruning:** Logical pruning (masking) saves no time on GPU (GPUs multiply by zero just as slow as by one). You must change the architecture (remove channels) to see gains, unless using Ampere 2:4 hardware sparsity.
4.  **Hardware Awareness:** Unstructured sparsity (90% zeros random) is often slower than Dense on GPUs due to memory access fragmentation.

### API Summary
```python
# Nvidia Quantization Lib
quant_nn.QuantConv2d(in, out, k)
module.enable_calib()
module.enable_quant()

# PyTorch Pruning
prune.ln_structured(layer, name="weight", amount=0.5, dim=0)
```

---

**Day 25 Complete** ✅

*Next: Day 26 - NVIDIA DALI (Data Loading Library) - Fixing the CPU bottleneck in data pipelines.*
