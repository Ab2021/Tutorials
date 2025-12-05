# Day 29: Model Compression (Quantization/Pruning)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 5: Edge AI & Optimization

---

> **📝 Content Creator Instructions:**
> A 500MB PyTorch model is useless on a 4GB RAM Raspberry Pi. We must shrink it.
> - **Focus:** Post-Training Quantization (PTQ), Quantization-Aware Training (QAT), and Structured Pruning and Knowledge Distillation.
> - **Code:** Compressing a ResNet model from FP32 to INT8 using PyTorch Quantization.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the difference between FP32, FP16, and INT8 inference and the trade-off (Speed vs Accuracy).
2.  **Apply** Post-Training Quantization (PTQ) to a generic PyTorch model.
3.  **Perform** Evaluation to measure accuracy drops caused by quantization.
4.  **Implement** Unstructured Pruning (Zeroing out weights) to reduce model size.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (CPU is fine for quantization).

### Software Environment
```bash
pip install torch torchvision numpy
```

### Prior Knowledge
- Neural Network Architecture (Conv2d, Linear).
- Floating Point arithmetic.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Compress?

Robots have constraints:
1.  **Latency:** Must run at 30Hz for real-time control.
2.  **Memory:** Jetson Nano has shared RAM (4GB). Large models cause OOM (Out of Memory).
3.  **Power:** GPU usage drains battery.

### 🔹 Part 2: Quantization (INT8)

Weights are usually stored as 32-bit Floats (FP32).
We map them to 8-bit Integers (INT8) $[-128, 127]$.
$$ Q(x) = \text{round}(x / S + Z) $$
*   $S$: Scale factor.
*   $Z$: Zero-point (offset).
*   **Result:** 4x smaller model, 4x faster math (Integer arithmetic is cheap).

### 🔹 Part 3: Pruning

Removing connections.
*   **Unstructured Pruning:** Randomly zero out 50% of weights. Good for compression, bad for speed (Sparse matrices are hard to compute efficiently on dense GPUs).
*   **Structured Pruning:** Remove entire *Channels* or *Filters*. "This 64-channel layer is now 32-channel". Great for speed up.

### 🔹 Part 4: Knowledge Distillation

Train a huge "Teacher" model (ResNet-101). Use its outputs to train a tiny "Student" model (MobileNet).
*   Student learns not just the label "Cat", but the Teacher's "soft probabilities" (e.g., "90% Cat, 9% Dog, 1% Car").
*   Result: Smaller model with higher accuracy than if trained from scratch.

---

## 💻 Implementation: Quantizing ResNet

We will use PyTorch's Eager Mode Quantization.

### 🛠️ Project Structure
```text
day29_compression/
├── src/
│   ├── model.py
│   ├── quantizer.py
│   └── evaluate.py
└── run_quantization.py
```

### 👨‍💻 Code Implementation (`src/quantizer.py`)

```python
import torch
import torch.nn as nn
import torch.quantization
from torchvision import models

def load_model():
    # Load Pretrained ResNet18
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

def quantize_model(model):
    # 1. fuse_modules
    # Fusing Conv+BN+Relu into one operator improves speed and accuracy
    # (Weights are adjusted during fusion)
    model.fuse_model() # Helper function logic omitted for brevity
    
    # 2. Assign configuration (fbgemm for x86, qnnpack for ARM)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 3. Prepare
    # Insert observers to collect statistics (Min/Max values of activations)
    torch.quantization.prepare(model, inplace=True)
    
    # 4. Calibrate (Dry Run)
    # Run a few trivial inputs so observers can see range of data
    input_tensor = torch.randn(1, 3, 224, 224)
    model(input_tensor)
    
    # 5. Convert
    # Convert FP32 weights to INT8
    torch.quantization.convert(model, inplace=True)
    
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
```

### 👨‍💻 Pruning Implementation

```python
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # L1 Unstructured Pruning
            # Remove lowest 50% of weights by magnitude
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make permanent (Remove mask, update weights)
            prune.remove(module, 'weight')
    return model
```

---

## 🔬 Lab Exercise: Benchmarking

### 1. Lab Objectives
- Measure Inference Time (Latency) of FP32 vs INT8 ResNet.
- Measure Model Size (MB).
- **Target:** < 15ms latency (Real-time 60fps), < 15MB size.

### 2. Guide
1.  Run Loop: `start = time.time(); model(x); end = time.time()`.
2.  Observed Speedup on CPU: ~2-3x faster.
3.  Observed Size Reduction: ~4x smaller (45MB -> 11MB).

---

## 🚀 Project: "Tiny Yolo"

**Goal:** Compress the YOLOv8-Nano object detector for a Raspberry Pi 4.
1.  **Export:** Use Ultralytics `yolo export format=tflite int8`.
2.  **Deploy:** Run with `tflite_runtime` interpreter.
3.  **Metric:** Achieve 10 FPS on RPi4 CPU (Un-optimized is 2 FPS).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Accuracy Drop" (Cat becomes Dog)
*   **Cause:** PTQ (Post Training Quantization) assumes weights are robust. Sometimes outliers skew the scale $S$.
*   **Fix:** **QAT (Quantization Aware Training)**. Retrain the model *simulating* quantization errors in the forward pass. The optimizer learns weights that "like" being integers.

#### 2. "Unsupported Ops"
*   **Symptom:** Quantization fails on custom layers (e.g., Mish activation).
*   **Fix:** Replace modern activations (SiLU, Mish) with ReLU. ReLU is quantization-friendly (clamp 0).

---

## ⚡ Optimization: Mixed Precision (AMP)

On GPU (Tensor Cores), use **FP16** (Half Precision).
*   INT8 is hard on GPUs (requires specific TensorRT HW).
*   FP16 is easy (`torch.cuda.amp`).
*   Result: 2x Speedup, nearly zero accuracy loss.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why does fusion (Conv+BN) help?
    *   **A:** Batch Norm is just a linear scaling. It can be mathematically merged into the Conv weights. This removes a memory access step during inference.
2.  **Q:** What is the "Zero Point"?
    *   **A:** Integers can't represent negative numbers efficiently in some formats (uint8). Zero Point shifts the range so that real value 0.0 corresponds to integer $Z$.
3.  **Q:** Does Pruning requires retraining?
    *   **A:** Usually yes. If you delete 50% of the brain, a little "Fine-tuning" helps recover lost accuracy.

### Challenge Task
> **Task:** Manual Quantization.
> 1. Take a vector `[-0.5, 0.2, 1.5]`.
> 2. Calculate Scale/Zero-Point for range `[-2, 2]` to `uint8 [0, 255]`.
> 3. Verify the rounding error.

---

## 📚 Further Reading
- **PyTorch Quantization:** Official Docs.
- **TinyML:** "Deep Learning on Microcontrollers".
- **The Lottery Ticket Hypothesis:** Frankle & Carbin (Pruning theory).

---

**Day 29 Complete**
