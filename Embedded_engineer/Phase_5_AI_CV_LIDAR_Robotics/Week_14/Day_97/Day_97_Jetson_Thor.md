# Day 97: Jetson Thor for Humanoids
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 14: Edge AI Deployment

---

> **📝 Content Creator Instructions:**
> The Brain for the Body.
> - **Focus:** The next-gen Jetson Thor (Blackwell architecture), Transformer Engine (FP8), and running Foundation Models (GR00T) on the edge.
> - **Code:** A demo comparing standard FP16 Matrix Multiplication vs Simulated FP8 (Transformer Engine Style) to understand the speed/accuracy trade-off for LLMs.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Describe** the Jetson Thor architecture (800 TOPS, Blackwell GPU, dedicated Transformer Engine).
2.  **Explain** why Transformers need specific hardware (Memory Bandwidth dominant, K-V Cache).
3.  **Simulate** FP8 quantization (E4M3 vs E5M2) for Large Language Models.
4.  **Architect** a system where Thor handles High-Level Reasoning (VLA) while Orin/MCU handles Real-time Control.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Conceptual (Hardware not widely available yet). Simulation on Desktop GPU (RTX 4090) if available.

### Software Environment
```bash
pip install torch bitlinear # or custom quant sim
```

### Prior Knowledge
- Foundation Models (Day 90).
- Jetson Platform (Day 92).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Thor?

Humanoids need to run:
1.  **VLA Model (7B+ params):** Reasoning.
2.  **Whole Body Control (1kHz):** Safety.
3.  **Perception (4K Cameras):** Vision.
Current Orin AGX (275 TOPS) struggles with 7B models at >10 token/s while doing everything else.
**Thor (800 TOPS):** Designed specifically to run GR00T (Generalist Robot 00 Technology).

### 🔹 Part 2: The Transformer Engine

A hardware block in Blackwell/Hopper GPUs.
*   **Problem:** FP16 is too slow for 100B params. INT8 is hard to quantize for Transformers (outliers in Layernorm).
*   **Solution:** **FP8**.
    *   **E4M3:** 4 exponent, 3 mantissa (High precision, low range). Good for weights.
    *   **E5M2:** 5 exponent, 2 mantissa (High range, low precision). Good for gradients/activations.
*   **Result:** 2x throughput vs FP16, same memory usage as INT8, but easy training (no complex calibration).

### 🔹 Part 3: Architecture for Humanoids

*   **Thor:** The Cortex (Brain). Runs Linux. Planning, VLA, Speech, High-Level Vision.
*   **MCU (Microcontroller):** The Spinal Cord. Runs RTOS. EtherCAT Master, 1kHz Joint Loops, IMU integration.
*   **Connection:** PCIe or 10G Ethernet.

---

## 💻 Implementation: Simulating FP8 Compute

Since we likely don't have a Thor chip yet, we simulate the effect of FP8 quantization on a matrix multiplication (the core of Attention).

### 🛠️ Project Structure
```text
day97_thor/
├── src/
│   ├── fp8_sim.py
│   └── benchmark_llm.py
└── output/
    └── quantization_error.png
```

### 👨‍💻 FP8 Simulator (`src/fp8_sim.py`)

```python
import torch
import matplotlib.pyplot as plt

def to_fp8_sim(tensor, format="E4M3"):
    """
    Simulate FP8 by clipping and rounding.
    Note: Real HW does this in bit-level. Here we map values.
    """
    if format == "E4M3":
        # Max value ~448. Range small. Precision descent.
        max_val = 448.0
        min_val = -448.0
        # 3 bits mantissa = 2^3 = 8 steps per exponent level.
        # Simplified: Quantize to Fixed Range
        scale = max_val / 127.0 # Map to int8 range
    else: # E5M2
        max_val = 57344.0
        min_val = -57344.0
        scale = max_val / 127.0

    # Fake Quantization
    # 1. Scale
    tensor_scaled = tensor / scale
    # 2. Round
    tensor_int = torch.round(tensor_scaled).clamp(-127, 127)
    # 3. De-scale
    tensor_deq = tensor_int * scale
    
    return tensor_deq

def benchmark():
    # Create large matrices (simulate LLM layer)
    N = 4096
    A = torch.randn(N, N, device='cpu')
    B = torch.randn(N, N, device='cpu')
    
    # FP32 Reference
    C_ref = torch.matmul(A, B)
    
    # FP8 Sim
    A_fp8 = to_fp8_sim(A, "E4M3")
    B_fp8 = to_fp8_sim(B, "E4M3")
    C_fp8 = torch.matmul(A_fp8, B_fp8)
    
    # Error
    error = torch.norm(C_ref - C_fp8) / torch.norm(C_ref)
    print(f"Matrix Size: {N}x{N}")
    print(f"FP8 Approximation Error: {error:.4f} (Should be < 1%)")
    
    # Plot Distribution
    plt.hist(C_ref.flatten().numpy(), bins=100, alpha=0.5, label='FP32')
    plt.hist(C_fp8.flatten().numpy(), bins=100, alpha=0.5, label='FP8')
    plt.legend()
    plt.savefig("output/quantization_error.png")

if __name__ == "__main__":
    benchmark()
```

### 👨‍💻 LLM Bandwidth Calc (`src/benchmark_llm.py`)

Why do we need 800 TOPS?
*   Model: LLaMA-7B.
*   Weights: 7B * 2 bytes (FP16) = 14GB.
*   Token Generaton: For each token, we load 14GB.
*   Memory Bandwidth: Orin AGX ~200GB/s.
*   Max Speed: 200 / 14 = **14 tokens/sec**. (Theoretical max).
*   Thor Bandwidth: ~500GB/s? -> **35 tokens/sec**.
*   Combined with FP8 (7GB weights), we get **70 tokens/sec**. Real-time conversation!

---

## 🔬 Lab Exercise: "Context Window Stress"

### 1. Lab Objectives
- **Simulate:** The "KV Cache" growth.
- **Task:** Chat with a local LLM (Jetson-AI-Lab container).
- **Action:** Feed it a very long prompt (History of Robotics).
- **Observation:** RAM usage grows linearly with context length.
- **Fail:** Eventually OOM (Out of Memory).
- **Lesson:** Humanoids need massive RAM (Unified Memory) to remember "Where did I put the keys 5 minutes ago?". 32GB is minimum. 64GB preferred.

---

## 🚀 Project: "The Hybrid Arch"

**Goal:** Design the Compute Stack for a Humanoid.
1.  **Diagram:**
    *   **Head:** Jetson Thor. (VLA, Speech, Face Rec).
    *   **Torso:** Micro-ROS Agent (STM32/Teensy).
    *   **Comm:** Ethernet over EtherCAT? Or Shared Memory if single board?
2.  **Implementation:**
    *   Mock Node A (Thor): Publishes `cmd_vel` based on VLA.
    *   Mock Node B (MCU): Subscribes `cmd_vel`, runs Inverse Kinematics, publishes `joint_states`.
    *   Measure Latency.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Hallucinations due to Quantization"
*   **Cause:** E4M3 is aggressive. If activation outliers (values > 448) exist, they get clipped.
*   **Fix:** SmoothQuant or outlier suppression techniques before quantizing.

#### 2. "Thermal Throttling on Thor"
*   **Cause:** Running 800 TOPS generates heat (100W TDP?).
*   **Fix:** Humanoids need active cooling (Fans/Liquid). You cannot run Thor fanless inside a plastic head.

---

## ⚡ Optimization: Speculative Decoding

How to speed up LLM on Edge?
*   **Draft Model (Small):** Runs on DLA/CPU. Guesses the next 5 tokens.
*   **Verify Model (Big):** Runs on Thor GPU. Checks if guesses are right.
*   **Result:** 2-3x speedup if draft is good.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between Hopper (H100) and Blackwell (Thor)?
    *   **A:** Blackwell is the successor, featuring even better FP8 support and specifically targeted at the intersection of AI and Edge Robotics in the Thor SKU.
2.  **Q:** Why is Memory Bandwidth the bottleneck for LLMs?
    *   **A:** LLMs are "Memory Bound", not "Compute Bound" during generation (decoding). The chip spends most time waiting for weights to arrive from DRAM.
3.  **Q:** What is "Unified Memory" advantage?
    *   **A:** CPU and GPU share the same RAM. No need to copy image from CPU RAM to GPU VRAM. Critical for latency.

### Challenge Task
> **Task:** Estimate Power Budget.
> 1. Thor: 100W.
> 2. Actuators (20x): 200W avg.
> 3. Sensors: 20W.
> 4. Battery: 1kWh (e.g., small e-bike battery).
> 5. Runtime: $1000 / (100+200+20) \approx 3$ hours.

---

## 📚 Further Reading
- **Project GR00T:** NVIDIA's foundation model initiative.
- **Transformer Engine Docs:** How FP8 works under the hood.

---

**Day 97 Complete**
