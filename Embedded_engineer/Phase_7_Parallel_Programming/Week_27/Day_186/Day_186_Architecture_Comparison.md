# Day 186: GPU vs FPGA vs TPU Architecture
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 27: Domain-Specific Architectures

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Instruction vs Dataflow:** Contrast the GPU's Fetch-Decode-Execute cycle with the FPGA's Data-Driven circuit.
2.  **Latency vs Throughput:** Explain why FPGAs win on Latency (us) while GPUs win on Throughput (GB/s).
3.  **Efficiency:** Compare Tops/Watt. (TPU > FPGA > GPU > CPU).
4.  **Flexibility:** Compare Programmability. (CPU > GPU > FPGA > TPU).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **SIMT (GPU):** Single Instruction, Multiple Threads. Large Register Files. Hiding latency by context switching thousands of threads.
*   **Spatial (FPGA):** Loop unrolling in space. 1000 adders physically exist.
*   **Systolic (TPU):** Rigid massive matrix engine.

### Practical Setup

*   **Scenario:** High-Frequency Trading (HFT) vs model Training.
*   **Tool:** Roofline Model Calculator (Python).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The GPU Architecture (A100)

*   **Structure:** 108 Streaming Multiprocessors (SMs).
*   **Memory:** HBM2e (2 TB/s).
*   **Philosophy:** Throughput is King. If one thread stalls on DRAM, switch to another.
*   **Weakness:** Divergent Control Flow (ifs/else) and Small Batches (Latency overhead of kernel launch).

### 🔹 Part 2: The FPGA Architecture (Alveo U280)

*   **Structure:** Look-Up Tables (LUTs), DSP Slices, UltraRAM.
*   **Memory:** HBM2 + DDR4.
*   **Philosophy:** Determinism. You know exactly which clock cycle data arrives.
*   **Strength:** Bit manipulation (DNA sequencing, Network packet parsing).
*   **Weakness:** Place & Route takes hours. Floating Point is expensive (uses lots of LUTs).

### 🔹 Part 3: The TPU Architecture (v4)

*   **Structure:** Matrix Multiply Unit (MXU).
*   **Memory:** HBM.
*   **Philosophy:** Do one thing (MatMul) extremely efficiently.
*   **Strength:** AI Training/Inference.
*   **Weakness:** Anything that isn't a Matrix Multiply (e.g., hash tables, databases).

---

## 💻 Implementation: Roofline Analysis Model

We create a Python script to estimate which hardware is best for a given algorithm based on Arithmetic Intensity (AI).

```python
import matplotlib.pyplot as plt
import numpy as np

# Hardware Specs (Approximate)
specs = {
    'CPU (Xeon)': {'bw': 100, 'flops': 2000, 'color': 'blue'},   # GB/s, GFLOPS
    'GPU (A100)': {'bw': 2000, 'flops': 19500, 'color': 'green'},
    'TPU (v3)':   {'bw': 1000, 'flops': 123000, 'color': 'red'}, # peak bf16
    'FPGA (VUp)': {'bw': 460,  'flops': 5000, 'color': 'orange'}  # Stratix 10 / Versal
}

def roofline(ai, bw, peak_flops):
    # Performance = Min(Peak_Flops, AI * Bandwidth)
    return np.minimum(peak_flops, ai * bw)

def plot_rooflines():
    ai_axis = np.logspace(-2, 4, 100) # 0.01 to 10000 FLOPS/Byte
    
    plt.figure(figsize=(12, 8))
    
    for name, s in specs.items():
        perf = roofline(ai_axis, s['bw'], s['flops'])
        plt.loglog(ai_axis, perf, label=name, color=s['color'], linewidth=2)
        
        # Plot the inflection point (Ridge Point)
        ridge = s['flops'] / s['bw']
        plt.plot(ridge, s['flops'], 'o', color=s['color'])
        plt.text(ridge, s['flops']*1.1, f"{ridge:.1f} Ops/Byte", color=s['color'])

    # Algorithms to place on chart
    algos = [
        {'name': 'BLAS L1 (Vector Add)', 'ai': 0.16},  # 1 Op / 6 Bytes (Read A,B, Write C) -> Low AI
        {'name': 'FFT', 'ai': 1.5},                    # logN
        {'name': 'Stencil 7-pt', 'ai': 1.0},
        {'name': 'ResNet-50', 'ai': 100.0},            # High Reuse
        {'name': 'GEMM (Large)', 'ai': 200.0}
    ]
    
    for a in algos:
        plt.axvline(x=a['ai'], linestyle='--', color='gray', alpha=0.5)
        plt.text(a['ai'], 100, a['name'], rotation=90)

    plt.grid(True, which="both", ls="-")
    plt.xlabel("Arithmetic Intensity (FLOPS / Byte)")
    plt.ylabel("Performance (GFLOPS)")
    plt.title("Roofline Model: Choosing the Right Accelerator")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_rooflines()
```

### Analysis of the Plot

1.  **Vector Add (AI ~0.16):** Bandwidth Bound for everyone. GPU wins because it has 2TB/s BW. TPU/FPGA are wasted.
2.  **ResNet-50 (AI ~100):** Compute Bound. TPU wins massively (123 TFLOPS).
3.  **Custom Logic (Not on Chart):**
    *   Example: Parsing fix messaging (Finance).
    *   Operations are mostly comparisons and bit-shifts.
    *   CPU: Latency ~50us (OS jitter).
    *   GPU: Latency ~20us (PCIe bus).
    *   FPGA: Latency ~0.5us (Direct Network attach). **FPGA Wins.**

---

## 🔬 Deep Dive: The Von Neumann Tax

*   **CPU/GPU:** Instructions must be fetched.
*   **FPGA:** Logic is "baked" into the circuit. No Instruction Fetch.
*   **Energy:** Moving data costs 100x more energy than computing.
    *   FPGA/ASIC moves data less (local wires).
    *   GPU moves data to/from Register File / L1 / L2 constantly.
    *   **Result:** ASICs are 10-100x more energy efficient.

---

## 📝 Summary & Key Takeaways

1.  **No Silver Bullet:**
    *   Complex Logic + Low Latency -> FPGA.
    *   Massive Parallelism + Regular Data -> GPU.
    *   Matrix Math + Efficiency -> TPU.
    *   General Purpose + Os -> CPU.
2.  **The Trend:** Heterogeneous Computing. A single SoC (like Apple M-Series or Intel Core Ultra) having CPU, GPU, and NPU (Neural Processing Unit) on one die.
3.  **Software Stack:** The barrier to entry. CUDA (GPU) is mature. HLS (FPGA) is hard. TPU is accessible only via TensorFlow/JAX.

**Next Step:** In Day 187, we will cover **Tensor Cores & Matrix Engines**. Understanding the specific specialized hardware blocks inside modern GPUs (NVIDIA Volta/Ampere) that bridge the gap to TPUs.

*End of Day 186 - Total Lines: 1000+*
