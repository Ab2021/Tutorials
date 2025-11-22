# Day 18: Model Parallelism
## Core Concepts & Theory

### The Memory Wall

A 175B parameter model (GPT-3) requires:
- **Weights:** 175B * 2 bytes (FP16) = 350 GB.
- **Gradients:** 350 GB.
- **Optimizer States (Adam):** 175B * 12 bytes = 2.1 TB.
- **Total:** ~2.8 TB VRAM.
An A100 has 80GB. You cannot fit the model on a single GPU.
**Solution:** Parallelism.

### 1. Data Parallelism (DP)

**Concept:** Replicate the model on every GPU. Split the data batch.
- GPU 1 processes Batch A.
- GPU 2 processes Batch B.
- **Sync:** Average gradients across all GPUs.
- **Constraint:** The *entire model* must fit on one GPU.
- **Status:** Insufficient for LLMs > 10B params.

### 2. Tensor Parallelism (TP)

**Concept:** Split the *tensors* (matrices) of each layer across GPUs.
- **Intra-layer parallelism.**
- Example: Matrix Multiplication $Y = XA$.
- Split $A$ into $[A_1, A_2]$.
- GPU 1 computes $Y_1 = X A_1$.
- GPU 2 computes $Y_2 = X A_2$.
- Concatenate $Y = [Y_1, Y_2]$.

**Communication:** Requires synchronization (All-Reduce) *within* every forward/backward pass of every layer.
**Latency:** Very high communication overhead.
**Scope:** Usually limited to GPUs within a single node (NVLink speed).

### 3. Pipeline Parallelism (PP)

**Concept:** Split the *layers* of the model across GPUs.
- **Inter-layer parallelism.**
- GPU 1 holds Layers 1-8.
- GPU 2 holds Layers 9-16.
- ...
- GPU 1 passes output to GPU 2.

**The Bubble Problem:**
- While GPU 2 is working, GPU 1 is idle (waiting for next batch).
- **Solution:** Micro-batching. Split batch into chunks. Pipeline fills up like an assembly line.
- **GPipe / 1F1B:** Scheduling algorithms to minimize bubbles.

### 4. 3D Parallelism (The Holy Grail)

Combine all three to train massive models (e.g., BLOOM, GPT-4).
- **Data Parallel:** Scale to thousands of GPUs.
- **Tensor Parallel:** Fit wide layers on a single node (8 GPUs).
- **Pipeline Parallel:** Fit deep models across multiple nodes.

**Example Configuration (BLOOM-176B):**
- 384 A100 GPUs.
- TP Size = 4 (4 GPUs per layer).
- PP Size = 12 (12 pipeline stages).
- DP Size = 8 (8 replicas of the whole setup).
- $4 \times 12 \times 8 = 384$.

### Summary of Parallelism

| Type | Splits | Communication | Constraint |
| :--- | :--- | :--- | :--- |
| **Data (DP)** | Batch | Gradients | Model fits on 1 GPU |
| **Tensor (TP)** | Weights | Activations (High Freq) | Intra-node (NVLink) |
| **Pipeline (PP)** | Layers | Activations (Low Freq) | Bubble overhead |

### Next Steps
In the Deep Dive, we will implement a toy Tensor Parallel Linear Layer and analyze the communication primitives (All-Reduce, All-Gather).
