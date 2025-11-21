# Day 24 (Part 1): Advanced Distributed Training

> **Phase**: 6 - Deep Dive
> **Topic**: Scaling to Clusters
> **Focus**: Parameter Server, All-Reduce, and Fault Tolerance
> **Reading Time**: 60 mins

---

## 1. Architectures

### 1.1 Parameter Server (PS)
*   **Roles**: Workers (Compute Gradients) and Servers (Store Weights).
*   **Flow**: Workers pull weights, compute grad, push grad. Servers update weights.
*   **Pros**: Handles sparse updates well (Embedding tables).
*   **Cons**: Bandwidth bottleneck at the Server.

### 1.2 Ring All-Reduce (NCCL)
*   **Roles**: All nodes are equal.
*   **Flow**: Nodes pass gradients to neighbor in a ring.
*   **Steps**: Scatter-Reduce + All-Gather.
*   **Pros**: Bandwidth optimal. Constant scaling.
*   **Cons**: Latency is determined by slowest node.

---

## 2. Gradient Compression

Bandwidth is the bottleneck.

### 2.1 Techniques
*   **FP16**: Send half-precision. (2x speedup).
*   **Gradient Clipping**: Not for speed, but stability.
*   **1-bit SGD**: Quantize gradients to $\pm 1$ and an error residual. (Extreme compression).

---

## 3. Tricky Interview Questions

### Q1: Synchronous vs Asynchronous SGD?
> **Answer**:
> *   **Sync**: All workers wait. Mathematically equivalent to large batch SGD. Stable. Slowest worker kills speed.
> *   **Async**: Workers update whenever ready. Fast. Stale gradients (Worker A updates weights while Worker B is computing on old weights). Noise acts as regularization but can diverge.

### Q2: How does "ZeRO" (Zero Redundancy Optimizer) work?
> **Answer**: Used in DeepSpeed.
> *   **ZeRO-1**: Shard Optimizer States across GPUs.
> *   **ZeRO-2**: Shard Gradients.
> *   **ZeRO-3**: Shard Model Parameters.
> *   Allows training 1 Trillion parameter models by using aggregate memory of cluster.

### Q3: What happens if a node fails in All-Reduce?
> **Answer**: The Ring breaks. The job hangs/crashes.
> *   **Fix**: Checkpointing. Restart job from last checkpoint on remaining nodes (Elastic Training).

---

## 4. Practical Edge Case: Ethernet vs InfiniBand
*   **Problem**: Training on AWS standard instances is slow.
*   **Reason**: TCP/IP overhead on standard Ethernet.
*   **Fix**: Use instances with **EFA (Elastic Fabric Adapter)** or InfiniBand. Uses **RDMA (Remote Direct Memory Access)** to bypass CPU and write directly to GPU memory on other node.

