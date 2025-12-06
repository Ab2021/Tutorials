# Day 39: High-Performance Networking: RDMA, NCCL, and GPUDirect
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 6: Distributed Training & Large Scale Systems

---

> **🎯 Focus Area:** Dive deep into the plumbing of distributed AI. Understand how **NCCL** uses **GPUDirect RDMA** to bypass the CPU and move data between GPUs at 400Gbps+.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Trace** the path of a packet during `dist.all_reduce` with and without GPUDirect.
2.  **Use** environment variables (`NCCL_DEBUG`, `NCCL_P2P_DISABLE`) to debug connectivity.
3.  **Benchmark** inter-GPU bandwidth using `nccl-tests`.
4.  **Explain** the difference between Ring and Tree algorithms in NCCL.
5.  **Configure** PyTorch to utilize InfiniBand/RoCE interfaces.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Ideally Multi-Node setup with InfiniBand or 100GbE RoCE.
- Or Multi-GPU Single Node (NVLink/PCIe).

### Software Environment
```bash
# Clone NCCL Tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make MPI=0
```

### Prior Knowledge
- TCP/IP Stack basics.
- Day 36: Usage of `.init_process_group("nccl")`.

---

## 📖 Theoretical Foundation

### 1. The Data Path

**Standard TCP/IP:**
1.  GPU0 writes to Host RAM.
2.  CPU copies Host RAM to OS Kernel Buffer.
3.  NIC reads Kernel Buffer.
4.  ... Network Transmission ...
5.  Receive NIC writes to Kernel Buffer.
6.  CPU copies to User RAM.
7.  GPU1 reads User RAM.
*   **Result:** High Latency, High CPU usage.

**GPUDirect RDMA (Remote Direct Memory Access):**
1.  GPU0 memory is mapped directly to the NIC.
2.  NIC reads GPU0, sends, Receiver NIC writes GPU1.
*   **Result:** Zero Copy, CPU is idle.

### 2. NCCL Algorithms
NCCL chooses algorithms based on topology and size.
*   **Ring:** Simple. Data passes GPU0->GPU1->GPU2->GPU0. Good for Bandwidth.
*   **Tree (Double Binary Tree):** Hierarchical. Good for Latency, used in massive clusters (1000+ GPUs).

### 3. NVLink vs PCIe vs Ethernet
*   **NVLink:** 600-900 GB/s (Intra-node).
*   **PCIe Gen4:** 64 GB/s (Intra-node).
*   **Ethernet/InfiniBand:** 12-50 GB/s (Inter-node).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Parsing NCCL Debug Output

We cannot "write" NCCL (it's close sourced binary), but we can debug it.

#### 📁 `src/debug_nccl.sh`
```bash
#!/bin/bash
# Day 39: NCCL Debugging
# Phase 6: Distributed Training

# 1. Enable Debug Logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 2. Force Algorithm (Optional, for learning)
# NCCL_ALGO=Ring or Tree
# export NCCL_ALGO=Ring

# 3. Network Selection (If you have multiple interfaces)
# Only use eth0 or ib0
# export NCCL_SOCKET_IFNAME=eth0

# 4. Run PyTorch DDP Script (from Day 36)
torchrun --nproc_per_node=2 src/ddp_training.py > nccl_log.txt 2>&1

echo "Log generated at nccl_log.txt"
```

**Analyzing Output:**
Look for lines like:
```text
NCCL INFO using network interface eth0
NCCL INFO Channel 00/02 :    0   1
NCCL INFO Ring 00 : 0[0] -> 1[1] -> 0[0] comm 
```
*   **P2P:** Look for `NCCL INFO P2P via direct pointer`. This means GPUDirect P2P (NVLink/PCIe) is working.
*   **Shm:** `via shared memory`. (Slower intra-node fallback).
*   **NET:** `via Network`. (Inter-node).

### 👨‍💻 Benchmarking with `nccl-tests`

This is the industry standard tool.

#### 📁 `src/run_benchmark.sh`
```bash
#!/bin/bash
# Compile usage: make MPI=0 CUDA_HOME=/usr/local/cuda

# Run AllReduce Performance Benchmark
# -b 8M : start bytes 8MB
# -e 128M : end bytes 128MB
# -f 2 : factor 2 (8, 16, 32...)
# -g 2 : number of GPUs

./build/all_reduce_perf -b 8M -e 128M -f 2 -g 2
```

**Understanding Output:**
```text
#                                                     out-of-place                       in-place          
#       size         count      type   redop    time   algbw   busbw  error     time   algbw   busbw  error
     8388608       2097152     float     sum  150.00   55.92   55.92  0e+00   150.00   55.92   55.92  0e+00
```
*   **BusBW:** The effective speed. If this is 20 GB/s on PCIe Gen4, you are good. If it is 1 GB/s, P2P is broken.

### 👨‍💻 Python Wrapper to Benchmark `dist.all_reduce`

#### 📁 `src/dist_bench.py`
```python
#!/usr/bin/env python3
import torch
import torch.distributed as dist
import time
import os

def run_bench():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    
    # 100MB Tensor
    size_mb = 100
    num_elements = size_mb * 1024 * 1024 // 4 # float32
    tensor = torch.randn(num_elements, device='cuda')
    
    # Warmup
    for _ in range(5):
        dist.all_reduce(tensor)
        
    torch.cuda.synchronize()
    start = time.time()
    
    iters = 100
    for _ in range(iters):
        dist.all_reduce(tensor)
        
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iters
    bw = size_mb / avg_time # MB/s
    
    if rank == 0:
        print(f"Tensor Size: {size_mb} MB")
        print(f"Avg Time: {avg_time*1000:.2f} ms")
        print(f"Bandwidth: {bw / 1024:.2f} GB/s")

if __name__ == "__main__":
    run_bench()
```

---

## 🔬 Lab Exercise: "Simulating Network Failure"

### Task
1.  Disable P2P (force data through CPU/SysMem):
    `export NCCL_P2P_DISABLE=1`
2.  Run the benchmark.
3.  **Observation:** Speed drops significantly (e.g., from 20GB/s to 4GB/s). This illustrates the importance of GPUDirect.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Topology Matters:** NCCL builds a graph of your GPUs. If you put 2 GPU cards in PCIe slots controlled by different CPUs (NUMA nodes), NCCL detects the QPI/UPI link bottleneck and degrades performance.
2.  **Buffers:** NCCL works best with large buffers. Sending 4 bytes is a waste. `bucket_cap_mb` in DDP buffers small gradients into one large 25MB bucket to maximize Bandwidth.
3.  **Hanging?** If `dist.init_process_group` hangs, it's 99% a firewall/network issue blocking the Random High Port that NCCL selected. Open ports!

### API Summary
```bash
NCCL_DEBUG=INFO      # Trace connection capability
NCCL_P2P_DISABLE=1   # Force disable NVLink/PCIe P2P
NCCL_IB_DISABLE=1    # Force disable InfiniBand
```

---

**Day 39 Complete** ✅

*Next: Day 40 - DeepSpeed & Megatron-LM - The frameworks that package all these tricks (FSDP, TP, PP) into usable libraries.*
