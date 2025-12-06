# Day 185: The Black Screen: GPU Debugging
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 27: Advanced Troubleshooting

---

> **🎯 Focus Area:** "CUDA Error: Device-side assert triggered". This error message tells you almost nothing. To enable AI at scale, you need to debug the invisible communication between the CPU Host and the GPU Device.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Interpret** Xid Errors (Xid 31 vs Xid 79) to distinguish Hardware failure from Software bugs.
2.  **Profile** CUDA streams using **Nsight Systems (nsys)** to find CPU-GPU synchronization bottlenecks.
3.  **Detect** "Zombie Processes" holding GPU memory using `nvidia-smi` and `fuser`.
4.  **Monitor** GPU healthMetrics using **DCGM** (Data Center GPU Manager).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine with NVIDIA GPU.

### Software Environment
- `nvidia-smi`, `nsys` (NVIDIA Nsight Systems), `dcgmi`.

---

## 📖 Theoretical Foundation

### 1. The Xid Table
The NVIDIA Driver emits "Xid" codes to the Kernel Log (`dmesg`).
*   **Xid 31:** GPU Memory Page Fault. (Software bug: accessing invalid pointer).
*   **Xid 43:** Engine Reset. (Driver hung).
*   **Xid 79:** GPU has fallen off the bus. (Hardware failure / Overheat / Loose PCIe cable).
*   **Xid 63:** ECC Error (Row Remapping). (Aging capability degradation).

### 2. Synchronization
*   **Async Execution:** CPU launches kernel `MatMul<<<...>>>`. CPU returns immediately. GPU runs in background.
*   **Blocking:** `cudaDeviceSynchronize()` or `.item()`. CPU waits for GPU.
*   **Bug:** If kernel crashes asynchronously, the Error is reported at the *next* synchronization point, which might be 100 lines later.
    *   *Fix:* `CUDA_LAUNCH_BLOCKING=1`. run slowly, but report error exactly where it happened.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Analyzing Hardware Health

Check if the GPU is dying.

```bash
# 1. Check dmesg for Xid
dmesg | grep NVRM

# 2. Check ECC Errors (Error Correction Code)
nvidia-smi -q -d ECC
# Expect: "Volatile: 0, Aggregate: 0"
# If Aggregate > 0, memory bits are flipping.

# 3. Check Thermal Throttling
nvidia-smi -q -d PERFORMANCE
# Look for "SW Power Cap" or "HW Slowdown"
```

### 👨‍💻 Core Implementation: Profiling with Nsight

Why is training slow?

```bash
# 1. Run script with Nsight
# -t: trace cuda, cudnn, osrt (os runtime), nvtx (custom markers)
nsys profile -t cuda,cudnn,osrt,nvtx -o my_profile python train.py

# 2. Analyze
# Open 'my_profile.nsys-rep' in Nsight Systems GUI.
```
**Visual Analysis:**
*   **Gaps in the Timeline:** If the GPU timeline has empty spaces, the GPU is idle waiting for the CPU (Data Loading bottleneck).
*   **Overlapping Streams:** Good. Copying data (H2D) while Computing (Kernel).

### 👨‍💻 Core Implementation: DCGM Diagnostics

Run a Level 3 diagnostic on the cluster.

```bash
# 1. Run Diagnostic
dcgmi diag -r 3
# Output:
# Deployment: Pass
# Reliability: Pass
# Memory: Fail (Page 52342 stuck bit)
```

---

## 🔬 Lab Exercise: "The Ghost Process"

### Task
Clear VRAM leakage.
1.  **Scenario:** User terminates PyTorch script with `Ctrl-Z` (Suspend) instead of `Ctrl-C` (Kill).
2.  **State:** Process is stopped, but Driver Context holds 16GB VRAM.
3.  **Action:** New script fails with OOM.
4.  **Debug:**
    *   `nvidia-smi` shows process ID 1234 using 16GB.
    *   User tries `kill 1234`. Nothing happens (Process is in `T` state).
    *   User tries `kill -9 1234`. Process dies. VRAM released.
5.  **Advanced:** If `nvidia-smi` shows "No running processes" but VRAM is full?
    *   Use `fuser -v /dev/nvidia0` to find hidden PIDs.

---

## 📖 Advanced Theory: CUDA Streams
By default, PyTorch uses the Default Stream (Stream 0). Operations are serialized.
**Multi-Stream:**
*   Stream 1: Copy Batch 2 to GPU.
*   Stream 2: Compute Batch 1.
*   **Overlap:** Hides latency.
*   **Debug:** Debugging logic race conditions between streams requires `compute-sanitizer --tool racecheck`.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Environment Variables:**
    *   `CUDA_LAUNCH_BLOCKING=1`: Debug crashes.
    *   `NCCL_DEBUG=INFO`: Debug distributed hangs.
    *   `TORCH_DISTRIBUTED_DEBUG=DETAIL`: Debug DDP errors.
2.  **Zombie Hardware:** Sometimes a GPU enters a bad state where it accepts commands but never finishes. The only fix is often a full Node Reboot (cold reset).
3.  **NVLink:** Use `nvidia-smi topo -m` to see topology. If GPUs are connected via PCIe instead of NVLink, communication will be slow.

### API Summary
```bash
nvidia-smi dmon # Monitor metrics (power, temp, sm, mem) in real-time
```

---

**Day 185 Complete** ✅

*Next: Day 186 - Memory Leak Hunting - OOM.*
