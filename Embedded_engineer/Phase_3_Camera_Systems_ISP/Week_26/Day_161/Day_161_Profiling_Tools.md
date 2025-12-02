# Day 161: Profiling Tools (Nsight & Perf)
## Phase 3: Camera Systems & ISP | Week 26: Performance Optimization & Power

---

## 🎯 Learning Objectives
1.  **Identify** bottlenecks in the camera pipeline (CPU vs GPU vs Memory).
2.  **Use** Linux tools (`top`, `htop`, `perf`) for CPU profiling.
3.  **Master** NVIDIA Nsight Systems for system-wide visualization (Timeline).
4.  **Analyze** GPU utilization using `tegrastats` (Jetson) or `nvidia-smi`.
5.  **Trace** GStreamer pipelines using `GST_DEBUG_DUMP_DOT_DIR`.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** NVIDIA Jetson or Linux PC.
*   **Software:** Nsight Systems (Host & Target), Graphviz.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Bottleneck Theory
*   **CPU Bound:** 100% CPU usage. GPU is idle. (e.g., OpenCV `cvtColor` on CPU).
*   **GPU Bound:** 100% GPU usage. CPU is idle. (e.g., Heavy Neural Network).
*   **Memory Bound:** CPU and GPU waiting for RAM. (e.g., Copying 4K frames).
*   **I/O Bound:** Waiting for Disk/Network.

### 🔹 Part 2: Linux Profiling Tools
*   **`top` / `htop`:** Real-time view of processes. Check Load Average.
*   **`perf`:** The Linux kernel profiler. Can sample stack traces to see *which function* is consuming CPU cycles.
    *   `perf record -g -p <PID>`
    *   `perf report`

### 🔹 Part 3: NVIDIA Nsight Systems
*   The ultimate tool for Jetson.
*   Visualizes:
    *   CPU Threads (Scheduling).
    *   GPU Kernels (CUDA).
    *   Memory Transfers (HtoD, DtoH).
    *   OS Events (Syscalls).
    *   NVTX (NVIDIA Tools Extension) markers.

---

## 💻 Implementation Examples

### Example 1: Using `tegrastats` (Jetson)

Monitor system vitals.

```bash
sudo tegrastats
# Output:
# RAM 1900/3964MB (lfb 2x4MB) CPU [5%@1479,0%@1479,0%@1479,0%@1479] EMC_FREQ 0%@1600 GR3D_FREQ 0%@921
```
*   **RAM:** Used/Total.
*   **CPU:** Usage per core @ Frequency.
*   **EMC:** External Memory Controller (RAM Bandwidth).
*   **GR3D:** GPU Usage.

### Example 2: NVTX Instrumentation (Python)

Add custom markers to your code to see them in Nsight.

```python
import nvtx

@nvtx.annotate("Process Frame", color="green")
def process_frame(frame):
    
    with nvtx.annotate("Pre-process", color="blue"):
        # Resize, Normalize...
        pass
        
    with nvtx.annotate("Inference", color="red"):
        # TensorRT...
        pass

# Run with: nsys profile -o report.qdstrm python3 app.py
```

### Example 3: GStreamer Tracing

Generate the pipeline graph.

```bash
# 1. Set Environment Variable
export GST_DEBUG_DUMP_DOT_DIR=/tmp/gst-dot/
mkdir -p $GST_DEBUG_DUMP_DOT_DIR

# 2. Run Pipeline
gst-launch-1.0 v4l2src ! ...

# 3. Convert .dot to .png
dot -Tpng /tmp/gst-dot/*.dot > pipeline.png
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Find the CPU Hog

**Objective:** Optimize a slow script.

**Steps:**
1.  Write a script that does heavy matrix multiplication in a loop using pure Python (slow).
2.  Run `htop`. See one core at 100%.
3.  Run `perf record -F 99 -g python3 slow_script.py`.
4.  Run `perf report`.
5.  **Observation:** It points to the multiplication function.
6.  **Fix:** Use NumPy. Profile again.

### Lab 2: Nsight Systems Timeline

**Objective:** Visualize concurrency.

**Steps:**
1.  Run your "Smart Camera" app from Week 47.
2.  Profile with `nsys profile --trace=cuda,nvtx,osrt python3 app.py`.
3.  Open the `.qdstrm` file in Nsight Systems (on PC).
4.  **Look for:** Gaps in GPU usage. This means the GPU is starving (waiting for CPU).
5.  **Goal:** Keep the GPU busy 100% of the time.

### Lab 3: Memory Bandwidth

**Objective:** Check EMC.

**Steps:**
1.  Run a 4K copy loop.
2.  Check `tegrastats`. Look at `EMC_FREQ`.
3.  If EMC is at max frequency (e.g., 2133 MHz) and utilization is high, you are Memory Bound.

---

## 🐛 Debugging Performance

### Debug 1: "It's slow but CPU/GPU are idle"

**Symptom:** 10 FPS. CPU 20%. GPU 10%.

**Cause:**
*   **Synchronization:** You are calling `cudaDeviceSynchronize()` or `stream.synchronize()` too often. The CPU is waiting for the GPU, and then the GPU waits for the CPU.
*   **Fix:** Use Asynchronous calls. Remove unnecessary syncs.

### Debug 2: "Perf report shows `[kernel.kallsyms]`"

**Symptom:** Top consumer is kernel code, not your app.

**Cause:**
*   High System Call overhead.
*   Reading small files? Allocating memory?
*   **Fix:** Reduce syscalls. Batch I/O. Pre-allocate memory.

---

## ⚡ Performance Optimization

### Optimization 1: CPU Affinity

*   Bind your critical thread (e.g., Inference) to a specific CPU core.
*   Prevents the OS from moving the thread between cores (Context Switch overhead + Cache Misses).
*   `taskset -c 0 python3 app.py`.

### Optimization 2: Jetson Clocks

*   By default, Jetson scales frequency to save power (DVFS).
*   For max performance, lock clocks to max.
*   `sudo jetson_clocks`.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "Sampling" in profiling?** (Interrupting the CPU N times per second to check the Instruction Pointer. Statistical approximation).
2.  **What is "Tracing"?** (Recording exact start/end times of events. Exact but high overhead).
3.  **Why is `memcpy` bad for performance?** (It consumes CPU cycles and Memory Bandwidth. Zero-Copy is better).

### Practical Challenges

1.  **Flame Graph:** Generate a Flame Graph from `perf` output to visualize the stack depth and CPU usage.
2.  **GPU Utilization:** Write a CUDA kernel that sleeps. Verify `GR3D` goes to 100%.

---

## 📚 Further Reading & Resources

### Documentation
*   **NVIDIA Nsight Systems User Guide.**
*   **Brendan Gregg's Linux Performance.**

---

## 🎓 Summary

Today we covered:
- ✅ **Bottlenecks:** CPU, GPU, Memory.
- ✅ **Perf:** The kernel microscope.
- ✅ **Nsight:** The timeline view.
- ✅ **Tegrastats:** The dashboard.
- ✅ **NVTX:** Annotating your code.

**Next:** Day 162 - Latency Optimization (Zero-Copy).

---

**Day 161 Complete** | Phase 3: Camera Systems & ISP | Week 26: Performance Optimization & Power


