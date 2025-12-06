# Day 183: X-Ray Vision: Tracing Systems with eBPF
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 27: Advanced Troubleshooting

---

> **🎯 Focus Area:** Your Python script hangs. `strace` slows it down by 100x. `print()` statements are useless because the hang happens inside the CUDA Driver. **eBPF (Extended Berkeley Packet Filter)** lets you trace Kernel and User functions with near-zero overhead.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** why eBPF is safer and faster than kernel modules or `ptrace`.
2.  **Use** standard BCC tools (`execsnoop`, `opensnoop`, `biolatency`) to debug platform issues.
3.  **Write** a custom eBPF program (BCC/Python) to trace PyTorch C++ function latency.
4.  **Visualize** Off-CPU profiling to see *why* a thread is sleeping.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Linux Machine (Kernel 5.4+). (Windows WSL2 works with some limitations).

### Software Environment
- `apt-get install bpfcc-tools linux-headers-$(uname -r)`.
- `pip install bcc`.

---

## 📖 Theoretical Foundation

### 1. The Probing Problem
*   **Logging:** You have to recompile code.
*   **Debugger (GDB):** Stops the process. Changes timing.
*   **eBPF:** Runs a sandbox VM *inside* the Kernel.
    *   **Kprobes:** Trace Kernel Functions (`sys_read`).
    *   **Uprobes:** Trace User Functions (`libtorch.so:forward`).
    *   **Tracepoints:** Static hooks defined by kernel devs.

### 2. High Level Workflow
1.  Write C code (the probe).
2.  Python script compiles it (LLVM/Clang) and loads it into Kernel.
3.  Kernel verifies it (Safety check).
4.  Probe attaches to event.
5.  Event fires -> Probe runs -> Writes to Map.
6.  Python script reads Map -> Prints stats.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Using Standard Tools

Debug a "Slow Data Loader".

```bash
# 1. Which files are being opened?
# (Check if it's opening 1M tiny files instead of 1 big file)
opensnoop -n python

# 2. Is disk I/O slow?
# Show histogram of disk latency
biolatency -m

# 3. Is the CPU Scheduler blocking us?
runqlat -m
```

### 👨‍💻 Core Implementation: Custom PyTorch Tracer

Trace how long `torch::Tensor::matmul` actually takes at the C++ level.

#### 📁 `src/trace_matmul.py`
```python
from bcc import BPF
import os

# 1. eBPF C Code
# Defines a struct to store start time
# Hooks entry and return of function
bpf_text = """
#include <uapi/linux/ptrace.h>

struct key_t {
    u32 pid;
    u32 tid;
};

BPF_HASH(start, struct key_t);

int probe_entry(struct pt_regs *ctx) {
    struct key_t key = {};
    key.pid = bpf_get_current_pid_tgid() >> 32;
    key.tid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    start.update(&key, &ts);
    return 0;
}

int probe_return(struct pt_regs *ctx) {
    struct key_t key = {};
    key.pid = bpf_get_current_pid_tgid() >> 32;
    key.tid = bpf_get_current_pid_tgid();
    
    u64 *tsp = start.lookup(&key);
    if (tsp != 0) {
        u64 delta = bpf_ktime_get_ns() - *tsp;
        bpf_trace_printk("Matmul took %d ns\\n", delta);
        start.delete(&key);
    }
    return 0;
}
"""

# 2. Load BPF
b = BPF(text=bpf_text)

# 3. Attach Uprobes
# Needs path to libtorch_cpu.so (or cuda)
library_path = "/usr/local/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so"
# Symbol might be mangled (use 'nm' to find exact Name)
# Example: _ZNK2at6Tensor6matmulERKS0_
symbol = "at::Tensor::matmul" 

try:
    b.attach_uprobe(name=library_path, sym=symbol, fn_name="probe_entry")
    b.attach_uretprobe(name=library_path, sym=symbol, fn_name="probe_return")
except Exception as e:
    print(f"Failed to attach: {e}")
    print("Hint: Use 'nm -C libtorch_cpu.so | grep matmul' to find symbol.")
    exit(1)

print("Tracing... Ctrl-C to stop.")

# 4. Print Loop
while True:
    try:
        (task, pid, cpu, flags, ts, msg) = b.trace_fields()
        print(f"PID {pid}: {msg.decode('utf-8')}")
    except KeyboardInterrupt:
        break
```

---

## 🔬 Lab Exercise: "The GIL Mystery"

### Task
Diagnose Python Global Interpreter Lock contention.
1.  **Scenario:** Multi-threaded Python Data Loader. CPU usage is low, but throughput is low.
2.  **Tool:** `pystacks` (or eBPF GIL tracer).
3.  **Action:** Trace `take_gil` and `drop_gil` symbols in `libpython`.
4.  **Observation:** Thread A holds GIL for 50ms (copying tensor). Thread B waits.
5.  **Insight:** Python threading is not parallel for CPU work.
6.  **Fix:** Move to `multiprocessing` (Process-based parallelism) to bypass GIL.

---

## 📖 Advanced Theory: Off-CPU Profiling
Standard Profilers (perf) sample the CPU. If your process is sleeping (Waiting for Network/Disk/Lock), `perf` sees nothing.
**Off-CPU Profiling:**
*   Traps `context_switch`.
*   Records stack trace when process is *scheduled out*.
*   Records duration until *scheduled in*.
*   **Result:** FlameGraph showing "We spent 40% of time waiting for `mutex_lock`".

---

## 📝 Daily Summary

### Key Takeaways
1.  **Observability without Code Change:** eBPF allows you to debug a binary running in Prod without restarting it and without adding `printf`.
2.  **Overhead:** Context switching to eBPF VM is cheap, but not zero. Don't trace a function called 1 million times/sec (like `malloc`) unless you aggregate in kernel (Maps/Histograms). `bpf_trace_printk` is slow; use Maps for production.
3.  **Permissions:** eBPF requires `root` (CAP_SYS_ADMIN). In K8s, the Pod needs `privileged: true` or specific capabilities.

### API Summary
```python
b.attach_uprobe(name="c", sym="malloc", fn_name="count_malloc")
```

---

**Day 183 Complete** ✅

*Next: Day 184 - Traffic Patterns: Network Debugging.*
