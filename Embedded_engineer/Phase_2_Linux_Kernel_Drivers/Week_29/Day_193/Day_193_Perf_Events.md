# Day 193: Perf Events & Performance Analysis
## Phase 2: Linux Kernel & Device Drivers | Week 29: Kernel Debugging

---

> **📝 Content Creator Instructions:**
> This document is designed to produce **comprehensive, industry-grade educational content**. 
> - **Target Length:** The final filled document should be approximately **1000+ lines** of detailed markdown.
> - **Depth:** Do not skim over details. Explain *why*, not just *how*.
> - **Structure:** If a topic is complex, **DIVIDE IT INTO MULTIPLE PARTS** (Part 1, Part 2, etc.).
> - **Code:** Provide complete, compilable code examples, not just snippets.
> - **Visuals:** Use Mermaid diagrams for flows, architectures, and state machines.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the `perf` subsystem and PMU (Performance Monitoring Unit).
2.  **Use** `perf stat`, `perf record`, and `perf report`.
3.  **Analyze** Cache Misses, Branch Mispredictions, and CPU Cycles.
4.  **Generate** Flame Graphs to visualize hotspots.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC (Virtual Machines often lack PMU access).
*   **Software Required:**
    *   `linux-tools-generic` (`perf`).
    *   `FlameGraph` scripts (Brendan Gregg).
*   **Prior Knowledge:**
    *   Day 191 (Ftrace).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The PMU (Performance Monitoring Unit)
*   **Hardware:** Special registers in the CPU that count events (Cycles, Instructions, Cache Hits).
*   **Sampling:** Instead of counting everything, the PMU can fire an interrupt every N events (e.g., every 1M cycles). The kernel records the Instruction Pointer (IP) at that moment.
*   **Statistical Profiling:** By aggregating thousands of samples, we get a statistical map of where the CPU spends its time.

### 🔹 Part 2: Software Events
*   `perf` also hooks into Kernel Tracepoints (Scheduler, Block I/O, Kmem).
*   Allows correlating hardware behavior with OS behavior (e.g., "Why did Cache Misses spike during this syscall?").

---

## 💻 Implementation: Profiling a Driver

> **Instruction:** We will use `perf` to analyze a CPU-intensive driver function.

### 👨‍💻 Step-by-Step Guide

#### Step 1: The "Busy" Driver
Create a driver with a busy loop (don't do this in production!).
```c
static ssize_t my_write(struct file *f, const char __user *buf, size_t len, loff_t *off) {
    unsigned long i;
    volatile int k = 0;
    
    // Burn CPU
    for (i = 0; i < 100000000; i++) {
        k = k + 1;
    }
    return len;
}
```

#### Step 2: Perf Stat (Counting)
Run a workload and count events.
```bash
perf stat -e cycles,instructions,cache-misses dd if=/dev/zero of=/dev/my_device count=1
```
**Output:**
```
       300,000,000      cycles
       100,000,000      instructions              # 0.33  insn per cycle
             1,500      cache-misses
```
*   **Interpretation:** IPC (Instructions Per Cycle) is 0.33. Low! (Ideally > 1). Likely stalled on dependencies or branch prediction.

#### Step 3: Perf Record (Sampling)
Capture samples to find *where* the time is spent.
```bash
perf record -g dd if=/dev/zero of=/dev/my_device count=1
```
*   `-g`: Capture Call Graph (Stack traces).

#### Step 4: Perf Report
Analyze the data.
```bash
perf report
```
**Output (TUI):**
```
  95.00%  dd       my_driver.ko     [k] my_write
   2.00%  dd       [kernel.vmlinux] [k] copy_user_enhanced_fast_string
```
*   It points directly to `my_write`.

---

## 🔬 Lab Exercise: Lab 193.1 - Flame Graphs

### 1. Lab Objectives
- Generate a Flame Graph for the entire system.
- Identify the tallest "tower" (deepest stack) and widest bar (most CPU time).

### 2. Step-by-Step Guide
1.  **Clone Repo:** `git clone https://github.com/brendangregg/FlameGraph`.
2.  **Record:**
    ```bash
    perf record -F 99 -a -g -- sleep 10
    ```
    *   `-F 99`: 99 Hz sampling frequency.
    *   `-a`: All CPUs.
    *   `-- sleep 10`: Run for 10 seconds.
3.  **Process:**
    ```bash
    perf script | ./FlameGraph/stackcollapse-perf.pl > out.perf-folded
    ./FlameGraph/flamegraph.pl out.perf-folded > perf.svg
    ```
4.  **View:** Open `perf.svg` in a browser.
5.  **Analyze:**
    *   **X-axis:** Population (CPU usage).
    *   **Y-axis:** Stack depth.
    *   Look for wide plateaus.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Cache Miss Analysis
- **Goal:** Optimize memory access.
- **Task:**
    1.  Write a program that accesses a large array sequentially (Row-major).
    2.  Profile with `perf stat -e cache-misses`.
    3.  Change to Column-major (Stride access).
    4.  Profile again. Misses should skyrocket.

### Lab 3: Tracepoints
- **Goal:** Trace Scheduler Latency.
- **Task:**
    1.  `perf record -e sched:sched_switch -e sched:sched_wakeup -a sleep 5`.
    2.  `perf script`.
    3.  See every context switch.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "PMU Hardware not found"
*   **Cause:** Running in VirtualBox/VMware without VT-x/AMD-V PMU passthrough.
*   **Fix:** Enable "Virtualize CPU Performance Counters" in VM settings. Or use `perf` software events only.

#### 2. "Permission denied"
*   **Cause:** `perf_event_paranoid` setting.
*   **Fix:** `echo -1 > /proc/sys/kernel/perf_event_paranoid`.

---

## ⚡ Optimization & Best Practices

### Annotate
*   Inside `perf report`, press `a` on a function symbol.
*   It shows the **Assembly Code** with percentages next to each instruction!
*   You can see exactly which `MOV` or `ADD` is taking 50% of the time.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `perf record` and `perf stat`?
    *   **A:** `stat` counts totals (low overhead). `record` saves individual samples to disk (high overhead) for post-processing.
2.  **Q:** Why is sampling at 99Hz better than 100Hz?
    *   **A:** To avoid "Lockstep" with periodic events. If a timer fires exactly at 100Hz, sampling at 100Hz might always catch the timer handler and miss the actual workload.

### Challenge Task
> **Task:** "The Optimizer".
> *   Take the "Bad Driver" from Day 188.
> *   Profile it.
> *   Optimize the loop (e.g., remove `volatile`).
> *   Prove the improvement using `perf stat`.

---

## 📚 Further Reading & References
- [Brendan Gregg's Perf Examples](https://www.brendangregg.com/perf.html)
- [Kernel Documentation: admin-guide/perf-security.rst](https://www.kernel.org/doc/html/latest/admin-guide/perf-security.html)

---
