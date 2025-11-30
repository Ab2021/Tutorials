# Day 191: Ftrace (Function Tracer)
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
1.  **Understand** how Ftrace works (mcount, nop patching).
2.  **Use** `trace-cmd` to record and analyze kernel events.
3.  **Filter** tracing to specific functions or modules.
4.  **Visualize** call graphs (`function_graph` tracer).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   `trace-cmd` (`sudo apt install trace-cmd`).
    *   Kernel with `CONFIG_FUNCTION_TRACER`.
*   **Prior Knowledge:**
    *   Day 190.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is Ftrace?
*   **Built-in:** It's part of the kernel, no external modules needed.
*   **Mechanism:** The compiler adds a call to `mcount` (or `__fentry__`) at the start of *every* function.
*   **Runtime Patching:** By default, the kernel replaces these calls with `NOP` (No Operation) instructions so there is zero overhead.
*   **Activation:** When you enable Ftrace, the kernel dynamically patches the code to call the tracer.

### 🔹 Part 2: The `tracefs` Filesystem
Located at `/sys/kernel/tracing` (or `/sys/kernel/debug/tracing`).
*   `current_tracer`: function, function_graph, nop.
*   `tracing_on`: 1 or 0.
*   `trace`: The output buffer.
*   `set_ftrace_filter`: Limit to specific functions.

---

## 💻 Implementation: Using Ftrace Manually

> **Instruction:** We will trace a specific kernel function (e.g., `do_sys_open`) without using external tools.

### 👨‍💻 Step-by-Step Guide

#### Step 1: Setup
```bash
cd /sys/kernel/tracing
echo 0 > tracing_on
echo function_graph > current_tracer
```

#### Step 2: Filter
```bash
echo do_sys_open > set_ftrace_filter
```

#### Step 3: Capture
```bash
echo 1 > tracing_on
# Run some command, e.g., cat /etc/passwd
echo 0 > tracing_on
```

#### Step 4: View
```bash
cat trace | head -n 20
```
**Output:**
```
 0)               |  do_sys_open() {
 0)               |    getname() {
 0)               |      getname_flags() {
 0)   0.543 us    |        kmem_cache_alloc();
 ...
```

---

## 🔬 Lab Exercise: Lab 191.1 - Using trace-cmd

### 1. Lab Objectives
- Use the userspace tool `trace-cmd` (easier than raw sysfs).
- Record the probe sequence of a driver.

### 2. Step-by-Step Guide
1.  **Unload Driver:** `rmmod my_driver`.
2.  **Record:**
    ```bash
    trace-cmd record -p function_graph -g my_probe_function -F insmod my_driver.ko
    ```
    *   `-p function_graph`: Use graph tracer.
    *   `-g my_probe_function`: Start tracing when this function is called.
    *   `-F`: Filter only this process.
3.  **Analyze:**
    ```bash
    trace-cmd report
    ```
4.  **Result:** You will see exactly what your probe function called (e.g., `kmalloc`, `gpio_request`, etc.) and how long each took.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Tracing Latency (Wakeup)
- **Goal:** Find what causes scheduling latency.
- **Task:**
    1.  `echo wakeup > current_tracer`.
    2.  `echo 1 > tracing_on`.
    3.  Wait.
    4.  `cat trace`.
    5.  It captures the longest latency event (time between wakeup and actual execution).

### Lab 3: KernelShark
- **Goal:** GUI Visualization.
- **Task:**
    1.  Run `trace-cmd record -e sched_switch -e sched_wakeup sleep 5`.
    2.  Run `kernelshark trace.dat`.
    3.  View the timeline of tasks switching on CPUs.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Device or resource busy"
*   **Cause:** Another tracer (like `perf` or another `trace-cmd` instance) is using the system.
*   **Fix:** Ensure no other tracing tools are running.

#### 2. Trace buffer overflow
*   **Cause:** Too many events.
*   **Fix:** Increase buffer size: `echo 10240 > buffer_size_kb`.

---

## ⚡ Optimization & Best Practices

### `trace_printk`
*   Faster than `printk`. Writes to the trace buffer, not the console.
*   Use it in interrupt handlers or critical sections where `printk` is too slow.
*   View in `cat trace`.
*   **Warning:** Do not leave in production code (it adds a big warning banner to the kernel log).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `function` and `function_graph` tracer?
    *   **A:** `function` only records the entry of functions (who called whom). `function_graph` records entry and exit, allowing it to calculate duration and draw a nested hierarchy.
2.  **Q:** How does Ftrace handle overhead?
    *   **A:** Dynamic patching. When not in use, the instructions are NOPs. When enabled, only the specific functions in the filter are patched to call the tracer.

### Challenge Task
> **Task:** "The Boot Tracer".
> *   Add `trace_event=sched:sched_switch ftrace=function_graph` to the kernel command line.
> *   Boot the system.
> *   Analyze the trace to see what happened during early boot before userspace started.

---

## 📚 Further Reading & References
- [Kernel Documentation: trace/ftrace.rst](https://www.kernel.org/doc/html/latest/trace/ftrace.html)
- [LWN: Ftrace Series](https://lwn.net/Articles/365835/)

---
