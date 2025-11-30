# Day 188: Debugging Power Management
## Phase 2: Linux Kernel & Device Drivers | Week 28: Power Management

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
1.  **Use** Kernel Parameters (`no_console_suspend`, `initcall_debug`) to trace suspend.
2.  **Analyze** `dmesg` for "PM: suspend entry" and "PM: suspend exit" timestamps.
3.  **Identify** drivers that block suspend (timeout or error).
4.  **Use** `ftrace` to profile suspend/resume latency.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC or Board.
*   **Software Required:**
    *   `trace-cmd` (optional).
*   **Prior Knowledge:**
    *   Week 28 Content.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Suspend Path of Doom
Suspend is fragile. If one driver fails or hangs:
1.  **Freeze Phase:** User processes stopped.
2.  **Device Suspend:** Drivers called in order.
3.  **Late Suspend:** Interrupts disabled.
4.  **Noirq Suspend:** Last chance.
5.  **Arch Suspend:** CPU off.

If a driver hangs in Step 3 or 4, the system is dead silent. No console, no network.

### 🔹 Part 2: Debugging Tools
*   **no_console_suspend:** Keeps the UART/VGA console active as long as possible.
*   **pm_test:** (Covered in Day 185) Test partial suspend.
*   **initcall_debug:** Prints timing for every probe/suspend call.

---

## 💻 Implementation: The "Bad" Driver

> **Instruction:** Create a driver that intentionally hangs or fails suspend to practice debugging.

### 👨‍💻 Code Implementation

```c
static int bad_suspend(struct device *dev) {
    pr_info("BadDriver: Suspending...\n");
    
    // Simulate a hang (Infinite loop or long delay)
    // msleep(10000); // 10 seconds delay
    
    // Simulate an error
    return -EBUSY; 
}

static const struct dev_pm_ops bad_pm_ops = {
    .suspend = bad_suspend,
};
```

---

## 🔬 Lab Exercise: Lab 188.1 - Tracing Suspend

### 1. Lab Objectives
- Enable debug parameters.
- Trigger suspend.
- Identify the slow/failing driver.

### 2. Step-by-Step Guide
1.  **Edit Boot Args:**
    *   Add `no_console_suspend initcall_debug ignore_loglevel`.
    *   (On GRUB: press 'e', add to `linux` line).
2.  **Load Bad Driver:** `insmod bad_driver.ko`.
3.  **Trigger Suspend:**
    ```bash
    echo mem > /sys/power/state
    ```
4.  **Analyze Log:**
    *   Look for: `calling  bad_suspend+0x0/0x... @ ...`
    *   Look for: `initcall bad_suspend returned -16 after ... usecs`
    *   The kernel will print which driver returned the error.

---

## 🧪 Additional / Advanced Labs

### Lab 2: SleepGraph (AnalyzeSuspend)
- **Goal:** Generate a visual HTML timeline of suspend.
- **Task:**
    1.  Download `sleepgraph.py` (from Intel 01.org or kernel source `scripts/`).
    2.  Run: `sudo ./sleepgraph.py -m mem -rtcwake 15`.
    3.  View `suspend-xxxx.html`.
    4.  It shows exactly how long each driver took.

### Lab 3: RTC Wake
- **Goal:** Wake up automatically.
- **Task:**
    1.  `rtcwake -m mem -s 10`.
    2.  System sleeps for 10s then wakes. Perfect for automated testing loops.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Console dies anyway
*   **Cause:** The UART driver itself suspended.
*   **Fix:** Use `earlyprintk` or ensure the UART driver has `no_console_suspend` support.

#### 2. System reboots instead of resuming
*   **Cause:** Triple fault during resume (often memory corruption or restoring registers to wrong values).
*   **Fix:** Very hard to debug. Use `pm_trace` (stores hash in RTC register).

---

## ⚡ Optimization & Best Practices

### Async Suspend
*   Drivers can suspend in parallel!
*   `device_enable_async_suspend(dev)`.
*   Greatly speeds up S3 entry/exit.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What does `initcall_debug` do?
    *   **A:** It logs every function call made by the init/suspend core, including the function name, return value, and duration. Essential for finding bottlenecks.
2.  **Q:** How do I find which driver caused a suspend failure?
    *   **A:** Check `dmesg`. The PM core prints "Some devices failed to suspend, or early wake event detected". It usually names the device.

### Challenge Task
> **Task:** "The Profiler".
> *   Use `sleepgraph` to profile your system.
> *   Identify the slowest driver in the suspend path.
> *   (Bonus) Try to optimize it (or just explain why it's slow).

---

## 📚 Further Reading & References
- [01.org Suspend/Resume Optimization](https://01.org/suspendresume)
- [Kernel Documentation: power/s2ram.rst](https://www.kernel.org/doc/html/latest/power/s2ram.html)

---
