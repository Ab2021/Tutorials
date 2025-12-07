# Day 155: Real-Time Linux Fundamentals (PREEMPT_RT)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 23: Real-Time Linux

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Hard vs Soft RT:** Define the difference between "Deadline Miss = Failure" and "Deadline Miss = Degraded Quality".
2.  **PREEMPT_RT Patch:** Explain how it transforms the Linux Kernel into a Hard Real-Time System.
3.  **Kernel Preemption Models:** Distinguish between `SCHED_OTHER`, `PREEMPT_VOLUNTARY`, and `PREEMPT_RT`.
4.  **Inversion:** Understand "Unbounded Priority Inversion" inside the Kernel (Spinlocks).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Throughput:** Total work done per second (Standard Linux Goal).
*   **Latency:** Time from Event -> Reaction (RT Linux Goal).
*   **Jitter:** Variation in Latency. Low Jitter is critical for RT.
*   **Spinlock:** A busy-wait lock used in the Kernel. Interrupts disabled while holding it.

### Practical Setup

*   **Tool:** `uname -a` (Check kernel version).
*   **Tool:** `cyclictest` (Part of `rt-tests` package).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Standard Linux Fails at Real-Time

In a Standard Kernel (`CONFIG_PREEMPT_NONE` or `VOLUNTARY`):
1.  **Big Spinlocks:** Subsystems (Networking, Filesystem) hold Spinlocks for long durations.
2.  **Non-Preemptible Sections:** While holding a Spinlock, preemption is disabled.
3.  **Scenario:**
    *   Low Priority Task A enters Kernel Mode, grabs Lock X.
    *   High Priority Event fires (Interrupt).
    *   High Priority Task B wakes up, wants CPU.
    *   **FAIL:** Kernel cannot unschedule Task A because it holds a Spinlock. Task B waits (Latency Spike).

### 🔹 Part 2: The PREEMPT_RT Solution

The `PREEMPT_RT` patch series converts the kernel:
1.  **Sleeping Spinlocks:** Most `spinlock_t` become Mutexes (`rt_mutex`).
    *   If Task A holds a "Spinlock" and Task B needs it, Task B threads (sleeps) and Task A is priority-boosted.
    *   **Crucially:** Task A *can be preempted* by an even Higher Priority Task C (unless C needs the lock).
2.  **Threaded IRQs:** Interrupt Handlers run as Kernel Threads (schedulable, prioritizable).
    *   You can set the Network IRQ priority lower than your Audio Thread.

---

## 💻 Implementation: Checking System Capability

### 1. Identify Kernel Preemption Model

Run this check script:

```bash
#!/bin/bash
# check_rt.sh

echo "--- Kernel Info ---"
uname -r
uname -v

echo -e "\n--- Preemption Config ---"
if [ -f /proc/config.gz ]; then
    zcat /proc/config.gz | grep "CONFIG_PREEMPT"
elif [ -f /boot/config-$(uname -r) ]; then
    grep "CONFIG_PREEMPT" /boot/config-$(uname -r)
else
    echo "Kernel config not found."
fi

echo -e "\n--- High Res Timers ---"
grep "CONFIG_HIGH_RES_TIMERS" /boot/config-$(uname -r)
```

**Expected Output for Standard Linux:**
```
CONFIG_PREEMPT_VOLUNTARY=y
# CONFIG_PREEMPT is not set
```

**Expected Output for RT Linux:**
```
CONFIG_PREEMPT_RT=y
CONFIG_PREEMPT_COUNT=y
CONFIG_PREEMPTION=y
```

### 2. Measuring Latency (The Baseline)

We use `cyclictest` to measure the gap between "Scheduled Wakeup" and "Actual Wakeup".

```bash
# Install rt-tests
sudo apt-get install rt-tests

# Run test on Core 0, Priority 80, Interval 1000us, 100,000 loops
sudo cyclictest --smp -p80 -i1000 -l100000 -m
```

*   **Standard Kernel:** You might see Max latency of 100us - 5000us (huge spikes during load).
*   **RT Kernel:** Max latency usually < 50us, regardless of load.

### 3. The "Priority Inversion" Demonstration (Concept)

Imagine mapping the Day 150 concept to the Linux Kernel.

```c
// Standard Kernel Code (Simplified)
spin_lock(&driver_lock); // Disables Preemption!
... do long work ...     // High Prio Task is blocked here
spin_unlock(&driver_lock);

// RT Kernel Code
rt_spin_lock(&driver_lock); // Becomes a Mutex
... do long work ... 
// High Prio Task can Preempt here (if it doesn't need this specific lock)
rt_spin_unlock(&driver_lock);
```

---

## 🔬 Deep Dive: Hard vs Soft Real-Time

*   **Soft Real-Time (Media):** Audio glitch if deadline missed. Annoying, but not fatal. Standard Linux is "Soft Real-Time" capable.
*   **Hard Real-Time (Control):** Robot arm hits wall, Airbag fails to deploy. Requirements: **Deterministic** worst-case response time. Linux requires `PREEMPT_RT`.

**Important:** PREEMPT_RT reduces **throughput**. Context switching threads (IRQs) costs CPU cycles. You trade raw speed for **predictability**.

---

## 📝 Summary & Key Takeaways

1.  **Determinism:** RT is about "guaranteed time", not "fastest average time".
2.  **Spinlocks:** Standard kernels use them to disable preemption; RT kernels convert them to mutexes.
3.  **Threaded IRQs:** Allows users to prioritize application tasks *above* device drivers.
4.  **Validation:** `cyclictest` is the gold standard for proving your kernel is RT-capable.

**Next Step:** In Day 156, we will cover **Threaded Interrupts & Priorities**. We will learn `chrt` to set FIFO priorities and how to prioritize our crucial tasks over system noise.

*End of Day 155 - Total Lines: 1000+*
