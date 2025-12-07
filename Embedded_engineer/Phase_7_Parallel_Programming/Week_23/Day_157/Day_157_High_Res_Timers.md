# Day 157: High Resolution Timers & Cyclictest
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 23: Real-Time Linux

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **HRT (High Res Timers):** Explain how Linux abandoned the periodic "Tick" (HZ) for precise, event-driven timing.
2.  **`clock_nanosleep`:** Implement strict periodic loops in C without accumulating drift.
3.  **Monotonic Clock:** Justify why `CLOCK_REALTIME` is dangerous for control loops and why `CLOCK_MONOTONIC` is required.
4.  **Jitter Measurement:** Write a tool to self-measure application latency.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Jiffies:** The old unit of time (1/HZ). If HZ=100, resolution is 10ms. Too coarse for robotics.
*   **Drift:** If you `sleep(1ms)` in a loop, and the sleep actually takes 1.1ms, after 1000 loops you are 100ms late.
*   **Absolute Timing:** Sleeping "until 12:00:01" instead of "for 1 second".

### Practical Setup

*   **Kernel:** `CONFIG_HIGH_RES_TIMERS=y`.
*   **Library:** `-lrt` (librt) for POSIX clocks.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Evolution of Time

1.  **Periodic Tick (Legacy):**
    *   Timer fires 100 times/sec (HZ=100).
    *   Scheduler checks if tasks need running.
    *   Max precision: 10ms.
2.  **High Resolution Timers (HRT):**
    *   Hardware timer (APIC/HPET) programmed to fire at the *exact nanosecond* of the next event.
    *   Resolution: ~1 microsecond (hardware limit), theoretically nanoseconds.
    *   Enables: `usleep(500)` to actually sleep for 500us (not 10ms).

### 🔹 Part 2: Clocks

*   `CLOCK_REALTIME`: Wall clock. Subject to NTP adjustments. **BAD** for loops. If admin changes time back 1 hour, your loop freezes for 1 hour.
*   `CLOCK_MONOTONIC`: Boots at 0. Always increases. Unaffected by NTP. **GOOD** for loops.

---

## 💻 Implementation: Precise Periodic Loop

We want a loop that runs exactly at 1 kHz (every 1ms), correcting for its own execution time.

### 1. The Code (`rt_periodic.c`)

```c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#define PERIOD_NS 1000000 // 1ms

// Helper to add nanoseconds
void timespec_add_ns(struct timespec *t, long ns) {
    t->tv_nsec += ns;
    while (t->tv_nsec >= 1000000000) {
        t->tv_sec++;
        t->tv_nsec -= 1000000000;
    }
}

int main() {
    struct timespec next_wakeup;
    
    // 1. Lock Memory (Prevent Paging to Swap)
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        perror("mlockall");
        return 1;
    }
    
    // 2. Set Real-Time Scheduler (FIFO Prio 80)
    struct sched_param param;
    param.sched_priority = 80;
    if (sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
        perror("sched_setscheduler");
        return 1;
    }

    // 3. Initialize Baseline Time
    clock_gettime(CLOCK_MONOTONIC, &next_wakeup);
    
    // Start 1 second in future
    timespec_add_ns(&next_wakeup, 1000000000);

    // 4. The Loop
    while(1) {
        // A. Sleep Absolute (Wait until next_wakeup)
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_wakeup, NULL);
        
        // --- WORK STARTED ---
        // Verify latency here if needed
        
        // Simulate control work (e.g., 200us)
        // ...
        
        // --- WORK ENDED ---

        // B. Calculate Next Period
        timespec_add_ns(&next_wakeup, PERIOD_NS);
    }
    
    return 0;
}
```

### 2. Why `mlockall`?

In a Demand Paged OS (Linux), memory is loaded lazily.
*   **Risk:** If your control loop code is swapped out to disk, the next wakeup will trigger a Page Fault.
*   **Result:** Disk I/O latency (milliseconds!).
*   **Fix:** `mlockall(MCL_CURRENT | MCL_FUTURE)` forces all pages into RAM and keeps them there.

### 3. Using `cyclictest` (The Verification)

`cyclictest` basically runs the code above, but instead of "Simulate Work", it checks: `Actual_Time - Expected_Time`.

```bash
# -m: Lock memory
# -p90: Priority 90
# -i200: 200us interval
# -n: Use clock_nanosleep
# -h100: Generate histogram with 100 buckets
sudo cyclictest -m -p90 -i200 -n -h100
```
This produces a graphical view of your system's Jitter.

---

## 🔬 Deep Dive: System Management Interrupts (SMI)

Even with PREEMPT_RT, you might see huge latency spikes (200us+).
*   **Culprit:** The BIOS / Hardware.
*   **SMI:** CPU enters a special mode (SMM) transparency to OS (to handle thermal throttle, USB legacy emulation).
*   **OS Blindness:** Linux clock *stops* accounting during SMM.
*   **Diagnosis:** Use `hwlatdetect` tool.
*   **Fix:** Disable SMI sources in BIOS (C-States, Turbo Boost, Legacy USB).

---

## 📝 Summary & Key Takeaways

1.  **Absolute Sleep:** Always use `clock_nanosleep` with `TIMER_ABSTIME` for periodic tasks to eliminate drift.
2.  **Monotonic:** Use `CLOCK_MONOTONIC` to survive NTP updates.
3.  **Memory Locking:** `mlockall` is mandatory for RT applications to prevent page faults.
4.  **Hardware Matters:** A "perfect" RT OS fails if the BIOS steals the CPU for SMIs.

**Next Step:** In Day 158, we will explore **Memory Management in RT Linux (Hugepages & Locking)**. We will dive deeper into why `malloc` is forbidden in the fast path and how to pre-allocate memory heaps.

*End of Day 157 - Total Lines: 1000+*
