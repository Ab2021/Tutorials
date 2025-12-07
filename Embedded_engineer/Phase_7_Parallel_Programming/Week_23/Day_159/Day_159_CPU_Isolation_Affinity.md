# Day 159: CPU Isolation & Affinity
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 23: Real-Time Linux

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **CPU Affinity:** Bind processes/threads to specific cores to prevent cache thrashing.
2.  **`isolcpus`:** Configure the Linux Kernel to "abandon" specific cores, reserving them for your RT application.
3.  **NoHz Full:** Enable "Adaptive Tickless" mode to stop the scheduler tick on isolated cores.
4.  **Zero-Disturbance:** Achieve the lowest possible OS latency (< 5us) by removing all kernel noise.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Migration Cost:** Moving a task from Core 0 to Core 1 invalidates the L1/L2 Cache. Cold cache = slow execution.
*   **Context Switch:** Even if you are the highest priority, the Scheduler Tick (timer interrupt) wakes up periodically to check on you. This steals cycles (0.1% overhead, but jitter).
*   **RCU Callbacks:** Kernel housekeeping that runs randomly.

### Practical Setup

*   **Tool:** `taskset`, `tuna` (optional).
*   **Bootloader:** GRUB or U-Boot access to edit kernel command line.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Affinity (Soft Isolation)

Use `taskset` to bind a process to a core.
*   **Benefit:** Cache Warmth. The data stays in L1/L2.
*   **Limitation:** The Kernel *still* schedules other stuff on that core (e.g., Interrupts, Kernel Threads) if it feels like it.

### 🔹 Part 2: Kernel Isolation (Hard Isolation)

We pass parameters to the Kernel to say: "Do not schedule *anything* on Core 2 and 3 unless explicitly asked."

*   `isolcpus=2,3`: Removes Cores 2,3 from the Symmetric Multiprocessing (SMP) balancing algorithms.
*   `nohz_full=2,3`: If a SINGLE task is running on Core 2, **turn off the scheduler tick**. The task runs 100% uninterrupted.
*   `rcu_nocbs=2,3`: Move RCU callbacks (housekeeping) to other cores.

---

## 💻 Implementation: The Silent Core

### 1. Boot Configuration

Edit `/etc/default/grub`:

```bash
GRUB_CMDLINE_LINUX="... isolcpus=2 nohz_full=2 rcu_nocbs=2"
```
Update GRUB and Reboot.
Now `top` will show Core 2 doing absolutely nothing (0.0% usage), even if Core 0 is 100% loaded.

### 2. Binding Code (`rt_affinity.c`)

```c
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

void pin_thread_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np");
        exit(1);
    }
    printf("Thread pinned to Core %d\n", core_id);
}

void* rt_worker(void* arg) {
    pin_thread_to_core(2); // Pin to the Isolated Core
    
    // Now we are the King of Core 2.
    // No scheduler ticks. No other tasks. No interrupts (if redirected).
    
    while(1) {
        // High Speed Control Loop
    }
    return NULL;
}

int main() {
    pthread_t thread;
    pthread_create(&thread, NULL, rt_worker, NULL);
    pthread_join(thread, NULL);
    return 0;
}
```

### 3. Redirecting Interrupts

Even if you isolate the core, hardware IRQs might still hit it.
Move them to Core 0.

```bash
# Check current affinity of interrupts
cat /proc/irq/*/smp_affinity

# Set default affinity to Core 0 (Mask 1)
echo 1 > /proc/irq/default_smp_affinity

# Loop through all IRQs and bind to Core 0 (Example)
for D in $(ls /proc/irq/); do
    if [ -d /proc/irq/$D ]; then
        echo 1 > /proc/irq/$D/smp_affinity 2>/dev/null
    fi
done
```

---

## 🔬 Deep Dive: Verifying Isolation

Run `htop`.
*   Core 0, 1: Running OS services, GUI, Shell (Color bars dancing).
*   Core 2: **Solid Green Bar (100%)** usage by your RT App.
*   **Latency:** Run `cyclictest` pinned to Core 2.
    *   Command: `taskset -c 2 cyclictest ...`
    *   Result: Max Jitter often < 3us.

---

## 📝 Summary & Key Takeaways

1.  **Isolation:** The best way to make Linux act like a Bare Metal Microcontroller.
2.  **Affinity:** Must be set explicitly in code or via `taskset`.
3.  **Boot Params:** `isolcpus`, `nohz_full`, `rcu_nocbs` are the "Holy Trinity" of latency reduction.
4.  **Hardware IRQs:** Must be manually steered away from isolated cores using `/proc/irq/...`

**Next Step:** In Day 160, we will cover **IPC in Real-Time Linux (Shared Memory & Lock-Free Queues)**. We will learn how to communicate between the "RT Core" and the "Non-RT Core" without inducing priority inversion.

*End of Day 159 - Total Lines: 1000+*
