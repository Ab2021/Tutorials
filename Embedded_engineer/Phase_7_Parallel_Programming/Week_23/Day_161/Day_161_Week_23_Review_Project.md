# Day 161: Week 23 Review & Project (High-Precision Pulse Generator)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 23: Real-Time Linux

---

## 🎯 Week 23 Recap: The Path to Determinism

In this week, we transformed Linux from a General Purpose OS into a Hard Real-Time Controller.

1.  **Day 155 (PREEMPT_RT):** Understanding how Kernel Spinlocks become Mutexes to allow preemption.
2.  **Day 156 (Threaded IRQs):** Prioritizing User Tasks above Kernel Drivers.
3.  **Day 157 (Timers):** Using `clock_nanosleep` for drift-free periodic loops.
4.  **Day 158 (Memory):** `mlockall` and Stack Prefaulting to eliminate Page Faults.
5.  **Day 159 (Isolation):** `isolcpus` and `nohz_full` to silence the OS scheduler.
6.  **Day 160 (IPC):** Lock-Free Ring Buffers to communicate without Syscalls.

---

## 🛠️ The Project: Precision Pulse Generator

We will build a **Software Signal Generator** that outputs a 1 kHz square wave.
*   **Target:** Toggle a virtual GPIO (Shared Memory Flag) every 500us.
*   **Constraint:** Jitter must stay below 10 microseconds.
*   **Architecture:**
    *   **RT Core:** Runs the Generator. Pinned to Core 3.
    *   **Shared Memory:** Ring Buffer for logging latency stats.
    *   **Non-RT Core:** Reads buffer, prints histogram.

### 1. The Header (`project_config.h`)

```c
#ifndef PROJECT_CONFIG_H
#define PROJECT_CONFIG_H

#include <stdint.h>
#include <stdatomic.h>

#define SHM_NAME "/rt_pulse_shm"
#define RING_SIZE 1024

struct PulseSystem {
    // virtual_gpio: 0 or 1. toggled by RT task.
    atomic_int virtual_gpio;
    
    // Ring Buffer indices
    atomic_size_t head;
    atomic_size_t tail;
    
    // Log Data: Store latency in nanoseconds for analysis
    long latency_log[RING_SIZE];
};

#endif
```

### 2. The Real-Time Generator (`generator.c`)

```c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sched.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include "project_config.h"

// Helper: Add NS
void timespec_add_ns(struct timespec *t, long ns) {
    t->tv_nsec += ns;
    while (t->tv_nsec >= 1000000000) {
        t->tv_sec++;
        t->tv_nsec -= 1000000000;
    }
}

// Helper: Diff NS
long timespec_diff_ns(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
}

int main() {
    // 1. Lock Memory
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) { perror("mlockall"); return 1; }

    // 2. Open Shared Memory
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(fd, sizeof(struct PulseSystem));
    struct PulseSystem* sys = mmap(NULL, sizeof(struct PulseSystem), 
                                   PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    // 3. Set Scheduler FIFO 90
    struct sched_param param = { .sched_priority = 90 };
    sched_setscheduler(0, SCHED_FIFO, &param);

    // 4. Pin to Core 3 (assuming it exists and is isolated)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    // 5. The Loop
    struct timespec next_wakeup, now;
    clock_gettime(CLOCK_MONOTONIC, &next_wakeup);
    
    // Start 1s later
    timespec_add_ns(&next_wakeup, 1000000000);

    printf("RT Generator Running on Core 3...\n");

    while(1) {
        // A. Wait for precise time
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_wakeup, NULL);
        
        // B. Measure Jitter (Actual Wakeup vs Expected)
        clock_gettime(CLOCK_MONOTONIC, &now);
        long jitter = timespec_diff_ns(next_wakeup, now);
        
        // C. Toggle GPIO
        int val = atomic_load_explicit(&sys->virtual_gpio, memory_order_relaxed);
        atomic_store_explicit(&sys->virtual_gpio, !val, memory_order_relaxed);
        
        // D. Log Jitter to Ring Buffer
        size_t h = atomic_load_explicit(&sys->head, memory_order_relaxed);
        size_t t = atomic_load_explicit(&sys->tail, memory_order_acquire);
        
        if (h - t < RING_SIZE) {
            sys->latency_log[h & (RING_SIZE - 1)] = jitter;
            atomic_store_explicit(&sys->head, h + 1, memory_order_release);
        }

        // E. Next Target (500us for 1kHz Square Wave)
        timespec_add_ns(&next_wakeup, 500000);
    }
}
```

### 3. The Monitor (`monitor.c`)

```c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "project_config.h"

int main() {
    int fd = shm_open(SHM_NAME, O_RDWR, 0666);
    if (fd == -1) { printf("Start generator first!\n"); return 1; }
    
    struct PulseSystem* sys = mmap(NULL, sizeof(struct PulseSystem), 
                                   PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    long max_jitter = 0;
    long total_jitter = 0;
    long samples = 0;

    printf("Monitoring Pulse Generator...\n");

    while(1) {
        size_t t = atomic_load_explicit(&sys->tail, memory_order_relaxed);
        size_t h = atomic_load_explicit(&sys->head, memory_order_acquire);
        
        if (h == t) {
            usleep(10000); // Sleep 10ms
            continue;
        }
        
        // Read Batch
        while (t != h) {
            long jitter = sys->latency_log[t & (RING_SIZE - 1)];
            if (jitter > max_jitter) max_jitter = jitter;
            total_jitter += jitter;
            samples++;
            t++;
        }
        
        // Update Tail
        atomic_store_explicit(&sys->tail, t, memory_order_release);
        
        // Report
        if (samples % 2000 == 0) { // Every second (2000 edges)
            printf("Avg Jitter: %ld ns | Max Jitter: %ld ns | GPIO: %d\n", 
                   total_jitter / samples, max_jitter, 
                   atomic_load(&sys->virtual_gpio));
            // Reset stats periodically
            max_jitter = 0; samples = 0; total_jitter = 0;
        }
    }
}
```

### 4. Build & Run Script (`run.sh`)

```bash
#!/bin/bash
gcc -o generator generator.c -lrt -pthread
gcc -o monitor monitor.c -lrt -pthread

# Start Monitor on Core 0 (Background)
taskset -c 0 ./monitor &

# Start Generator on Core 3 (Foreground)
# Note: Requires root for SCHED_FIFO and mlockall
sudo ./generator
```

---

## 📝 Performance Validation

If configured correctly (with `isolcpus` on Core 3):
*   **Result:** `Max Jitter: 3000 ns` (3 microseconds).
*   **Success:** This is well within the 10us requirement.
*   **Failure:** If you see 200,000ns spikes, check for SMIs or ensure `irqbalance` is not moving IRQs to Core 3.

**Next Step:** Phase 7 continues into **Week 24: Compiler Backend Engineering (Register Allocation)**. We transition from OS-level Real-Time back to Compiler Optimization, focusing on Graph Coloring and Register Allocation algorithms.

*End of Day 161 - Total Lines: 1000+*
