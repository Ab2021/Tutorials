# Day 34: Real-Time OS (RTOS)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 5: Edge AI & Optimization

---

> **📝 Content Creator Instructions:**
> Linux is not Real-Time (usually). If a Garbage Collector runs, the robot crashes.
> - **Focus:** Hard Real-Time constraints, Preemptive Scheduling, and Priority Inversion.
> - **Code:** Simulating a Scheduler, using `PREEMPT_RT` patch on Linux.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Distinguish** between Hard Real-Time (Plane crashes if missed) and Soft Real-Time (Video lags if missed).
2.  **Explain** Rate Monotonic Scheduling (RMS) and Earliest Deadline First (EDF).
3.  **Identify** Priority Inversion scenarios and solve them with Priority Inheritance.
4.  **Configure** a Linux kernel with `PREEMPT_RT` to minimize Latency Jitter.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Linux Machine (or VM).

### Software Environment
```bash
sudo apt install rt-tests # cyclictest
# Optional: FreeRTOS Simulator for Windows/Linux
```

### Prior Knowledge
- Threading (Mutex, Semaphore).
- Interrupts ($IRQ$).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Determinism

In standard Linux, if you sleep(1ms), you might wake up in 1.1ms or 10ms (Jitter).
In RTOS (FreeRTOS, QNX), you wake up in $1.000ms \pm 1\mu s$.
*   **Why?** Standard kernels optimize for *Throughput* (Server load). RT kernels optimize for *Latency* (Robot control).

### 🔹 Part 2: Scheduling Algorithms

1.  **Rate Monotonic (RMS):** Static priority. Shorter period = Higher priority.
    *   Example: Gyro (1kHz) > Lidar (10Hz).
    *   Condition: Utilization $U < n(2^{1/n} - 1) \approx 69\%$.
2.  **Earliest Deadline First (EDF):** Dynamic priority. Job with closest deadline wins.
    *   Condition: $U \le 100\%$.

### 🔹 Part 3: Priority Inversion (The Mars Pathfinder Bug)

Scenario:
1.  **Low Task (L)** grabs Mutex M.
2.  **High Task (H)** pre-empts L, tries to grab Mutex M. H sleeps (blocked).
3.  **Medium Task (M)** pre-empts L. M runs for a long time.
4.  **Result:** H is waiting for L, but L cannot run because M is hogging CPU. H misses deadline.
5.  **Fix:** **Priority Inheritance**. When H blocks on M, temporarily boost L's priority to H.

---

## 💻 Implementation: Measuring Jitter

We will write a C program to measure how "Real-Time" our system is.

### 🛠️ Project Structure
```text
day34_rtos/
├── src/
│   ├── jitter_test.c
│   └── priority_inversion.c
└── run_tests.sh
```

### 👨‍💻 Code Implementation (`src/jitter_test.c`)

```c
#define _GNU_SOURCE
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#define ITERATIONS 1000
#define PERIOD_NS 1000000 // 1ms

// Add Timespecs
void add_ns(struct timespec *t, long ns) {
    t->tv_nsec += ns;
    if (t->tv_nsec >= 1000000000) {
        t->tv_sec++;
        t->tv_nsec -= 1000000000;
    }
}

int main() {
    // 1. Lock Memory (Prevent Paging)
    mlockall(MCL_CURRENT | MCL_FUTURE);

    // 2. Set RT Priority
    struct sched_param param;
    param.sched_priority = 80;
    if (sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
        perror("sched_setscheduler failed (Run as Sudo?)");
        return 1;
    }

    struct timespec next_fart;
    clock_gettime(CLOCK_MONOTONIC, &next_fart);

    long max_jitter = 0;

    for (int i = 0; i < ITERATIONS; i++) {
        // 3. Sleep until absolute time
        add_ns(&next_fart, PERIOD_NS);
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_fart, NULL);

        // 4. Measure Wakeup Time
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);

        long jitter = (now.tv_sec - next_fart.tv_sec) * 1000000000 + 
                      (now.tv_nsec - next_fart.tv_nsec);
        
        if (jitter > max_jitter) max_jitter = jitter;
    }

    printf("Max Jitter: %ld ns (%.3f us)\n", max_jitter, max_jitter/1000.0);
    return 0;
}
```

---

## 🔬 Lab Exercise: Stress Testing

### 1. Lab Objectives
- Run `jitter_test` on idle system.
- Run `stress --cpu 4 --io 2 --vm 2` in background.
- **Compare:**
    - Standard Kernel: Jitter spikes to 200us-1000us.
    - PREEMPT_RT Kernel: Jitter stays < 50us.

### 2. Guide to PREEMPT_RT
1.  Download kernel source.
2.  Apply `patch-x.y.z-rt.patch`.
3.  Config: `Fully Preemptible Kernel (RT)`.
4.  Build and Install.

---

## 🚀 Project: "Watchdog Safety System"

**Goal:** Implement a Watchdog Timer (WDT).
1.  **Concept:** A hardware timer counts down from 100ms.
2.  **Software:** The control loop must "Kick the Dog" (reset timer) every cycle.
3.  **Failure:** If control loop hangs (Deadlock / Infinite Loop), timer reaches 0.
4.  **Action:** Hardware Reset or Emergency Stop (Cut motor power).

**Implementation (Python pseudo):**
```python
import threading
import time
import os

def control_loop():
    while True:
        compute_control()
        kick_dog()
        time.sleep(0.01)

def watchdog_hardware_sim():
    while True:
        if (time.time() - last_kick) > 0.1:
            print("WATCHDOG BITE! KILLING MOTORS!")
            os._exit(1) # Hard crash
        time.sleep(0.001)
```

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Operation not permitted"
*   **Cause:** Changing Scheduler to `SCHED_FIFO` requires Root privileges (`CAP_SYS_NICE`).
*   **Fix:** `sudo ./program` or edit `/etc/security/limits.conf`.

#### 2. "Page Faults Latency"
*   **Cause:** Kernel moves your data to swap. Retrieving it takes milliseconds.
*   **Fix:** `mlockall()` locks your process RAM so it is never swapped out.

---

## ⚡ Optimization: CPU Affinity

Isolating cores.
*   **Idea:** Reserve Core 3 for the Control Loop ONLY. Move all OS interrupts/drivers to Cores 0-2.
*   **Cmd:** `isolcpus=3` in grub.
*   **Code:** `pthread_setaffinity_np`.
*   **Result:** Zero interruption from network packets or mouse clicks.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is Priority Inversion?
    *   **A:** When a medium priority task blocks a high priority task by preempting the low priority task holding a needed resource.
2.  **Q:** Why not run everything at Max Priority?
    *   **A:** If a high priority task loops infinitely (bug), the OS hangs completely (Mouse/Keyboard won't work).
3.  **Q:** Difference between `SCHED_FIFO` and `SCHED_RR`?
    *   **A:** FIFO runs until it yields/blocks. RR (Round Robin) has time slices for same-priority tasks.

### Challenge Task
> **Task:** Deadline Monitor.
> 1. Wrap your control function.
> 2. Measure runtime $T_{exec}$.
> 3. If $T_{exec} > T_{budget}$ for 3 consecutive cycles, switch to "Safe Mode" (simpler controller).

---

## 📚 Further Reading
- **Real-Time Linux Wiki:** kernel.org.
- **FreeRTOS Book:** Using the FreeRTOS Real Time Kernel.
- **Mars Pathfinder Bug:** "What really happened on Mars?" (Mike Jones).

---

**Day 34 Complete**
