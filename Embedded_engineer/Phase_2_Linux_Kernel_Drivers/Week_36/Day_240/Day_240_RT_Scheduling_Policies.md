# Day 240: Real-Time Scheduling Policies (SCHED_FIFO, SCHED_RR, SCHED_DEADLINE)
## Phase 2: Linux Kernel & Device Drivers | Week 36: Real-Time Systems

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1. **Understand** different Linux scheduling policies and their use cases
2. **Implement** real-time applications using SCHED_FIFO and SCHED_RR
3. **Use** SCHED_DEADLINE for deadline-based scheduling
4. **Avoid** common pitfalls like priority inversion and starvation
5. **Measure** and optimize scheduling latency

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Linux Scheduling Policies

Linux provides several scheduling policies:

| Policy | Type | Priority Range | Preemption | Use Case |
|--------|------|----------------|------------|----------|
| **SCHED_OTHER** | Normal | Nice -20 to 19 | Yes | General purpose |
| **SCHED_BATCH** | Normal | Nice -20 to 19 | Yes | Batch processing |
| **SCHED_IDLE** | Normal | Lowest | Yes | Background tasks |
| **SCHED_FIFO** | Real-Time | 1-99 | Yes | Hard RT, no time slicing |
| **SCHED_RR** | Real-Time | 1-99 | Yes | RT with round-robin |
| **SCHED_DEADLINE** | Real-Time | N/A | Yes | Deadline-based RT |

### 🔹 Part 2: SCHED_FIFO (First-In-First-Out)

**Characteristics:**
- Runs until completion, blocks, or preempted by higher priority
- No time slicing among same priority tasks
- Highest priority always runs first
- **WARNING:** Can starve lower priority tasks!

**Algorithm:**
```
while (true) {
    task = highest_priority_runnable_task();
    run(task) until {
        task blocks (I/O, sleep, mutex)
        OR higher priority task becomes ready
    }
}
```

**Example Use Case:** Motor control loop that must run every 1ms without interruption.

### 🔹 Part 3: SCHED_RR (Round-Robin)

**Characteristics:**
- Like SCHED_FIFO but with time slicing
- Tasks at same priority share CPU in round-robin fashion
- Default time slice: 100ms (configurable)
- Better fairness than FIFO

**Algorithm:**
```
while (true) {
    task = highest_priority_runnable_task();
    run(task) for time_slice OR until {
        task blocks
        OR higher priority task becomes ready
    }
    if (time_slice_expired && same_priority_tasks_exist) {
        move_to_end_of_queue(task);
    }
}
```

**Example Use Case:** Multiple sensor processing tasks with equal importance.

### 🔹 Part 4: SCHED_DEADLINE

**Characteristics:**
- Based on Earliest Deadline First (EDF) algorithm
- Tasks specify: runtime, deadline, period
- Kernel guarantees deadline if schedulable
- Most sophisticated RT policy

**Parameters:**
- **Runtime:** Max CPU time per period
- **Deadline:** Relative deadline from period start
- **Period:** Task activation interval

**Admission Control:**
```
Sum of (runtime_i / period_i) for all tasks ≤ 1.0
```

**Example Use Case:** Video encoder with 33ms frame deadline.

---

## 💻 Implementation: Real-Time Application Examples

### Example 1: SCHED_FIFO Application

```c
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <sys/mlock.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

#define NSEC_PER_SEC 1000000000ULL
#define STACK_SIZE (8*1024*1024)

// High-precision sleep
void sleep_ns(long long ns) {
    struct timespec req = {
        .tv_sec = ns / NSEC_PER_SEC,
        .tv_nsec = ns % NSEC_PER_SEC
    };
    clock_nanosleep(CLOCK_MONOTONIC, 0, &req, NULL);
}

// Get current time in nanoseconds
long long get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * NSEC_PER_SEC + ts.tv_nsec;
}

// Setup real-time environment
int setup_rt(int priority) {
    struct sched_param param;
    
    // Lock all memory to prevent paging
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        return -1;
    }
    
    // Set SCHED_FIFO policy
    param.sched_priority = priority;
    if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
        perror("sched_setscheduler failed");
        return -1;
    }
    
    printf("RT setup: SCHED_FIFO priority %d\n", priority);
    return 0;
}

// Prefault stack to avoid page faults during RT execution
void prefault_stack(void) {
    unsigned char dummy[STACK_SIZE];
    memset(dummy, 0, STACK_SIZE);
}

// Simulated RT task: periodic sensor reading
void rt_periodic_task(long long period_ns, int iterations) {
    long long next_wakeup = get_time_ns();
    long long start, end, latency;
    long long min_latency = LLONG_MAX;
    long long max_latency = 0;
    long long sum_latency = 0;
    
    for (int i = 0; i < iterations; i++) {
        // Wait for next period
        next_wakeup += period_ns;
        long long now = get_time_ns();
        
        if (now < next_wakeup) {
            sleep_ns(next_wakeup - now);
        }
        
        // Measure wakeup latency
        start = get_time_ns();
        latency = start - next_wakeup;
        
        // Update statistics
        if (latency < min_latency) min_latency = latency;
        if (latency > max_latency) max_latency = latency;
        sum_latency += latency;
        
        // Simulate work (read sensor, process data)
        // In real application, this would be actual sensor I/O
        volatile int dummy = 0;
        for (int j = 0; j < 1000; j++) {
            dummy += j;
        }
        
        end = get_time_ns();
        
        if (i % 100 == 0) {
            printf("Iteration %d: latency %lld ns, work time %lld ns\n",
                   i, latency, end - start);
        }
    }
    
    printf("\nStatistics:\n");
    printf("  Min latency: %lld ns (%.2f us)\n", min_latency, min_latency/1000.0);
    printf("  Max latency: %lld ns (%.2f us)\n", max_latency, max_latency/1000.0);
    printf("  Avg latency: %lld ns (%.2f us)\n", sum_latency/iterations, 
           (sum_latency/iterations)/1000.0);
}

int main(int argc, char *argv[]) {
    int priority = 80;
    long long period_ms = 10;  // 10ms period
    int iterations = 1000;
    
    if (argc > 1) priority = atoi(argv[1]);
    if (argc > 2) period_ms = atoll(argv[2]);
    if (argc > 3) iterations = atoi(argv[3]);
    
    printf("Starting RT periodic task:\n");
    printf("  Priority: %d\n", priority);
    printf("  Period: %lld ms\n", period_ms);
    printf("  Iterations: %d\n", iterations);
    
    // Prefault stack
    prefault_stack();
    
    // Setup RT
    if (setup_rt(priority) != 0) {
        fprintf(stderr, "Failed to setup RT (need root?)\n");
        return 1;
    }
    
    // Run periodic task
    rt_periodic_task(period_ms * 1000000, iterations);
    
    return 0;
}
```

### Example 2: SCHED_RR Application

```c
#include <stdio.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mlock.h>
#include <unistd.h>

#define NUM_THREADS 4

void *worker_thread(void *arg) {
    int id = *(int *)arg;
    struct sched_param param;
    
    // Set SCHED_RR for this thread
    param.sched_priority = 50;  // Same priority for all
    pthread_setschedparam(pthread_self(), SCHED_RR, &param);
    
    printf("Thread %d started with SCHED_RR priority 50\n", id);
    
    // Simulate work
    for (int i = 0; i < 10; i++) {
        printf("Thread %d: iteration %d\n", id, i);
        
        // CPU-intensive work
        volatile long sum = 0;
        for (long j = 0; j < 10000000; j++) {
            sum += j;
        }
        
        // Small sleep to allow observation
        usleep(100000);  // 100ms
    }
    
    printf("Thread %d finished\n", id);
    return NULL;
}

int main(void) {
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    
    // Lock memory
    mlockall(MCL_CURRENT | MCL_FUTURE);
    
    printf("Creating %d threads with SCHED_RR...\n", NUM_THREADS);
    
    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, worker_thread, &thread_ids[i]);
    }
    
    // Wait for completion
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("All threads completed\n");
    return 0;
}
```

### Example 3: SCHED_DEADLINE Application

```c
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>
#include <linux/sched.h>

// SCHED_DEADLINE parameters structure
struct sched_attr {
    uint32_t size;
    uint32_t sched_policy;
    uint64_t sched_flags;
    int32_t sched_nice;
    uint32_t sched_priority;
    uint64_t sched_runtime;
    uint64_t sched_deadline;
    uint64_t sched_period;
};

// Syscall wrappers (not in glibc yet)
static int sched_setattr(pid_t pid, const struct sched_attr *attr, unsigned int flags) {
    return syscall(__NR_sched_setattr, pid, attr, flags);
}

static int sched_getattr(pid_t pid, struct sched_attr *attr, unsigned int size, unsigned int flags) {
    return syscall(__NR_sched_getattr, pid, attr, size, flags);
}

// Setup SCHED_DEADLINE
int setup_deadline(uint64_t runtime_ns, uint64_t deadline_ns, uint64_t period_ns) {
    struct sched_attr attr;
    
    attr.size = sizeof(attr);
    attr.sched_policy = SCHED_DEADLINE;
    attr.sched_flags = 0;
    attr.sched_nice = 0;
    attr.sched_priority = 0;
    attr.sched_runtime = runtime_ns;
    attr.sched_deadline = deadline_ns;
    attr.sched_period = period_ns;
    
    if (sched_setattr(0, &attr, 0) != 0) {
        perror("sched_setattr");
        return -1;
    }
    
    printf("SCHED_DEADLINE configured:\n");
    printf("  Runtime: %lu ns (%.2f ms)\n", runtime_ns, runtime_ns/1e6);
    printf("  Deadline: %lu ns (%.2f ms)\n", deadline_ns, deadline_ns/1e6);
    printf("  Period: %lu ns (%.2f ms)\n", period_ns, period_ns/1e6);
    
    return 0;
}

// Deadline task
void deadline_task(int iterations) {
    struct timespec start, end;
    
    for (int i = 0; i < iterations; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Simulate work (video frame processing, etc.)
        volatile long sum = 0;
        for (long j = 0; j < 5000000; j++) {
            sum += j;
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        long long elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000LL +
                              (end.tv_nsec - start.tv_nsec);
        
        printf("Iteration %d: work time %.2f ms\n", i, elapsed_ns/1e6);
        
        // Wait for next period (kernel handles this automatically)
        sched_yield();
    }
}

int main(int argc, char *argv[]) {
    // Example: 30 FPS video processing
    // Runtime: 20ms (max CPU time per frame)
    // Deadline: 33ms (must finish within frame time)
    // Period: 33ms (30 FPS = 33.33ms per frame)
    
    uint64_t runtime_ns = 20 * 1000000;   // 20ms
    uint64_t deadline_ns = 33 * 1000000;  // 33ms
    uint64_t period_ns = 33 * 1000000;    // 33ms
    
    if (argc > 1) runtime_ns = atoll(argv[1]) * 1000000;
    if (argc > 2) deadline_ns = atoll(argv[2]) * 1000000;
    if (argc > 3) period_ns = atoll(argv[3]) * 1000000;
    
    // Check admission control
    double utilization = (double)runtime_ns / period_ns;
    printf("Task utilization: %.2f%%\n", utilization * 100);
    
    if (utilization > 1.0) {
        fprintf(stderr, "Error: Utilization > 100%% (not schedulable)\n");
        return 1;
    }
    
    // Setup SCHED_DEADLINE
    if (setup_deadline(runtime_ns, deadline_ns, period_ns) != 0) {
        fprintf(stderr, "Failed to setup SCHED_DEADLINE (need root and PREEMPT_RT kernel)\n");
        return 1;
    }
    
    // Run deadline task
    deadline_task(100);
    
    return 0;
}
```

---

## 🔬 Lab Exercises

### Lab 1: Priority Inversion Demonstration

```c
#include <stdio.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void *low_priority(void *arg) {
    struct sched_param param = {.sched_priority = 10};
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    
    printf("LOW: Acquiring lock...\n");
    pthread_mutex_lock(&lock);
    printf("LOW: Lock acquired, working...\n");
    sleep(5);  // Simulate work
    pthread_mutex_unlock(&lock);
    printf("LOW: Lock released\n");
    
    return NULL;
}

void *medium_priority(void *arg) {
    struct sched_param param = {.sched_priority = 50};
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    
    sleep(1);  // Let low priority start
    printf("MEDIUM: Running (blocking high priority!)\n");
    sleep(10);  // CPU-bound work
    printf("MEDIUM: Done\n");
    
    return NULL;
}

void *high_priority(void *arg) {
    struct sched_param param = {.sched_priority = 90};
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    
    sleep(2);  // Let others start
    printf("HIGH: Trying to acquire lock...\n");
    pthread_mutex_lock(&lock);  // Will block on low priority task!
    printf("HIGH: Lock acquired (finally!)\n");
    pthread_mutex_unlock(&lock);
    
    return NULL;
}

int main(void) {
    pthread_t low, med, high;
    
    // Enable priority inheritance
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
    pthread_mutex_init(&lock, &attr);
    
    pthread_create(&low, NULL, low_priority, NULL);
    pthread_create(&med, NULL, medium_priority, NULL);
    pthread_create(&high, NULL, high_priority, NULL);
    
    pthread_join(low, NULL);
    pthread_join(med, NULL);
    pthread_join(high, NULL);
    
    return 0;
}
```

---

## 🧠 Assessment & Review

### Knowledge Check

1. **Q:** When should you use SCHED_FIFO vs SCHED_RR?
   **A:** SCHED_FIFO for tasks that must run to completion without interruption. SCHED_RR when multiple tasks at same priority need fair CPU sharing.

2. **Q:** What is the danger of SCHED_FIFO?
   **A:** A runaway SCHED_FIFO task at high priority can completely starve all other tasks, including system daemons, potentially hanging the system.

3. **Q:** How does SCHED_DEADLINE differ from SCHED_FIFO?
   **A:** SCHED_DEADLINE uses EDF algorithm and provides admission control to guarantee deadlines. SCHED_FIFO is priority-based without deadline guarantees.

4. **Q:** What is priority inheritance?
   **A:** When a low-priority task holds a resource needed by high-priority task, the low-priority task temporarily inherits the high priority to prevent priority inversion.

5. **Q:** Can SCHED_DEADLINE tasks miss deadlines?
   **A:** If admission control passes (total utilization ≤ 100%), deadlines are guaranteed. If a task exceeds its runtime budget, it may be throttled and miss its deadline.

---

## 📚 Further Reading

- Linux man pages: `sched(7)`, `sched_setscheduler(2)`, `sched_setattr(2)`
- "Deadline scheduling in the Linux kernel" by Juri Lelli et al.
- "Real-Time Linux Kernel Scheduler" documentation

---

## 🎓 Summary

Today we learned:
1. **Scheduling Policies:** SCHED_FIFO, SCHED_RR, SCHED_DEADLINE
2. **RT Application Development:** Memory locking, priority setting, latency measurement
3. **Priority Inversion:** Problem and solution (priority inheritance)
4. **SCHED_DEADLINE:** EDF algorithm, admission control, deadline guarantees

**Key Takeaway:** Choose the right scheduling policy for your RT requirements. SCHED_FIFO for simple priority-based RT, SCHED_RR for fairness, SCHED_DEADLINE for deadline guarantees.

---

## 🚀 Next Steps

Tomorrow (Day 241), we'll explore **Interrupt Handling in Real-Time Systems**, including threaded interrupts, IRQ affinity, and interrupt latency optimization.

---
