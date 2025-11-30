# Day 242: High-Resolution Timers and RT Clock Sources
## Phase 2: Linux Kernel & Device Drivers | Week 36: Real-Time Systems

---

## 🎯 Learning Objectives
1. **Understand** high-resolution timers (hrtimers)
2. **Use** different clock sources (CLOCK_MONOTONIC, CLOCK_REALTIME)
3. **Implement** precise timing in RT applications
4. **Measure** timer resolution and accuracy
5. **Optimize** timer performance

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Timer Types

**Legacy Timers (jiffies-based):**
- Resolution: HZ (typically 100-1000 Hz = 1-10ms)
- Insufficient for RT applications

**High-Resolution Timers (hrtimers):**
- Resolution: nanoseconds
- Hardware-dependent (TSC, HPET, etc.)
- Essential for RT systems

### 🔹 Part 2: Clock Sources

```c
// Available clocks
CLOCK_REALTIME      // Wall clock time (can jump)
CLOCK_MONOTONIC     // Monotonic time (never jumps backward)
CLOCK_MONOTONIC_RAW // Not adjusted by NTP
CLOCK_BOOTTIME      // Includes suspend time
CLOCK_PROCESS_CPUTIME_ID  // Process CPU time
CLOCK_THREAD_CPUTIME_ID   // Thread CPU time
```

---

## 💻 Implementation Examples

### Example 1: High-Resolution Sleep

```c
#include <time.h>
#include <stdio.h>

void precise_sleep_ns(long long nanoseconds) {
    struct timespec req = {
        .tv_sec = nanoseconds / 1000000000,
        .tv_nsec = nanoseconds % 1000000000
    };
    
    // Use CLOCK_MONOTONIC for RT
    clock_nanosleep(CLOCK_MONOTONIC, 0, &req, NULL);
}

// Measure timer accuracy
void test_timer_accuracy(void) {
    struct timespec start, end;
    long long target_ns = 1000000;  // 1ms
    long long actual_ns;
    
    for (int i = 0; i < 1000; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        precise_sleep_ns(target_ns);
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        actual_ns = (end.tv_sec - start.tv_sec) * 1000000000LL +
                    (end.tv_nsec - start.tv_nsec);
        
        printf("Target: %lld ns, Actual: %lld ns, Error: %lld ns\n",
               target_ns, actual_ns, actual_ns - target_ns);
    }
}
```

### Example 2: Kernel hrtimer

```c
#include <linux/hrtimer.h>
#include <linux/ktime.h>

static struct hrtimer my_timer;
static ktime_t period;

static enum hrtimer_restart timer_callback(struct hrtimer *timer) {
    // Timer expired - do work
    pr_info("Timer fired at %lld ns\n", ktime_get_ns());
    
    // Restart timer
    hrtimer_forward_now(timer, period);
    return HRTIMER_RESTART;
}

static int __init timer_init(void) {
    // Initialize timer
    hrtimer_init(&my_timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
    my_timer.function = timer_callback;
    
    // Set period (1ms)
    period = ktime_set(0, 1000000);
    
    // Start timer
    hrtimer_start(&my_timer, period, HRTIMER_MODE_REL);
    
    return 0;
}

static void __exit timer_exit(void) {
    hrtimer_cancel(&my_timer);
}
```

### Example 3: Periodic RT Task with Deadline

```c
#include <stdio.h>
#include <time.h>
#include <sched.h>

struct periodic_task {
    long long period_ns;
    long long next_wakeup;
    long long deadline_ns;
    int missed_deadlines;
};

void init_periodic_task(struct periodic_task *task, long long period_ns) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    
    task->period_ns = period_ns;
    task->next_wakeup = now.tv_sec * 1000000000LL + now.tv_nsec;
    task->deadline_ns = period_ns;
    task->missed_deadlines = 0;
}

int wait_next_period(struct periodic_task *task) {
    struct timespec now, sleep_time;
    long long now_ns, sleep_ns;
    
    // Calculate next wakeup
    task->next_wakeup += task->period_ns;
    
    // Get current time
    clock_gettime(CLOCK_MONOTONIC, &now);
    now_ns = now.tv_sec * 1000000000LL + now.tv_nsec;
    
    // Check if deadline missed
    if (now_ns > task->next_wakeup) {
        task->missed_deadlines++;
        printf("WARNING: Deadline missed by %lld ns\n", 
               now_ns - task->next_wakeup);
        task->next_wakeup = now_ns;  // Reset
        return -1;
    }
    
    // Sleep until next period
    sleep_ns = task->next_wakeup - now_ns;
    sleep_time.tv_sec = sleep_ns / 1000000000;
    sleep_time.tv_nsec = sleep_ns % 1000000000;
    
    clock_nanosleep(CLOCK_MONOTONIC, 0, &sleep_time, NULL);
    return 0;
}

void periodic_rt_task(void) {
    struct periodic_task task;
    struct sched_param param = {.sched_priority = 80};
    
    // Setup RT scheduling
    sched_setscheduler(0, SCHED_FIFO, &param);
    
    // Initialize task (10ms period)
    init_periodic_task(&task, 10000000);
    
    for (int i = 0; i < 1000; i++) {
        // Do work
        volatile long sum = 0;
        for (long j = 0; j < 100000; j++) sum += j;
        
        // Wait for next period
        wait_next_period(&task);
        
        if (i % 100 == 0) {
            printf("Iteration %d, missed deadlines: %d\n", 
                   i, task.missed_deadlines);
        }
    }
}
```

---

## 🔬 Lab Exercises

### Lab 1: Timer Resolution Test

```bash
# Check available clock sources
cat /sys/devices/system/clocksource/clocksource0/available_clocksource

# Check current clock source
cat /sys/devices/system/clocksource/clocksource0/current_clocksource

# Change clock source (if needed)
echo tsc > /sys/devices/system/clocksource/clocksource0/current_clocksource
```

### Lab 2: Measure Clock Overhead

```c
#include <time.h>
#include <stdio.h>

void measure_clock_overhead(void) {
    struct timespec ts;
    long long start, end;
    int iterations = 1000000;
    
    start = clock_gettime(CLOCK_MONOTONIC, &ts);
    
    for (int i = 0; i < iterations; i++) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
    }
    
    end = clock_gettime(CLOCK_MONOTONIC, &ts);
    
    printf("clock_gettime() overhead: %lld ns per call\n",
           (end - start) / iterations);
}
```

---

## 🧠 Assessment

**Q:** Why use CLOCK_MONOTONIC instead of CLOCK_REALTIME for RT?
**A:** CLOCK_MONOTONIC never jumps backward (immune to NTP adjustments), providing predictable timing.

**Q:** What is the typical resolution of hrtimers?
**A:** Nanosecond resolution, limited by hardware (typically 1-100ns on modern x86).

---

## 🎓 Summary

Covered high-resolution timers, clock sources, precise timing implementation, and deadline monitoring for RT applications.

---

## 🚀 Next Steps

Day 243: RT Synchronization Primitives (Mutexes, Futexes, Priority Inheritance)

---
