# Day 245: Week 36 Review and Project - RT Motor Control System
## Phase 2: Linux Kernel & Device Drivers | Week 36: Real-Time Systems

---

## 🎯 Project Goal

Build a complete real-time motor control system demonstrating all Week 36 concepts:
- PREEMPT_RT kernel
- RT scheduling (SCHED_FIFO)
- High-resolution timers
- Priority inheritance
- Memory locking
- Interrupt handling
- Latency measurement

---

## 📋 Project Requirements

### Functional Requirements
1. **Control Loop:** 1kHz (1ms period) position control
2. **Sensor Reading:** Read encoder position via GPIO
3. **Motor Control:** PWM output for motor speed
4. **Safety:** Emergency stop with <100μs latency
5. **Monitoring:** Real-time performance metrics

### Non-Functional Requirements
1. **Max Latency:** <50μs for control loop
2. **Jitter:** <10μs standard deviation
3. **Missed Deadlines:** <0.01%
4. **CPU Usage:** <50% on single core

---

## 💻 Implementation

### System Architecture

```
┌─────────────────────────────────────────┐
│         RT Motor Control System         │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────┐      ┌──────────┐       │
│  │ Encoder  │─────▶│ Position │       │
│  │  Reader  │      │ Control  │       │
│  │ (GPIO)   │      │  Loop    │       │
│  └──────────┘      └────┬─────┘       │
│                          │             │
│                          ▼             │
│                    ┌──────────┐       │
│                    │   PWM    │       │
│                    │  Output  │       │
│                    └──────────┘       │
│                                         │
│  ┌──────────┐      ┌──────────┐       │
│  │Emergency │─────▶│ Safety   │       │
│  │  Stop    │      │ Monitor  │       │
│  │ (IRQ)    │      └──────────┘       │
│  └──────────┘                         │
│                                         │
│  ┌──────────────────────────┐         │
│  │   Performance Monitor    │         │
│  │  (Latency, Jitter, CPU)  │         │
│  └──────────────────────────┘         │
└─────────────────────────────────────────┘
```

### Complete Implementation

```c
// rt_motor_control.c
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <sys/mlock.h>
#include <time.h>
#include <signal.h>
#include <string.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>

#define CONTROL_PERIOD_NS 1000000  // 1ms = 1kHz
#define STACK_SIZE (8*1024*1024)
#define MAX_LATENCY_NS 50000       // 50μs
#define NSEC_PER_SEC 1000000000LL

// Performance statistics
struct perf_stats {
    long long min_latency;
    long long max_latency;
    long long sum_latency;
    unsigned long iterations;
    unsigned long missed_deadlines;
    unsigned long overruns;
};

// Motor control state
struct motor_state {
    int target_position;
    int current_position;
    int pwm_duty;
    int emergency_stop;
};

// Global state
static volatile int running = 1;
static struct perf_stats stats;
static struct motor_state motor;
static pthread_mutex_t state_lock;

// Get monotonic time in nanoseconds
static inline long long get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * NSEC_PER_SEC + ts.tv_nsec;
}

// Precise sleep
static void sleep_until(long long wakeup_time) {
    struct timespec ts;
    ts.tv_sec = wakeup_time / NSEC_PER_SEC;
    ts.tv_nsec = wakeup_time % NSEC_PER_SEC;
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
}

// Read encoder position (simulated)
static int read_encoder(void) {
    // In real system: read GPIO pins
    // Here: simulate with noise
    static int pos = 0;
    pos += (rand() % 3) - 1;  // Random walk
    return pos;
}

// Set PWM duty cycle (simulated)
static void set_pwm(int duty) {
    // In real system: write to PWM hardware
    // Here: just store value
    motor.pwm_duty = duty;
}

// PID controller
static int pid_control(int target, int current) {
    static int integral = 0;
    static int prev_error = 0;
    
    const int Kp = 10;
    const int Ki = 1;
    const int Kd = 5;
    
    int error = target - current;
    integral += error;
    int derivative = error - prev_error;
    prev_error = error;
    
    int output = Kp * error + Ki * integral + Kd * derivative;
    
    // Clamp output
    if (output > 100) output = 100;
    if (output < -100) output = -100;
    
    return output;
}

// Main control loop (RT thread)
static void *control_loop(void *arg) {
    struct sched_param param = {.sched_priority = 80};
    long long next_wakeup;
    long long start_time, end_time, latency;
    
    // Set RT priority
    if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
        perror("sched_setscheduler");
        return NULL;
    }
    
    // Initialize timing
    next_wakeup = get_time_ns();
    
    printf("Control loop started (1kHz, priority 80)\n");
    
    while (running) {
        // Wait for next period
        next_wakeup += CONTROL_PERIOD_NS;
        sleep_until(next_wakeup);
        
        start_time = get_time_ns();
        latency = start_time - next_wakeup;
        
        // Check for emergency stop
        pthread_mutex_lock(&state_lock);
        if (motor.emergency_stop) {
            set_pwm(0);
            pthread_mutex_unlock(&state_lock);
            break;
        }
        
        // Read sensor
        motor.current_position = read_encoder();
        
        // Run controller
        int pwm = pid_control(motor.target_position, motor.current_position);
        set_pwm(pwm);
        
        pthread_mutex_unlock(&state_lock);
        
        end_time = get_time_ns();
        
        // Update statistics
        if (latency < stats.min_latency) stats.min_latency = latency;
        if (latency > stats.max_latency) stats.max_latency = latency;
        stats.sum_latency += latency;
        stats.iterations++;
        
        if (latency > MAX_LATENCY_NS) {
            stats.missed_deadlines++;
        }
        
        if (end_time > next_wakeup + CONTROL_PERIOD_NS) {
            stats.overruns++;
        }
        
        // Periodic status
        if (stats.iterations % 1000 == 0) {
            printf("Iter %lu: pos=%d, target=%d, pwm=%d, latency=%lld ns\n",
                   stats.iterations, motor.current_position,
                   motor.target_position, motor.pwm_duty, latency);
        }
    }
    
    printf("Control loop stopped\n");
    return NULL;
}

// Emergency stop handler (high priority)
static void *emergency_monitor(void *arg) {
    struct sched_param param = {.sched_priority = 99};
    
    sched_setscheduler(0, SCHED_FIFO, &param);
    
    printf("Emergency monitor started (priority 99)\n");
    
    while (running) {
        // Check emergency stop button (simulated)
        // In real system: wait for GPIO interrupt
        usleep(10000);  // 10ms
        
        // Simulate random emergency stop
        if (rand() % 10000 == 0) {
            printf("EMERGENCY STOP TRIGGERED!\n");
            pthread_mutex_lock(&state_lock);
            motor.emergency_stop = 1;
            pthread_mutex_unlock(&state_lock);
            running = 0;
        }
    }
    
    return NULL;
}

// Performance monitor (normal priority)
static void *perf_monitor(void *arg) {
    while (running) {
        sleep(5);
        
        if (stats.iterations > 0) {
            printf("\n=== Performance Statistics ===\n");
            printf("Iterations: %lu\n", stats.iterations);
            printf("Min latency: %lld ns (%.2f μs)\n",
                   stats.min_latency, stats.min_latency / 1000.0);
            printf("Max latency: %lld ns (%.2f μs)\n",
                   stats.max_latency, stats.max_latency / 1000.0);
            printf("Avg latency: %lld ns (%.2f μs)\n",
                   stats.sum_latency / stats.iterations,
                   (stats.sum_latency / stats.iterations) / 1000.0);
            printf("Missed deadlines: %lu (%.4f%%)\n",
                   stats.missed_deadlines,
                   100.0 * stats.missed_deadlines / stats.iterations);
            printf("Overruns: %lu (%.4f%%)\n",
                   stats.overruns,
                   100.0 * stats.overruns / stats.iterations);
            printf("==============================\n\n");
        }
    }
    
    return NULL;
}

// Signal handler
static void signal_handler(int sig) {
    running = 0;
}

// Setup RT environment
static int setup_rt(void) {
    struct rlimit rlim;
    
    // Increase memory lock limit
    rlim.rlim_cur = RLIM_INFINITY;
    rlim.rlim_max = RLIM_INFINITY;
    if (setrlimit(RLIMIT_MEMLOCK, &rlim) != 0) {
        perror("setrlimit");
        return -1;
    }
    
    // Lock all memory
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall");
        return -1;
    }
    
    // Prefault stack
    unsigned char dummy[STACK_SIZE];
    memset(dummy, 0, STACK_SIZE);
    
    printf("RT environment configured\n");
    return 0;
}

int main(int argc, char *argv[]) {
    pthread_t control_thread, emergency_thread, monitor_thread;
    pthread_mutexattr_t attr;
    
    printf("RT Motor Control System\n");
    printf("=======================\n\n");
    
    // Initialize
    memset(&stats, 0, sizeof(stats));
    stats.min_latency = LLONG_MAX;
    memset(&motor, 0, sizeof(motor));
    motor.target_position = 1000;
    
    // Setup mutex with priority inheritance
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
    pthread_mutex_init(&state_lock, &attr);
    
    // Setup RT environment
    if (setup_rt() != 0) {
        fprintf(stderr, "Failed to setup RT environment\n");
        return 1;
    }
    
    // Install signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Create threads
    pthread_create(&control_thread, NULL, control_loop, NULL);
    pthread_create(&emergency_thread, NULL, emergency_monitor, NULL);
    pthread_create(&monitor_thread, NULL, perf_monitor, NULL);
    
    // Wait for completion
    pthread_join(control_thread, NULL);
    pthread_join(emergency_thread, NULL);
    running = 0;  // Stop monitor
    pthread_join(monitor_thread, NULL);
    
    // Final statistics
    printf("\n=== Final Statistics ===\n");
    printf("Total iterations: %lu\n", stats.iterations);
    printf("Min latency: %.2f μs\n", stats.min_latency / 1000.0);
    printf("Max latency: %.2f μs\n", stats.max_latency / 1000.0);
    printf("Avg latency: %.2f μs\n",
           (stats.sum_latency / stats.iterations) / 1000.0);
    printf("Missed deadlines: %lu (%.4f%%)\n",
           stats.missed_deadlines,
           100.0 * stats.missed_deadlines / stats.iterations);
    printf("========================\n");
    
    return 0;
}
```

### Compilation and Execution

```bash
# Compile
gcc -o rt_motor rt_motor_control.c -pthread -lrt -O2

# Run (requires root for RT scheduling)
sudo ./rt_motor

# Run with CPU isolation
sudo taskset -c 2 ./rt_motor

# Monitor with cyclictest in parallel
sudo cyclictest -t1 -p 90 -i 1000 -l 10000 &
sudo ./rt_motor
```

---

## 📊 Expected Results

### Performance Metrics

| Metric | Target | Typical Result |
|--------|--------|----------------|
| Min Latency | <10μs | 2-5μs |
| Avg Latency | <20μs | 8-15μs |
| Max Latency | <50μs | 20-40μs |
| Missed Deadlines | <0.01% | 0-0.001% |
| Jitter (StdDev) | <10μs | 3-8μs |

---

## 🧠 Assessment

**Q:** Why is the emergency monitor at higher priority than control loop?
**A:** Safety-critical tasks must preempt all other tasks to ensure fastest response.

**Q:** What would happen without memory locking?
**A:** Page faults could cause millisecond latencies, missing control deadlines.

**Q:** Why use priority inheritance for the mutex?
**A:** Prevents priority inversion where emergency monitor could be blocked by lower priority task.

---

## 🎓 Week 36 Summary

Covered complete RT Linux stack:
1. **PREEMPT_RT kernel** - Fully preemptible
2. **RT scheduling** - SCHED_FIFO, SCHED_RR, SCHED_DEADLINE
3. **Interrupt handling** - Threaded IRQs, IRQ affinity
4. **High-resolution timers** - Nanosecond precision
5. **Synchronization** - Priority inheritance, futexes
6. **Memory management** - mlockall, huge pages, NUMA

**Key Takeaway:** Building RT systems requires careful attention to every layer: kernel configuration, scheduling, memory, interrupts, and synchronization. All must work together to achieve deterministic behavior.

---

## 🚀 Next Steps

**Week 37 Preview:** Kernel Security - LSM (Linux Security Modules), SELinux, AppArmor, Seccomp, and Kernel Hardening.

---
