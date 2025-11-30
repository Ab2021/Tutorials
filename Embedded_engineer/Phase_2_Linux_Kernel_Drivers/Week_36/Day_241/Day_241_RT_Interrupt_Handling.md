# Day 241: Interrupt Handling in Real-Time Systems
## Phase 2: Linux Kernel & Device Drivers | Week 36: Real-Time Systems

---

## 🎯 Learning Objectives
1. **Understand** threaded interrupts in PREEMPT_RT
2. **Implement** RT-safe interrupt handlers
3. **Configure** IRQ affinity for RT optimization
4. **Measure** interrupt latency
5. **Optimize** interrupt handling for deterministic behavior

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Interrupt Handling in Standard vs RT Linux

**Standard Linux:**
- Hard IRQ context (atomic, non-preemptible)
- Softirqs and tasklets for deferred work
- Can cause unbounded latencies

**PREEMPT_RT Linux:**
- Threaded interrupts (preemptible)
- IRQ threads with configurable priorities
- Bounded, predictable latencies

### 🔹 Part 2: Threaded Interrupts

```c
// Standard interrupt handler
static irqreturn_t my_irq_handler(int irq, void *dev_id) {
    // Runs in hard IRQ context
    // Must be fast, no sleeping
    // Schedule bottom half
    return IRQ_WAKE_THREAD;
}

static irqreturn_t my_irq_thread(int irq, void *dev_id) {
    // Runs in thread context (PREEMPT_RT)
    // Can sleep, use mutexes
    // Can be preempted by higher priority tasks
    return IRQ_HANDLED;
}

// Registration
request_threaded_irq(irq, my_irq_handler, my_irq_thread, 
                     IRQF_ONESHOT, "my_device", dev);
```

### 🔹 Part 3: IRQ Affinity

**CPU Isolation for RT:**
```bash
# Boot parameter
isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3

# Move all IRQs to CPU 0-1
for irq in /proc/irq/*/smp_affinity; do
    echo 3 > $irq  # CPUs 0-1 (binary: 0011)
done

# Run RT tasks on CPU 2-3
taskset -c 2,3 ./rt_application
```

---

## 💻 Implementation Examples

### Example 1: RT-Safe Interrupt Handler

```c
#include <linux/module.h>
#include <linux/interrupt.h>
#include <linux/kthread.h>
#include <linux/sched.h>

struct my_device {
    int irq;
    struct task_struct *thread;
    wait_queue_head_t wq;
    atomic_t data_ready;
};

// Top half (minimal work)
static irqreturn_t my_top_half(int irq, void *dev_id) {
    struct my_device *dev = dev_id;
    
    // Acknowledge hardware interrupt
    // ... hardware-specific code ...
    
    // Signal thread
    atomic_set(&dev->data_ready, 1);
    wake_up(&dev->wq);
    
    return IRQ_HANDLED;
}

// Bottom half (threaded)
static int my_irq_thread(void *data) {
    struct my_device *dev = data;
    struct sched_param param = { .sched_priority = 50 };
    
    // Set RT priority
    sched_setscheduler(current, SCHED_FIFO, &param);
    
    while (!kthread_should_stop()) {
        // Wait for interrupt
        wait_event_interruptible(dev->wq, 
            atomic_read(&dev->data_ready) || kthread_should_stop());
        
        if (kthread_should_stop())
            break;
        
        atomic_set(&dev->data_ready, 0);
        
        // Process data (can sleep, use mutexes)
        // ... actual work ...
    }
    
    return 0;
}

static int __init my_init(void) {
    struct my_device *dev;
    
    dev = kzalloc(sizeof(*dev), GFP_KERNEL);
    init_waitqueue_head(&dev->wq);
    atomic_set(&dev->data_ready, 0);
    
    // Request IRQ
    request_irq(dev->irq, my_top_half, IRQF_SHARED, "my_device", dev);
    
    // Create thread
    dev->thread = kthread_run(my_irq_thread, dev, "my_irq_thread");
    
    return 0;
}
```

### Example 2: IRQ Latency Measurement

```c
#include <linux/module.h>
#include <linux/interrupt.h>
#include <linux/ktime.h>

static ktime_t irq_timestamp;
static s64 min_latency = S64_MAX;
static s64 max_latency = 0;
static s64 sum_latency = 0;
static unsigned long count = 0;

static irqreturn_t latency_irq_handler(int irq, void *dev_id) {
    ktime_t now = ktime_get();
    s64 latency_ns;
    
    if (ktime_to_ns(irq_timestamp) != 0) {
        latency_ns = ktime_to_ns(ktime_sub(now, irq_timestamp));
        
        if (latency_ns < min_latency) min_latency = latency_ns;
        if (latency_ns > max_latency) max_latency = latency_ns;
        sum_latency += latency_ns;
        count++;
        
        if (count % 1000 == 0) {
            pr_info("IRQ latency: min=%lld ns, max=%lld ns, avg=%lld ns\n",
                    min_latency, max_latency, sum_latency / count);
        }
    }
    
    irq_timestamp = now;
    return IRQ_HANDLED;
}
```

---

## 🔬 Lab Exercises

### Lab 1: Measure Interrupt Latency

```bash
# Use ftrace to measure IRQ latency
echo 1 > /sys/kernel/debug/tracing/events/irq/enable
echo irqsoff > /sys/kernel/debug/tracing/current_tracer
echo 1 > /sys/kernel/debug/tracing/tracing_on

# Generate interrupts (e.g., network traffic)
ping -f localhost &

# After some time
echo 0 > /sys/kernel/debug/tracing/tracing_on
cat /sys/kernel/debug/tracing/trace_stat/function0
```

### Lab 2: IRQ Thread Priority Tuning

```bash
# Find IRQ thread PID
ps aux | grep irq

# Change priority
chrt -f -p 90 <IRQ_THREAD_PID>

# Verify
chrt -p <IRQ_THREAD_PID>
```

---

## 🧠 Assessment

**Q:** Why are interrupts threaded in PREEMPT_RT?
**A:** To make them preemptible by higher priority RT tasks, providing bounded latencies.

**Q:** What is the trade-off of threaded interrupts?
**A:** Slightly higher interrupt latency, but much better worst-case latency for RT tasks.

---

## 🎓 Summary

Covered threaded interrupts, IRQ affinity, latency measurement, and RT-safe interrupt handling patterns.

---

## 🚀 Next Steps

Day 242: High-Resolution Timers and RT Clock Sources

---
