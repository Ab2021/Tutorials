# Day 254: Ftrace and Function Tracing
## Phase 2: Linux Kernel & Device Drivers | Week 38: Performance Optimization

---

## 🎯 Learning Objectives
1. **Master** ftrace framework for kernel tracing
2. **Use** function tracer and function graph tracer
3. **Analyze** trace events and custom tracepoints
4. **Debug** kernel issues with tracing
5. **Measure** function latencies

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Ftrace Architecture

Ftrace (Function Tracer) is the kernel's built-in tracing framework.

**Tracers Available:**
- `function` - Trace all kernel functions
- `function_graph` - Show call graphs with timing
- `irqsoff` - Trace IRQ disabled periods
- `preemptoff` - Trace preemption disabled periods
- `wakeup` - Trace task wakeup latency
- `blk` - Block I/O tracer

### 🔹 Part 2: Trace Events

Kernel has 1000+ predefined trace events:
```bash
# List all events
ls /sys/kernel/debug/tracing/events/

# Categories:
# - sched: Scheduler events
# - irq: Interrupt events
# - block: Block I/O
# - syscalls: System calls
# - net: Network events
```

---

## 💻 Implementation Examples

### Example 1: Basic Function Tracing

```bash
#!/bin/bash
# ftrace_basic.sh

cd /sys/kernel/debug/tracing

# 1. Select tracer
echo function > current_tracer

# 2. Set filter (optional)
echo 'tcp_*' > set_ftrace_filter

# 3. Enable tracing
echo 1 > tracing_on

# 4. Run workload
sleep 5

# 5. Disable tracing
echo 0 > tracing_on

# 6. View trace
cat trace | head -100

# 7. Clear trace
echo > trace

# 8. Reset
echo nop > current_tracer
```

### Example 2: Function Graph Tracer

```bash
cd /sys/kernel/debug/tracing

# Enable function graph
echo function_graph > current_tracer

# Set depth limit
echo 5 > max_graph_depth

# Filter specific function
echo do_sys_open > set_graph_function

# Trace
echo 1 > tracing_on
cat /etc/passwd > /dev/null
echo 0 > tracing_on

# View with timing
cat trace

# Output shows:
#  2)               |  do_sys_open() {
#  2)   0.123 us    |    getname();
#  2)   0.456 us    |    get_unused_fd_flags();
#  2)               |    do_filp_open() {
#  2)   1.234 us    |      path_openat();
#  2)   2.345 us    |    }
#  2)   3.456 us    |  }
```

### Example 3: Trace Events

```bash
# Enable scheduler events
echo 1 > events/sched/sched_switch/enable
echo 1 > events/sched/sched_wakeup/enable

# Trace
echo 1 > tracing_on
sleep 1
echo 0 > tracing_on

# View
cat trace

# Filter by PID
echo 'pid == 1234' > events/sched/sched_switch/filter
```

---

## 🔬 Lab Exercises

### Lab 1: Trace System Call Latency

```bash
# Trace open() syscall
cd /sys/kernel/debug/tracing
echo 1 > events/syscalls/sys_enter_open/enable
echo 1 > events/syscalls/sys_exit_open/enable
echo 1 > tracing_on

# Generate activity
ls -R / > /dev/null 2>&1 &

# Analyze
cat trace | grep open
```

### Lab 2: Find Latency Hotspots

```bash
# Use irqsoff tracer
echo irqsoff > current_tracer
echo 1 > tracing_on

# Run workload
sleep 10

echo 0 > tracing_on
cat trace

# Shows longest IRQ-disabled period
```

---

## 🧠 Assessment

**Q:** What is the difference between function and function_graph tracer?
**A:** function lists all function calls. function_graph shows call hierarchy with timing.

**Q:** How to reduce ftrace overhead?
**A:** Use filters, limit depth, trace specific events only.

---

## 🎓 Summary

Covered ftrace framework, function tracing, trace events, and latency analysis for kernel debugging and optimization.

---

## 🚀 Next Steps

Day 255: eBPF and Dynamic Tracing

---
