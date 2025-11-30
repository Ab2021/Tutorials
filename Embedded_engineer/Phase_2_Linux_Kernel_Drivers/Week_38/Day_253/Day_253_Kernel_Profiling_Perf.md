# Day 253: Kernel Profiling with perf
## Phase 2: Linux Kernel & Device Drivers | Week 38: Performance Optimization

---

## 🎯 Learning Objectives
1. **Use** perf for kernel profiling
2. **Analyze** CPU hotspots and bottlenecks
3. **Profile** cache misses and branch mispredictions
4. **Generate** flame graphs
5. **Optimize** based on profiling data

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Performance Counters

**Hardware Performance Counters** track:
- CPU cycles
- Instructions executed
- Cache hits/misses
- Branch predictions
- TLB misses
- Memory bandwidth

### 🔹 Part 2: perf Tool

```bash
# Record system-wide
perf record -a -g sleep 10

# Record specific process
perf record -p <PID> -g

# Record specific event
perf record -e cycles,cache-misses ./myapp

# Report
perf report

# Top (live view)
perf top
```

---

## 💻 Implementation Examples

### Example 1: CPU Profiling

```bash
# Profile application
perf record -g ./myapp

# View report
perf report --stdio

# Generate flame graph
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

### Example 2: Cache Analysis

```bash
# Profile cache misses
perf stat -e cache-references,cache-misses ./myapp

# Detailed cache analysis
perf record -e cache-misses -g ./myapp
perf report
```

---

## 🧠 Assessment

**Q:** What is a flame graph?
**A:** Visualization showing call stack samples, width represents time spent.

**Q:** How to reduce cache misses?
**A:** Improve data locality, use cache-friendly data structures, prefetching.

---

## 🎓 Summary

Covered perf tool usage, CPU profiling, cache analysis, and performance optimization techniques.

---

## 🚀 Next Steps

Day 254: Ftrace and Function Tracing

---
