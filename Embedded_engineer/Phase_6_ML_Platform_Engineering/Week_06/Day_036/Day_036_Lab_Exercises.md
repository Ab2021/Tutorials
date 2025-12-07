# Days 36-42: Week 6 - Profiling & Optimization Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Day 36: Nsight Systems

```bash
# Profile Python training script
nsys profile -o training_profile python train.py

# View in Nsight Systems GUI
nsys-ui training_profile.nsys-rep
```

---

## Day 37: Nsight Compute

```bash
# Detailed kernel analysis
ncu --set full python inference.py

# Generate report
ncu --export report.ncu-rep python inference.py
```

---

## Day 38: PyTorch Profiler

```python
from torch.profiler import profile, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step, (data, target) in enumerate(dataloader):
        if step >= 10:
            break
        output = model(data.cuda())
        loss = criterion(output, target.cuda())
        loss.backward()

# View in TensorBoard
# tensorboard --logdir=./logs
```

---

## Day 39: Memory Profiling

```python
import torch

# Memory snapshot
torch.cuda.memory._record_memory_history(max_entries=100000)

# Run workload
# ...

# Save snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# Visualize at pytorch.org/memory_viz
```

---

## Day 40: Performance Anti-Patterns

```python
# BAD: CPU-GPU sync in loop
for i in range(100):
    x = model(data)
    print(x[0].item())  # Forces sync!

# GOOD: Batch operations
results = []
for i in range(100):
    x = model(data)
    results.append(x)
# Print after loop
```

---

## Day 41: Benchmarking Framework

```python
import torch
import time

class Benchmark:
    def __init__(self, warmup=10, iterations=100):
        self.warmup = warmup
        self.iterations = iterations
    
    def run(self, fn, *args):
        # Warmup
        for _ in range(self.warmup):
            fn(*args)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(self.iterations):
            fn(*args)
        torch.cuda.synchronize()
        
        return (time.perf_counter() - start) / self.iterations * 1000
```

---

## Day 42: Week 6 Project

```python
# Optimization Case Study
# 1. Profile initial implementation
# 2. Identify top 3 bottlenecks
# 3. Apply optimizations:
#    - Batch operations
#    - Reduce CPU-GPU transfers
#    - Enable mixed precision
#    - Fuse layers
# 4. Measure improvement
# 5. Document findings
```

---

## 📝 Week 6 Summary
| Day | Topic | Tool |
|-----|-------|------|
| 36 | Nsight Systems | nsys |
| 37 | Nsight Compute | ncu |
| 38 | PyTorch Profiler | torch.profiler |
| 39 | Memory Profiling | memory_snapshot |
| 40 | Anti-Patterns | Best practices |
| 41 | Benchmarking | Statistical methods |
| 42 | Project | Case study |

---

## 🎓 Phase 6A Complete!
GPU Fundamentals & CUDA Programming (Weeks 1-6)
