# Day 1: GPU Architecture Deep Dive - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🔬 Exercise 1: GPU Discovery (Beginner)

### Objective
Use command-line tools to understand your GPU's capabilities.

### Instructions

```bash
# 1. Basic GPU Info
nvidia-smi

# 2. Query specific properties
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv

# 3. Monitor GPU in real-time (run in separate terminal)
watch -n 1 nvidia-smi

# 4. View GPU topology
nvidia-smi topo -m
```

### Questions to Answer
1. What is your GPU's Compute Capability?
2. How much VRAM does your GPU have?
3. What driver version is installed?

---

## 🔬 Exercise 2: Memory Hierarchy Exploration (Intermediate)

### Objective
Measure and compare different memory access speeds.

### Code

```python
import torch
import time

def benchmark_memory(size_mb=100, iterations=10):
    """Benchmark different memory operations."""
    size = size_mb * 1024 * 1024 // 4  # float32 elements
    
    results = {}
    
    # 1. CPU to GPU (Host to Device)
    cpu_tensor = torch.randn(size)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        gpu_tensor = cpu_tensor.cuda()
        torch.cuda.synchronize()
    h2d_time = (time.perf_counter() - start) / iterations
    results['H2D_GB_s'] = (size_mb / 1024) / h2d_time
    
    # 2. GPU to CPU (Device to Host)
    gpu_tensor = torch.randn(size, device='cuda')
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        cpu_tensor = gpu_tensor.cpu()
        torch.cuda.synchronize()
    d2h_time = (time.perf_counter() - start) / iterations
    results['D2H_GB_s'] = (size_mb / 1024) / d2h_time
    
    # 3. GPU to GPU (Device to Device)
    src = torch.randn(size, device='cuda')
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        dst = src.clone()
        torch.cuda.synchronize()
    d2d_time = (time.perf_counter() - start) / iterations
    results['D2D_GB_s'] = (size_mb / 1024) / d2d_time
    
    return results

if __name__ == "__main__":
    print("Memory Bandwidth Benchmark")
    print("=" * 40)
    results = benchmark_memory(size_mb=500)
    for key, value in results.items():
        print(f"{key}: {value:.2f} GB/s")
```

### Expected Output
```
Memory Bandwidth Benchmark
========================================
H2D_GB_s: 12.50 GB/s
D2H_GB_s: 13.20 GB/s
D2D_GB_s: 750.00 GB/s
```

### Analysis Questions
1. Why is D2D so much faster than H2D/D2H?
2. What limits the PCIe transfer speed?

---

## 🔬 Exercise 3: Compute Capability Analysis (Advanced)

### Objective
Write a script that determines optimal kernel configurations based on GPU specs.

### Code

```python
import torch

def analyze_gpu(device_id=0):
    """Analyze GPU and provide optimization recommendations."""
    props = torch.cuda.get_device_properties(device_id)
    
    cc = (props.major, props.minor)
    
    analysis = {
        "name": props.name,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "sm_count": props.multi_processor_count,
        "max_threads_per_block": props.max_threads_per_block,
        "max_threads_per_sm": props.max_threads_per_multi_processor,
        "warp_size": props.warp_size,
        "max_shared_mem_per_block_kb": props.max_shared_memory_per_block / 1024,
        "total_memory_gb": props.total_memory / (1024**3),
    }
    
    # Recommendations
    recommendations = []
    
    # Block size recommendation
    optimal_block = 256 if cc[0] >= 7 else 128
    recommendations.append(f"Optimal block size: {optimal_block} threads")
    
    # Tensor cores
    if cc[0] >= 7:
        recommendations.append("✓ Tensor Cores available - use mixed precision!")
    
    # Memory
    if analysis["total_memory_gb"] >= 16:
        recommendations.append("✓ Large VRAM - can train large models")
    else:
        recommendations.append("⚠ Limited VRAM - use gradient checkpointing")
    
    return analysis, recommendations

if __name__ == "__main__":
    analysis, recs = analyze_gpu()
    
    print("GPU Analysis Report")
    print("=" * 50)
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    print("\nRecommendations:")
    for rec in recs:
        print(f"  • {rec}")
```

---

## 🐛 Common Errors & Fixes

### Error 1: CUDA not available
```python
>>> torch.cuda.is_available()
False
```
**Fix:** Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Error 2: Out of Memory
```
RuntimeError: CUDA out of memory
```
**Fix:** Clear cache: `torch.cuda.empty_cache()`

### Error 3: Driver/CUDA mismatch
```
CUDA driver version is insufficient
```
**Fix:** Update NVIDIA driver or downgrade CUDA toolkit.

---

## 📚 Additional Resources
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [GPU Compute Capability Chart](https://developer.nvidia.com/cuda-gpus)
