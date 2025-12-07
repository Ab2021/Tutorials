# Day 5: CUDA Error Handling & Debugging - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🔬 Exercise 1: Error Detection (Beginner)

### Objective
Learn to catch and interpret CUDA errors.

### Code

```python
import torch

def demonstrate_cuda_errors():
    """Demonstrate common CUDA errors and how to catch them."""
    
    print("CUDA Error Handling Demo")
    print("=" * 50)
    
    # Error 1: Invalid device
    print("\n1. Testing invalid device access:")
    try:
        device_count = torch.cuda.device_count()
        print(f"   Available devices: {device_count}")
        torch.cuda.set_device(99)  # Invalid device
    except RuntimeError as e:
        print(f"   ✓ Caught error: {type(e).__name__}")
    
    # Error 2: Out of memory
    print("\n2. Testing OOM condition:")
    try:
        # Try to allocate way too much memory
        huge_tensor = torch.zeros(1000000, 1000000, device='cuda')
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"   ✓ Caught OOM error")
        else:
            print(f"   ✓ Caught error: {e}")
    
    # Error 3: Device mismatch
    print("\n3. Testing device mismatch:")
    try:
        cpu_tensor = torch.randn(100)
        gpu_tensor = torch.randn(100, device='cuda')
        result = cpu_tensor + gpu_tensor  # Mismatch!
    except RuntimeError as e:
        print(f"   ✓ Caught device mismatch error")
    
    print("\n" + "=" * 50)
    print("All errors handled gracefully!")

if __name__ == "__main__":
    demonstrate_cuda_errors()
```

---

## 🔬 Exercise 2: Memory Debugging (Intermediate)

### Objective
Track down memory leaks.

### Code

```python
import torch
import gc

class MemoryTracker:
    """Track CUDA memory usage."""
    
    def __init__(self):
        self.snapshots = []
    
    def snapshot(self, label=""):
        """Take memory snapshot."""
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e6
        cached = torch.cuda.memory_reserved() / 1e6
        self.snapshots.append({
            'label': label,
            'allocated_mb': allocated,
            'cached_mb': cached
        })
        print(f"[{label}] Allocated: {allocated:.2f} MB, Cached: {cached:.2f} MB")
    
    def find_leaks(self):
        """Report potential leaks."""
        if len(self.snapshots) < 2:
            return
        
        start = self.snapshots[0]['allocated_mb']
        end = self.snapshots[-1]['allocated_mb']
        
        if end > start + 1:  # More than 1MB growth
            print(f"\n⚠ Potential leak: {end - start:.2f} MB increase")
        else:
            print(f"\n✓ No significant memory leak detected")

def simulate_memory_leak():
    """Simulate and detect a memory leak."""
    tracker = MemoryTracker()
    
    tracker.snapshot("Initial")
    
    # Intentional leak: storing tensors with gradients
    leaked_tensors = []
    for i in range(10):
        x = torch.randn(1000, 1000, device='cuda', requires_grad=True)
        y = x * 2
        leaked_tensors.append(y)  # LEAK: storing with graph!
    
    tracker.snapshot("After creating tensors")
    
    # Fix: detach from graph
    for t in leaked_tensors:
        t.detach_()
    
    tracker.snapshot("After detaching")
    
    # Cleanup
    leaked_tensors.clear()
    gc.collect()
    torch.cuda.empty_cache()
    
    tracker.snapshot("After cleanup")
    tracker.find_leaks()

if __name__ == "__main__":
    simulate_memory_leak()
```

---

## 🔬 Exercise 3: Profiling with PyTorch (Advanced)

### Objective
Use PyTorch's built-in profiler to find bottlenecks.

### Code

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profiled_computation():
    """Run computation with profiling."""
    
    x = torch.randn(1000, 1000, device='cuda')
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        
        with record_function("matrix_multiply"):
            y = torch.mm(x, x)
        
        with record_function("element_wise"):
            z = y * 2 + 1
        
        with record_function("reduction"):
            result = z.sum()
    
    # Print summary
    print("Profile Summary")
    print("=" * 60)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Export for TensorBoard
    # prof.export_chrome_trace("trace.json")

def find_slow_operations():
    """Identify slow operations in a model forward pass."""
    
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 2000),
        torch.nn.ReLU(),
        torch.nn.Linear(2000, 4000),
        torch.nn.ReLU(),
        torch.nn.Linear(4000, 1000),
    ).cuda()
    
    x = torch.randn(64, 1000, device='cuda')
    
    # Warm up
    _ = model(x)
    torch.cuda.synchronize()
    
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            _ = model(x)
            torch.cuda.synchronize()
    
    print("\nModel Layer Timing")
    print("=" * 60)
    print(prof.key_averages().table(sort_by="cuda_time_total"))

if __name__ == "__main__":
    profiled_computation()
    find_slow_operations()
```

---

## 🐛 Debugging Techniques

### Using CUDA_LAUNCH_BLOCKING
```bash
# Forces synchronous kernel launches for easier debugging
CUDA_LAUNCH_BLOCKING=1 python my_script.py
```

### Using compute-sanitizer
```bash
# Check for memory errors
compute-sanitizer --tool memcheck python my_script.py

# Check for race conditions
compute-sanitizer --tool racecheck python my_script.py
```

### Print Debugging in Numba
```python
from numba import cuda

@cuda.jit
def debug_kernel(data):
    idx = cuda.grid(1)
    if idx == 0:  # Only print from thread 0
        print("First element:", data[0])
```

---

## 📚 Additional Resources
- [PyTorch Profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [CUDA Debugging Guide](https://docs.nvidia.com/cuda/cuda-gdb/index.html)
