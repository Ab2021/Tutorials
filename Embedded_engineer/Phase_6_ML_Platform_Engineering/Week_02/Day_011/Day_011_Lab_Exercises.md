# Day 11: Dynamic Parallelism - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🔬 Exercise 1: Understanding Dynamic Parallelism (Beginner)

### Objective
Understand when and why to use dynamic parallelism.

### Conceptual Code (CUDA C - for understanding)

```cpp
// Parent kernel launches child kernels
__global__ void parent_kernel(int* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // Conditionally launch child based on data
        if (data[idx] > threshold) {
            // Launch child kernel from GPU!
            child_kernel<<<1, 32>>>(data, idx);
        }
    }
}

__global__ void child_kernel(int* data, int parent_idx) {
    // Process sub-problem
    int idx = threadIdx.x;
    data[parent_idx * 32 + idx] *= 2;
}
```

### Python Simulation

```python
import torch

def simulate_dynamic_parallelism():
    """Simulate dynamic parallelism concept."""
    
    print("Dynamic Parallelism Simulation")
    print("=" * 40)
    
    # Data with varying "complexity"
    data = torch.randint(0, 10, (1000,), device='cuda')
    threshold = 5
    
    # Count high-complexity items
    complex_items = (data > threshold).sum().item()
    
    print(f"Total items: {len(data)}")
    print(f"Complex items (>{threshold}): {complex_items}")
    print(f"Would launch {complex_items} child kernels")
    
    # In real dynamic parallelism:
    # - Parent kernel runs on GPU
    # - Parent decides which items need more work
    # - Parent launches child kernels for those items ONLY
    # - Avoids launching work for simple items

if __name__ == "__main__":
    simulate_dynamic_parallelism()
```

---

## 🔬 Exercise 2: Recursive Processing (Intermediate)

### Objective
Implement recursive-style processing patterns.

### Code

```python
import torch

def adaptive_processing():
    """Simulate adaptive mesh refinement pattern."""
    
    print("Adaptive Mesh Refinement Simulation")
    print("=" * 40)
    
    # Simulate 2D grid with varying detail needed
    grid_size = 64
    grid = torch.rand(grid_size, grid_size, device='cuda')
    
    # Mark regions needing refinement
    needs_refinement = grid > 0.9  # 10% of regions
    
    levels = 0
    total_cells = grid_size * grid_size
    
    while needs_refinement.any():
        count = needs_refinement.sum().item()
        print(f"Level {levels}: {count} cells need refinement")
        
        # In dynamic parallelism:
        # - Launch child kernels only for cells needing refinement
        # - Each child subdivides its cell into 4
        # - Children may launch grandchildren
        
        # Simulate: refine 50% of cells each level
        if count > 0:
            # Random subset continues to need refinement
            needs_refinement = needs_refinement & (torch.rand_like(grid) > 0.5)
        
        levels += 1
        if levels > 10:
            break
    
    print(f"Total levels: {levels}")

if __name__ == "__main__":
    adaptive_processing()
```

---

## 🔬 Exercise 3: Alternatives to Dynamic Parallelism (Advanced)

### Objective
Implement patterns that avoid dynamic parallelism overhead.

### Code

```python
import torch

def compact_and_process():
    """
    Alternative to dynamic parallelism:
    Compact items needing work, then process in batch.
    """
    
    print("Compact-and-Process Pattern")
    print("=" * 40)
    
    n = 1_000_000
    data = torch.rand(n, device='cuda')
    threshold = 0.9
    
    # Step 1: Identify items needing work
    mask = data > threshold
    
    # Step 2: Compact (gather only relevant items)
    indices = mask.nonzero(as_tuple=True)[0]
    selected = data[indices]
    
    # Step 3: Process only selected items
    result = selected * 2 + 1
    
    # Step 4: Scatter back
    data[indices] = result
    
    print(f"Total items: {n:,}")
    print(f"Items processed: {len(indices):,} ({100*len(indices)/n:.1f}%)")
    print(f"Avoided processing: {n - len(indices):,} items")
    
    # This achieves similar effect to dynamic parallelism
    # without GPU-side kernel launches

if __name__ == "__main__":
    compact_and_process()
```

---

## 🐛 Common Issues

### Issue: Dynamic parallelism not supported
**Cause:** Requires Compute Capability 3.5+
**Fix:** Use compact-and-process pattern instead

### Issue: High overhead
**Cause:** Child kernel launch overhead
**Fix:** Batch small workloads together

---

## 📚 Additional Resources
- [CUDA Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism)
- [When to Use Dynamic Parallelism](https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/)
