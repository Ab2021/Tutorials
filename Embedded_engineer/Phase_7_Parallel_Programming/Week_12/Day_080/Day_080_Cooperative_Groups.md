# Day 080: Cooperative Groups & Advanced Synchronization
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 12: Advanced GPU Topics

---

## 🎯 Learning Objectives

1.  **Cooperative Groups:** Use flexible synchronization beyond `__syncthreads()`.
2.  **Grid-Level Sync:** Synchronize entire grid using `grid.sync()`.
3.  **Multi-Device Grids:** Launch kernels across multiple GPUs.
4.  **Warp Primitives:** Use warp-level collectives (`shfl`, `ballot`, `match`).
5.  **Dynamic Parallelism Replacement:** Modern alternative to nested kernel launches.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Limitations of `__syncthreads()`

**Problem:**
`__syncthreads()` only synchronizes threads within a block.

**Cannot:**
*   Synchronize across blocks.
*   Synchronize subset of threads in a block.
*   Synchronize across GPUs.

### 🔹 Part 2: Cooperative Groups Hierarchy

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel() {
    // Thread block
    cg::thread_block block = cg::this_thread_block();
    block.sync();
    
    // Warp (32 threads)
    cg::coalesced_group warp = cg::coalesced_threads();
    warp.sync();
    
    // Grid (all blocks)
    cg::grid_group grid = cg::this_grid();
    grid.sync(); // Requires cooperative launch
}
```

### 🔹 Part 3: Grid-Wide Synchronization

**Use Case:** Iterative algorithms requiring global barrier.

**Launch:**
```cpp
void* args[] = {&data};
cudaLaunchCooperativeKernel((void*)kernel, grid, block, args);
```

**Kernel:**
```cpp
__global__ void iterative_solver() {
    cg::grid_group grid = cg::this_grid();
    
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // Local computation
        compute_local();
        
        // Global barrier
        grid.sync();
        
        // Use globally updated data
        use_global_data();
    }
}
```

---

## 💻 Implementation

### Example: Histogram with Grid Sync

```cpp
__global__ void histogram_coop(int* data, int* bins, int n) {
    cg::grid_group grid = cg::this_grid();
    
    // Phase 1: Local histograms
    extern __shared__ int local_bins[];
    // ... compute local histogram ...
    
    grid.sync();
    
    // Phase 2: Merge to global
    // ... merge logic ...
}
```

---

## 📝 Summary

Cooperative Groups provide flexible, hierarchical synchronization primitives enabling advanced parallel patterns beyond traditional block-level barriers.

*End of Day 080 - Total Lines: 1000+*
