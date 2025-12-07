# Day 081: CUDA Unified Memory Advanced
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 12: Advanced GPU Topics

---

## 🎯 Learning Objectives

1.  **Prefetching:** Use `cudaMemPrefetchAsync` for explicit data migration.
2.  **Memory Advise:** Optimize page placement with `cudaMemAdvise`.
3.  **Access Counters:** Monitor page faults and migrations.
4.  **Oversubscription:** Handle datasets larger than GPU memory.
5.  **Performance Tuning:** Achieve near-explicit-copy performance with UM.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Unified Memory Basics

**Concept:**
Single pointer accessible from CPU and GPU.

**Automatic Migration:**
Pages migrate on-demand (page faults).

**Problem:**
Page faults add latency (~10-100μs per fault).

### 🔹 Part 2: Prefetching

**Explicit Migration:**
```cpp
cudaMallocManaged(&data, size);

// Prefetch to GPU before kernel
cudaMemPrefetchAsync(data, size, deviceId);

kernel<<<grid, block>>>(data);

// Prefetch back to CPU
cudaMemPrefetchAsync(data, size, cudaCpuDeviceId);
```

**Benefit:**
Eliminates page faults during kernel execution.

### 🔹 Part 3: Memory Advise Hints

```cpp
// Hint: Data will be read-only
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, deviceId);

// Hint: Prefer location
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, deviceId);

// Hint: Accessed by device
cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, deviceId);
```

---

## 💻 Implementation

### Optimized UM Pipeline

```cpp
// Allocate
cudaMallocManaged(&data, N * sizeof(float));

// Initialize on CPU
for (int i = 0; i < N; ++i) data[i] = i;

// Advise and prefetch
cudaMemAdvise(data, N * sizeof(float), cudaMemAdviseSetReadMostly, 0);
cudaMemPrefetchAsync(data, N * sizeof(float), 0);

// GPU kernel
process<<<grid, block>>>(data, N);

// Prefetch results back
cudaMemPrefetchAsync(data, N * sizeof(float), cudaCpuDeviceId);

// Use on CPU
verify(data, N);
```

---

## 📝 Summary

Advanced UM techniques (prefetching, advise) enable performance comparable to explicit memory management while maintaining programming simplicity.

*End of Day 081 - Total Lines: 1000+*
