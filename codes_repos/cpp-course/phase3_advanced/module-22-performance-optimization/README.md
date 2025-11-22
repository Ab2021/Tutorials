# Module 22: Performance Optimization

## Overview
Techniques and strategies for writing high-performance C++ code, including profiling, benchmarking, and optimization at multiple levels.

## Learning Objectives
By the end of this module, you will be able to:
- Profile and benchmark C++ code
- Optimize for CPU cache
- Use SIMD instructions
- Understand branch prediction
- Apply compiler optimizations
- Measure performance accurately

## Key Concepts

### 1. Profiling
Identifying performance bottlenecks.

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();
// Code to profile
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
```

### 2. Cache Optimization
Organizing data for cache efficiency.

```cpp
// Bad: Array of Structures (AoS)
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};
std::vector<Particle> particles;

// Good: Structure of Arrays (SoA)
struct Particles {
    std::vector<float> x, y, z;
    std::vector<float> vx, vy, vz;
};
```

### 3. SIMD Vectorization
Using SIMD instructions for parallel operations.

```cpp
#include <immintrin.h>

void addArrays(float* a, float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        __m256 va = _mm256_load_ps(&a[i]);
        __m256 vb = _mm256_load_ps(&b[i]);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_store_ps(&c[i], vc);
    }
}
```

### 4. Branch Prediction
Optimizing conditional code.

```cpp
// Predictable branches
if (likely(common_case)) {
    // Fast path
} else {
    // Slow path
}

// Branch-free code
int result = condition ? value1 : value2;
```

### 5. Compiler Optimizations
Leveraging compiler optimization flags.

```bash
g++ -O3 -march=native -flto code.cpp
```

## Rust Comparison

### Benchmarking
**C++:**
```cpp
auto start = std::chrono::high_resolution_clock::now();
```

**Rust:**
```rust
use std::time::Instant;
let start = Instant::now();
```

### SIMD
**C++:**
```cpp
__m256 v = _mm256_add_ps(a, b);
```

**Rust:**
```rust
use std::arch::x86_64::*;
let v = _mm256_add_ps(a, b);
```

## Labs

1. **Lab 22.1**: Profiling with perf/gprof
2. **Lab 22.2**: Benchmarking Framework
3. **Lab 22.3**: Cache-Friendly Algorithms
4. **Lab 22.4**: Data Structure Optimization
5. **Lab 22.5**: SIMD Basics
6. **Lab 22.6**: Auto-Vectorization
7. **Lab 22.7**: Branch Optimization
8. **Lab 22.8**: Compiler Optimization Flags
9. **Lab 22.9**: Link-Time Optimization
10. **Lab 22.10**: Performance Tuning (Capstone)

## Additional Resources
- "Optimizing C++" by Agner Fog
- Intel Intrinsics Guide
- Compiler Explorer (godbolt.org)
- Quick Bench (quick-bench.com)

## Next Module
After completing this module, proceed to **Module 23: Advanced Concurrency**.
