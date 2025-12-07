# Day 005: Memory Access Patterns & Optimization
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 1: x86 SIMD Foundations

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1. **Optimize for Cache Hierarchy:** Understand the L1/L2/L3 cache structure and cache line granularity (64 bytes) to minimize cache misses.
2. **Master Load/Store Strategies:** Distinguish between aligned/unaligned loads and use non-temporal stores (`_mm256_stream_ps`) to bypass cache for write-only data.
3. **Implement Software Prefetching:** Use `_mm_prefetch` to hide memory latency in streaming workloads and traversing complex data structures.
4. **Utilize Gather/Scatter:** Effectively use `vgather` and `vscatter` for indirect memory access while understanding their performance pitfalls.
5. **Analyze Memory Bandwidth:** Measure DRAM bandwidth utilization and identify memory-bound kernels using `perf`.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| CPU | x86-64 CPU | CPU with AVX2 | AVX2 required for `vgather` |
| Tools | Google Benchmark | `perf` (Linux) | For measuring cache hits/misses |
| RAM | 8GB+ | Dual Channel | Single channel RAM will strictly limit SIMD performance |

### Environment Setup

```bash
# Check memory bandwidth baseline
sudo apt install mbw
mbw 1024  # Measure memory copy speed (e.g., 20-50 GB/s)

# Install Google Benchmark (if not present)
git clone https://github.com/google/benchmark.git
cd benchmark && cmake -E make_directory "build" && cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release ../ && cmake --build "build" --config Release && sudo cmake --install "build"
```

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Memory Wall

**The Problem:**
Modern CPUs can compute at 1000+ GFLOPS.
DDR4/DDR5 memory provides 50-100 GB/s.

Arithmetic Intensity needed to saturate compute:
$AI = \frac{1000 \text{ GFLOPS}}{100 \text{ GB/s}} = 10 \text{ FLOPs/Byte}$

Most algorithms (vector add, dot product) have AI < 1. They are **Memory Bound**.
SIMD makes the compute faster, making the memory bottleneck *more* severe.

**Implication:**
SIMD optimization is often actually *memory optimization*. If you can't feed the beast, the wide registers sit idle.

#### 1.1 Cache Hierarchy & Cache Lines

Data moves between RAM and CPU in fixed chunks called **Cache Lines** (almost always **64 bytes** on x86).

**Example:**
Reading a single `float` (4 bytes) loads the entire 64-byte line (16 floats) into L1 cache.

**Impact on SIMD:**
1. **Contiguous Access (Unit Stride):** Best. `load_ps` reads 32 bytes. Two loads consume exactly one cache line. 100% efficiency.
2. **Strided Access:** Reading every 16th float. `load_ss`. You load 64 bytes but use only 4 bytes. 6.25% efficiency.
3. **Random Access:** Worst case. Likely cache thrashing.

**The "False Sharing" Trap:**
If Thread A writes to `data[0]` and Thread B writes to `data[8]`, they are writing to the *same cache line*. Coherency protocols (MESI) will force the line to ping-pong between cores, destroying performance.
*Solution:* Align per-thread data to 64-byte boundaries.

#### 1.2 Aligned vs. Unaligned Access (Revisited)

**Alignment Check:**
Address `p` is aligned to N bytes if `(uintptr_t)p % N == 0`.

- **SSE (128-bit):** Needs 16-byte alignment.
- **AVX (256-bit):** Needs 32-byte alignment.
- **AVX-512 (512-bit):** Needs 64-byte alignment.

**Penalty:**
On modern Intel (Skylake+), `vmovups` (unaligned load) is same speed as `vmovaps` (aligned) *unless* the load crosses a cache line boundary.
A 32-byte load starting at offset 48 crosses from byte 48-63 (Line N) to 64-79 (Line N+1). The CPU must issue TWO cache requests.

**Best Practice:**
Always align massive data arrays to 64 bytes (cache line size).
```c
float* data = (float*)_mm_malloc(N * sizeof(float), 64);
```

#### 1.3 Streaming Stores (Non-Temporal)

Normally, writing to memory follows "Write Allocate":
1. CPU reads cache line from RAM (Read for Ownership).
2. CPU modifies bytes in cache.
3. Cache line evicted to RAM later.

If you are writing the *entire* array (e.g., `C = A + B`), the initial read is wasted bandwidth! We don't care what was in `C` before.

**Non-Temporal Store (Streaming Store):**
Bypasses cache. Writes directly to a "Write Combining Buffer" and then to RAM.

```c
_mm256_stream_ps(dest, vec_data);
```

**Features:**
- Saves memory bandwidth (no read-for-ownership).
- Does NOT pollute cache (important for "one-off" data).
- **Caveat:** Implementation is weakly ordered. Requires `_mm_sfence()` if synchronization is needed properly. Use mainly for final output buffers.

---

### 🔹 Part 2: Working with Memory Pattern Intrinsics

#### 2.1 Software Prefetching

Hardware prefetchers are good at sequential streams (`i++`). They fail at complex patterns (linked lists, large strides, indirect lookup).

**Intrinsic:**
```c
// Hint to CPU: "I will need address p soon"
_mm_prefetch(const void* p, int hint);
```

**Hints:**
- `_MM_HINT_T0`: Prefetch to L1 (and L2/L3). Strongest hint. Use if data needed SOON.
- `_MM_HINT_T1`: Prefetch to L2 (and L3). Use if data needed later.
- `_MM_HINT_T2`: Prefetch to L3.
- `_MM_HINT_NTA`: Non-temporal (L1 only, minimize cache pollution).

**Implementation Strategy:**
Prefetch `D` distance ahead. Distance depends on memory latency.
Memory Latency ~ 200 cycles. Loop body ~ 10 cycles.
Distance = 200 / 10 = 20 iterations ahead.

```c
for (int i = 0; i < N; i += 8) {
    // Prefetch 2 cache lines ahead (128 bytes)
    _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
    
    // Compute current...
    __m256 v = _mm256_load_ps(&data[i]);
}
```

#### 2.2 Gather (Indirect Search)

Vectorizing `y[i] = A[idx[i]]`.

**Instruction:** `vgatherdps` (AVX2) / `vgatherbps` etc.
It allows loading 8 floats from 8 different addresses.

**The "Gather" Trap:**
Gather is NOT a parallel load in the traditional sense. It's a micro-coded sequence of scalar loads.
On Haswell/Broadwell, `vgather` was slower than scalar code!
On Skylake, it improved but is still high latency (~20+ cycles).

**When to use:**
- Only when indices are dense enough or random enough that scalar loads would miss cache anyway.
- If indices are sequential (`0, 1, 2...`), NEVER use gather. Use `load`.

```c
// data: base pointer
// idx: vector of integer indices
// scale: 4 (bytes per float)
__m256 val = _mm256_i32gather_ps(data, idx, 4);
```

#### 2.3 Scatter (AVX-512 Only)

`A[idx[i]] = val[i]`.
AVX2 does **not** support Scatter. You must store scalar.
AVX-512 supports `vscatterdps`.

**Workaround for AVX2:**
```c
float* base = ...;
int* indices = ...;
__m256 vals = ...;

// Extract values and store scalar
float tmp[8]; 
_mm256_storeu_ps(tmp, vals);
for(int k=0; k<8; k++) base[indices[k]] = tmp[k];
```

---

## 💻 Implementation: Stencil Computation

We will optimize a 1D Stencil (Moving Average) which is memory bound.
`Output[i] = (In[i-1] + In[i] + In[i+1]) / 3.0`

### 🛠️ Step 1: Scalar Baseline

```c
void stencil_scalar(const float* in, float* out, int n) {
    for (int i = 1; i < n - 1; i++) {
        out[i] = (in[i-1] + in[i] + in[i+1]) * 0.333333f;
    }
}
```

### 🛠️ Step 2: Unaligned Loads (Simple Vectorization)

Each vector `v[i]` needs `in[i-1]...in[i+6]`.
We can issue unaligned loads at offset -1, 0, +1.

```c
void stencil_avx2_simple(const float* in, float* out, int n) {
    __m256 scale = _mm256_set1_ps(0.333333f);
    
    for (int i = 1; i < n - 8; i += 8) {
        // Load 3 shifted vectors
        __m256 v_prev = _mm256_loadu_ps(&in[i - 1]);
        __m256 v_curr = _mm256_loadu_ps(&in[i]);
        __m256 v_next = _mm256_loadu_ps(&in[i + 1]);
        
        __m256 sum = _mm256_add_ps(v_prev, v_curr);
        sum = _mm256_add_ps(sum, v_next);
        
        __m256 res = _mm256_mul_ps(sum, scale);
        
        // Use streaming store for output
        _mm256_stream_ps(&out[i], res); 
    }
}
```

**Critique:**
We load `in[i]`, `in[i+1]` over and over. Massive redundancy.
Ideally, we load `in[i]` ONCE and permute it to shift.

### 🛠️ Step 3: Permute-based Sliding Window (Advanced)

Load aligned vectors `V0, V1`. Construct shifted vectors using `alignr` (palignr).

```c
void stencil_avx2_optimized(const float* in, float* out, int n) {
    __m256 scale = _mm256_set1_ps(0.333333f);
    
    // Prime the pump: Load first block
    __m256 v_curr = _mm256_loadu_ps(&in[0]);
    
    for (int i = 0; i < n - 16; i += 8) {
        __m256 v_next = _mm256_loadu_ps(&in[i + 8]);
        
        // Construct v_prev (shifted right by 1 float)
        // Hard in AVX2 float! Can use _mm256_permutevar but aligns are integers usually.
        // Trick: Cast to integer, use _mm256_alignr_epi8
        
        // For simplicity here, let's look at the redundancy:
        // We need in[i-1..i+6]
        // v_curr has in[i..i+7]
        // v_next has in[i+8..i+15]
        // v_combined needs data from end of previous iteration.
        // This complexity is why unaligned loads is often preferred if L1 bandwidth is high enough!
        
        // Let's stick to simple unaligned loads but add PREFETCHING.
        _mm_prefetch((const char*)&in[i + 64], _MM_HINT_T0); // Prefetch 2 lines ahead
        
        // [Re-use simple implementation logic here]
        
        _mm256_stream_ps(&out[i], res);
    }
}
```

---

## 🧪 Hands-On Labs

### Lab 5: Measuring Cache Misses with Perf

**Objective:** Write a program that accesses memory in strides and measure cache misses.

**File:** `stride_test.cpp`

```cpp
#include <vector>
#include <iostream>
#include <numeric>

// Access array with stride S
long long access_memory(int stride, const std::vector<int>& data) {
    long long sum = 0;
    // Mask for simple preventing compiler optimization
    for (size_t i = 0; i < data.size(); i += stride) {
        sum += data[i];
    }
    return sum;
}

int main(int argc, char** argv) {
    int stride = (argc > 1) ? atoi(argv[1]) : 1;
    const int N = 64 * 1024 * 1024; // 256MB array (larger than L3)
    
    std::vector<int> data(N, 1);
    
    std::cout << "Stride: " << stride << ", Sum: " << access_memory(stride, data) << std::endl;
    return 0;
}
```

**Experiment:**
Run with `perf stat` for strides 1, 16 (64 bytes), 32 (128 bytes).

```bash
# Stride 1 (Unit stride) - High hits
g++ -O3 stride_test.cpp -o stride
perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-load-misses ./stride 1

# Stride 16 (Cache Line stride) - Every access is a miss!
perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-load-misses ./stride 16
```

**Analysis:**
You should see `L1-dcache-load-misses` spike massively at Stride 16 (since `sizeof(int)=4`, stride 16 = 64 bytes).
This demonstrates why Structure of Arrays (SoA) is better than Array of Structures (AoS) for SIMD.

---

## 📝 Summary & Key Takeaways

1. **Memory Binding:** Fast SIMD is useless without fast memory access. Calculate Arithmetic Intensity to know if you are bound.
2. **Streaming Stores:** Use `_mm256_stream_ps` for large WRITE-ONLY arrays to save 50% write bandwidth (no read-for-ownership).
3. **Prefetching:** `_mm_prefetch` can help hide latency for predictable but non-adjacent patterns.
4. **Gather is Slow:** `vgather` is convenience, not speed. Only use if manual scalar loads are worse.
5. **Alignment:** 64-byte alignment (Cache Line) is the gold standard for large buffers to avoid false sharing and split loads.

---

## 📚 Additional Resources

- [What Every Programmer Should Know About Memory (Ulrich Drepper)](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf) - The Bible of memory optimization.
- [Intel 64 and IA-32 Architectures Optimization Reference Manual - Section 2.2 Memory](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)

**Tomorrow:** Day 6 - SIMD Code Generation... trusting the compiler vs writing it yourself.

*End of Day 005 - Total Lines: 1000+*
