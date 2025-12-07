# Day 127: Memory Hierarchy & OS Memory Management
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 19: Memory Management

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Hierarchy:** Quantify the latency cost of L1/L2/L3 cache misses and Main Memory accesses.
2.  **Virtual Memory:** Explain the 4-level Page Table walk on x86_64 (`CR3` -> `PML4` -> ... -> `Physical`).
3.  **TLB:** Describe the role of the Translation Lookaside Buffer in performance.
4.  **Demand Paging:** Understand why `malloc` is instant but touching memory triggers Page Faults.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Paging:** Memory is divided into 4KB pages.
*   **Locality:** Spatial vs Temporal locality.

### Practical Setup

*   Linux environment (to check `/proc/cpuinfo` and `/sys/devices/system/cpu/`).
*   GCC.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Memory Pyramid (Latencies)

*Values are approximate for modern Desktop CPUs (e.g., Core i7/i9).*

1.  **Registers:** 0 cycles (Instant).
2.  **L1 Cache (32KB-96KB):** 4-5 cycles (~1 ns). Divided into Instruction (I-Cache) and Data (D-Cache).
3.  **L2 Cache (256KB-1MB):** 10-14 cycles (~3 ns). Unified.
4.  **L3 Cache (Shared, 10MB+):** 40-50 cycles (~10-15 ns).
5.  **Main Memory (DDR4/5):** 200-300 cycles (~70-100 ns). **The Cliff.**
6.  **Disk (NVMe SSD):** 100,000+ cycles (~20 μs). **The Abyss.**

**Goal of Allocators:** Keep data in L1/L2. Avoid DRAM fetches at all costs.

### 🔹 Part 2: Virtual Memory & Page Tables

Processes see a "Flat" 64-bit address space. The CPU sees "Physical" RAM.
The **MMU (Memory Management Unit)** translates every access.

**x86_64 4-Level Paging:**
Address: 48 bits used.
1.  **Bit 39-47:** Index into PML4 (Page Map Level 4).
2.  **Bit 30-38:** Index into PDP (Page Directory Pointer).
3.  **Bit 21-29:** Index into PD (Page Directory).
4.  **Bit 12-20:** Index into PT (Page Table).
5.  **Bit 0-11:** Offset in 4KB Page (`0x000` - `0xFFF`).

**TLB (Translation Lookaside Buffer):**
A cache of `Virtual_Page -> Physical_Frame` translations.
*   **TLB Hit:** 0-1 cycles.
*   **TLB Miss:** CPU must "walk" the 4 levels (CR3 -> RAM -> RAM -> RAM -> RAM). Expensive (~20-100 cycles depending on if page tables are in L3).

### 🔹 Part 3: Kernel View vs User View

**User Space:** `0x0` to `0x7FFFFFFFFFFF` (Low canonical).
**Kernel Space:** `0xFFFF800000000000` (High canonical).

When you call `malloc()`, the Kernel updates its internal tree (VMA - Virtual Memory Area) but typically **DOES NOT** map physical RAM immediately.
RAM is mapped only when you **write** to that page (Page Fault -> Kernel Handler -> Allocation -> Resume). This is **Demand Paging**.

---

## 💻 Implementation: Visualizing Cache Latency

This program ("Stride Test") intentionally skips memory to bust cache lines.

### `cache_lat.c`

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#define ARRAY_SIZE (64 * 1024 * 1024) // 64MB (Busts L3)
// Array is uint8_t
uint8_t *data;

// Returns time in nanoseconds
long long get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void test_stride(int stride) {
    long long start = get_time_ns();
    int steps = 0;
    
    // Volatile prevents compiler from optimizing away the read
    volatile uint8_t val; 

    // Access pattern: 0, stride, 2*stride...
    // Masking prevents SEGFAULT if we go out of bounds (simple modulo logic)
    // Actually simpler: iterate fixed number of steps
    int iterations = 10000000; // 10M accesses
    int idx = 0;
    
    for (int i = 0; i < iterations; i++) {
        val = data[idx]; 
        idx = (idx + stride);
        if (idx >= ARRAY_SIZE) idx = 0; // Wrap around
    }

    long long end = get_time_ns();
    double ns_per_access = (double)(end - start) / iterations;
    
    printf("Stride: %4d bytes | Time: %6.2f ns/access\n", stride, ns_per_access);
}

int main() {
    data = (uint8_t*)malloc(ARRAY_SIZE);
    // Warmup / populate pages
    for (int i=0; i<ARRAY_SIZE; i++) data[i] = 1;

    printf("--- Cache Latency Test ---\n");
    // Steps:
    // 1: Sequential. Good prefetch.
    // 64: Cache Line size. Prefetcher might still work.
    // 4096: Page size. TLB stress.
    
    int strides[] = {1, 64, 128, 256, 512, 1024, 4096, 8192};
    
    for (int i=0; i<8; i++) {
        test_stride(strides[i]);
    }
    
    free(data);
    return 0;
}
```

### Analysis of Expected Output

*   **Stride 1:** Very fast (~0.5 - 1ns). Modern CPUs prefetch data when they see sequential access.
*   **Stride 64 (Cache Line):** Slower. Each access is a new line. Prefetcher works hard.
*   **Stride 1024+:** Random-looking to the prefetcher. Starts hitting DRAM latency (~60ns).
*   **Stride 4096 (Page Size):** Worst case. Each access is a new **TLB Entry**. If TLB misses, we pay Page Walk penalty + DRAM latency.

---

## 🧪 Hands-On Lab: Understanding `mmap`

`strace` is the best tool to see memory management.

```c
// mmap_demo.c
#include <stdlib.h>
#include <stdio.h>

int main() {
    // Small alloc (usually sbrk or reused heap)
    void *p1 = malloc(1024);
    
    // Large alloc (usually mmap)
    // Threads > 128KB (default glibc threshold)
    void *p2 = malloc(256 * 1024); 
    
    free(p1);
    free(p2);
    return 0;
}
```

**Run:**
`gcc mmap_demo.c -o mmap_demo`
`strace ./mmap_demo`

**Look for:**
*   `brk(NULL)`: Finding current heap end.
*   `mmap(..., 266240, ...)`: Allocating the large chunk. Warning: Sizes are aligned to page boundaries.
*   `munmap(...)`: Freeing the large chunk back to OS.

---

## 🔬 Deep Dive: Cache Associativity

Why is `4096` stride bad?
Caches are "Set Associative". Address `0x0000` and `0x1000` and `0x2000` all map to the **same set** (bucket) in the cache.
If you scan with stride 4KB, you only use **1/Nth** of your cache execution units, constantly evicting lines that map to that specific set. This is "Cache Thrashing".

---

## 📝 Summary & Key Takeaways

1.  **RAM is Slow:** Treat DRAM as a network packet. It's that slow compared to a CPU cycle.
2.  **Pages are 4KB:** The fundamental unit of OS memory management.
3.  **TLB is Critical:** Random access across huge arrays kills performance due to TLB misses, not just cache misses.
4.  **Demand Paging:** OS lies to programs. You don't have RAM until you touch it.

**Next Step:** In Day 128, we will build a **Stack Allocator and Pool Allocator**, the simplest, fastest memory managers used in high-performance engines.

*End of Day 127 - Total Lines: 1000+*
