# Day 244: RT Memory Management
## Phase 2: Linux Kernel & Device Drivers | Week 36: Real-Time Systems

---

## 🎯 Learning Objectives
1. **Understand** memory-related latencies in RT systems
2. **Implement** memory locking (mlockall)
3. **Avoid** page faults in RT paths
4. **Configure** NUMA for RT workloads
5. **Measure** memory access latency

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Memory Latency Sources

**Page Faults:**
- Major fault: Load from disk (milliseconds!)
- Minor fault: Allocate/map page (microseconds)
- Both unacceptable for hard RT

**Solutions:**
- Lock all memory (`mlockall`)
- Prefault stack and heap
- Use huge pages
- Disable swap

### 🔹 Part 2: Memory Locking

```c
#include <sys/mlock.h>

// Lock all current and future memory
if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return -1;
}

// Prefault stack
void prefault_stack(void) {
    unsigned char dummy[MAX_STACK_SIZE];
    memset(dummy, 0, MAX_STACK_SIZE);
}

// Prefault heap
void *rt_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr) {
        memset(ptr, 0, size);  // Touch all pages
    }
    return ptr;
}
```

### 🔹 Part 3: NUMA Considerations

```bash
# Check NUMA topology
numactl --hardware

# Bind RT process to specific NUMA node
numactl --cpunodebind=0 --membind=0 ./rt_app

# In code
#include <numa.h>

void setup_numa(void) {
    if (numa_available() < 0) {
        printf("NUMA not available\n");
        return;
    }
    
    // Bind to node 0
    struct bitmask *mask = numa_allocate_nodemask();
    numa_bitmask_setbit(mask, 0);
    numa_bind(mask);
    numa_free_nodemask(mask);
}
```

---

## 💻 Implementation Examples

### Example 1: Complete RT Memory Setup

```c
#include <sys/mlock.h>
#include <sys/resource.h>
#include <malloc.h>
#include <stdio.h>

#define STACK_SIZE (8*1024*1024)
#define HEAP_SIZE (64*1024*1024)

int setup_rt_memory(void) {
    struct rlimit rlim;
    
    // 1. Increase locked memory limit
    rlim.rlim_cur = RLIM_INFINITY;
    rlim.rlim_max = RLIM_INFINITY;
    if (setrlimit(RLIMIT_MEMLOCK, &rlim) != 0) {
        perror("setrlimit");
        return -1;
    }
    
    // 2. Lock all memory
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall");
        return -1;
    }
    
    // 3. Prefault stack
    unsigned char stack_dummy[STACK_SIZE];
    memset(stack_dummy, 0, STACK_SIZE);
    
    // 4. Preallocate heap
    void *heap = malloc(HEAP_SIZE);
    if (!heap) {
        perror("malloc");
        return -1;
    }
    memset(heap, 0, HEAP_SIZE);
    
    // 5. Disable malloc trimming
    mallopt(M_TRIM_THRESHOLD, -1);
    mallopt(M_MMAP_MAX, 0);
    
    printf("RT memory setup complete\n");
    return 0;
}
```

### Example 2: Huge Pages for RT

```c
#include <sys/mman.h>
#include <fcntl.h>

#define HUGE_PAGE_SIZE (2*1024*1024)  // 2MB

void *alloc_huge_page(size_t size) {
    void *ptr;
    
    // Round up to huge page size
    size = (size + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1);
    
    // Allocate with huge pages
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    
    if (ptr == MAP_FAILED) {
        perror("mmap huge page");
        return NULL;
    }
    
    // Touch all pages
    memset(ptr, 0, size);
    
    return ptr;
}

// Configure huge pages (run as root)
void setup_huge_pages(int nr_pages) {
    FILE *fp = fopen("/proc/sys/vm/nr_hugepages", "w");
    if (fp) {
        fprintf(fp, "%d\n", nr_pages);
        fclose(fp);
    }
}
```

### Example 3: Memory Access Latency Measurement

```c
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE (64*1024*1024)  // 64MB

void measure_memory_latency(void) {
    char *array;
    struct timespec start, end;
    long long latency_ns;
    int iterations = 1000000;
    
    // Allocate and lock memory
    array = malloc(ARRAY_SIZE);
    memset(array, 0, ARRAY_SIZE);
    mlock(array, ARRAY_SIZE);
    
    // Measure sequential access
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        volatile char dummy = array[i % ARRAY_SIZE];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    latency_ns = (end.tv_sec - start.tv_sec) * 1000000000LL +
                 (end.tv_nsec - start.tv_nsec);
    
    printf("Sequential access: %.2f ns per access\n",
           (double)latency_ns / iterations);
    
    // Measure random access
    srand(time(NULL));
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        int idx = rand() % ARRAY_SIZE;
        volatile char dummy = array[idx];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    latency_ns = (end.tv_sec - start.tv_sec) * 1000000000LL +
                 (end.tv_nsec - start.tv_nsec);
    
    printf("Random access: %.2f ns per access\n",
           (double)latency_ns / iterations);
    
    munlock(array, ARRAY_SIZE);
    free(array);
}
```

---

## 🔬 Lab Exercises

### Lab 1: Page Fault Detection

```bash
# Monitor page faults
perf stat -e page-faults ./rt_app

# Detailed page fault tracing
perf record -e page-faults -g ./rt_app
perf report
```

### Lab 2: Memory Bandwidth Test

```bash
# Install mbw (memory bandwidth benchmark)
sudo apt-get install mbw

# Test memory bandwidth
mbw 100

# Test with NUMA binding
numactl --cpunodebind=0 --membind=0 mbw 100
```

---

## 🧠 Assessment

**Q:** Why is mlockall important for RT applications?
**A:** Prevents page faults which can cause unbounded latencies (disk I/O).

**Q:** What is the difference between MCL_CURRENT and MCL_FUTURE?
**A:** MCL_CURRENT locks currently mapped pages. MCL_FUTURE locks all future mappings.

**Q:** Why use huge pages for RT?
**A:** Reduces TLB misses, improves memory access predictability, reduces page table overhead.

---

## 🎓 Summary

Covered RT memory management: memory locking, page fault avoidance, NUMA optimization, and huge pages for deterministic memory access.

---

## 🚀 Next Steps

Day 245: Week 36 Review and Project - Building a Complete RT Control System

---
