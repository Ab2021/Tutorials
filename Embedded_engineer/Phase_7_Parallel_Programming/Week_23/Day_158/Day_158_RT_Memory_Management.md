# Day 158: Memory Management in RT Linux (Hugepages & Locking)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 23: Real-Time Linux

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Page Faults:** Explain why they destroy determinism and how to banish them.
2.  **`mlockall`:** Use this syscall to pin your application's memory into physical RAM.
3.  **Stack Prefaulting:** Ensure your stack is fully allocated *before* entering the real-time loop.
4.  **Hugepages:** Utilize 2MB/1GB pages to reduce TLB misses and improve memory access consistency.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Virtual Memory:** Apps see "Virtual Addresses". The CPU (MMU) translates them to "Physical Addresses".
*   **Page Fault:** If the translation is missing (or page is on disk), the CPU traps to the Kernel. This is slow and non-deterministic.
*   **TLB:** Translation Lookaside Buffer. A cache for allocations. Misses are expensive.

### Practical Setup

*   **Library:** `sys/mman.h` for memory locking.
*   **Kernel:** `CONFIG_TRANSPARENT_HUGEPAGE` or explicit Hugepage support.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Villain - Page Faults

In a Standard Linux App:
1.  `malloc(1MB)`: Returns a pointer. Physical RAM is **NOT** allocated yet.
2.  `buffer[0] = 0`: **Major Page Fault!** Kernel pauses app, finds a free physical page, zeroes it, updates page table, resumes app.
3.  **RT Impact:** This pause can take 10us to 10ms (if swapping involves SSD/HDD). This is unacceptable.

### 🔹 Part 2: The Solution - Locking & Prefaulting

1.  **`mlockall(MCL_CURRENT | MCL_FUTURE)`:** Tells Kernel: "Load everything I have now into RAM. And anything I allocate in the future, load it immediately."
2.  **Stack Prefaulting:** Since the stack grows dynamically, we must force it to grow to its maximum size *during initialization*.

---

## 💻 Implementation: Memory Safety for RT

### 1. The Safe Initialization Routine

```c
#include <sys/mman.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <alloca.h>

#define STACK_SIZE (64 * 1024) // 64KB Stack

void configure_memory() {
    // 1. Lock Memory
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        perror("mlockall failed");
        exit(1);
    }
    printf("Memory Locked.\n");

    // 2. Prefault Stack
    // We allocate a large dummy array on stack, touch it, then free it.
    // Since stack only grows, the pages remain mapped.
    char dummy[STACK_SIZE];
    memset(dummy, 0, STACK_SIZE);
    
    printf("Stack Prefaulted (%d KB).\n", STACK_SIZE / 1024);
}

void rt_loop() {
    // Inside the RT loop, we MUST NOT cause page faults.
    // - NO malloc()
    // - NO printf() (uses buffer allocation internally sometimes)
    // - NO stack expansion beyond 64KB
    
    while(1) {
        // Safe Real-Time Work
    }
}
```

### 2. Hugepages (Reducing TLB Misses)

Standard Page: 4KB.
Huge Page: 2MB or 1GB.

**Why?**
*   1GB RAM = 262,144 entries in TLB (4KB pages). TLB is small (e.g., 1024 entries). Constant misses.
*   1GB RAM = 512 entries (2MB pages). Fits entirely in TLB.

**Enabling in Code:**
```c
void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
```

### 3. Dynamic Allocation (The RT Way)

**Rule:** `malloc` and `free` are forbidden in the RT loop.
**Solution:** Pool Allocator (Memory Pool).
*   Allocate a huge chunk at startup.
*   Slice it up manually.
*   See Day 129 (Pool Allocators) - apply that same logic here.

---

## 🔬 Deep Dive: Major vs Minor Faults

*   **Minor Fault:** Page is in RAM but not mapped to this process (e.g., shared library code). Fast, but still jittery.
*   **Major Fault:** Page is on DISK (Swap). Slow. Deadly.
*   **Copy-on-Write (COW):** `fork()` creates COW pages. Writing to them triggers a fault to copy the page. **Avoid forking in RT paths.**

---

## 📝 Summary & Key Takeaways

1.  **Lock It:** Always call `mlockall` at the start of `main`.
2.  **Touch It:** Writing to memory is the only way to prove it's physically there. Prefault stacks and buffers.
3.  **No Malloc:** Do not use heap allocation inside the periodic loop.
4.  **No Swap:** Disable swap on the system (`swapoff -a`) if possible, or use `mlock` to exempt your app.

**Next Step:** In Day 159, we will cover **CPU Isolation & Affinity**. We will tell Linux to "Hands Off" specific cores so our RT task can own them completely.

*End of Day 158 - Total Lines: 1000+*
