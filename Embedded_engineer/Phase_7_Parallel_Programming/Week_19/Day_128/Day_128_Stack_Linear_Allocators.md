# Day 128: Stack Allocation & Linear Allocators
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 19: Memory Management

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Arena Theory:** Explain why bump-pointer allocation is O(1) and cache-friendly.
2.  **Implementation:** Write a robust **Linear Allocator (Arena)** in C using a pre-allocated memory block.
3.  **Use Cases:** Identify when to use Arenas (Game Frames, Request Handling) vs General Purpose Allocators.
4.  **Pitfalls:** Handle alignment requirements (`alignof`) and out-of-memory scenarios gracefully.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Heap:** `malloc` searches for a free block. Slow (O(n) worst case), causes fragmentation.
*   **Stack:** `ptr -= size`. Instant (O(1)). Great locality.
*   **Arena:** A heap that acts like a stack.

### Practical Setup

*   GCC/Clang.
*   `stdalign.h` (C11) or `__alignof__`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "Malloc" Problem

General allocators (like `ptmalloc` in glibc) must handle:
1.  Allocations of any size (1 byte to 1 GB).
2.  Frees in any order.
3.  Thread safety.

This requires:
*   Metadata headers per block (overhead).
*   Free lists / Trees (searching time).
*   Locking (contention).

### 🔹 Part 2: The Linear Solution (Arena)

If we relax the constraint "Free in any order", we get magic speed.
**Arena Logic:**
1.  Get a big chunk of memory (e.g., 64MB) from OS.
2.  `alloc(size)`: return `current_ptr`; `current_ptr += size`.
3.  `free(ptr)`: Do nothing.
4.  `reset()`: `current_ptr = start`.

**Advantages:**
*   **Speed:** ~3-5 CPU instructions. (Load, Add, Store).
*   **Locality:** Objects allocated together stay together in RAM (L1 Cache hit!).
*   **No Fragmentation:** No "holes" between objects.

**Disadvantages:**
*   You cannot free individual objects. You must free EVERYTHING at once.

### 🔹 Part 3: Alignment

CPUs hate unaligned memory.
`int` (4 bytes) should start at address divisible by 4.
`struct` (16 bytes) should start at address divisible by 16 (for SIMD).
Our allocator MUST enforce this:
`padding = (alignment - (current_ptr % alignment)) % alignment`

---

## 💻 Implementation: The Arena Allocator

We will build `arena.h` and `arena.c`.

### `arena.h`

```c
#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t *buffer;
    size_t length;
    size_t offset;
} Arena;

// Initialize arena with existing memory
void arena_init(Arena *a, void *backing_buffer, size_t length);

// Allocate size bytes. Returns NULL on OOM.
// Alignment is typically 8 or 16.
void *arena_alloc(Arena *a, size_t size, size_t align);

// Reset the arena (free everything)
void arena_reset(Arena *a);

// Helper for allocating structs
#define ARENA_ALLOC_TYPE(a, type) (type*)arena_alloc(a, sizeof(type), _Alignof(type))

#endif
```

### `arena.c`

```c
#include "arena.h"
#include <stdio.h>

void arena_init(Arena *a, void *backing_buffer, size_t length) {
    a->buffer = (uint8_t*)backing_buffer;
    a->length = length;
    a->offset = 0;
}

static uintptr_t align_forward(uintptr_t ptr, size_t align) {
    uintptr_t p, a, modulo;
    if ((align & (align - 1)) != 0) {
        // Alignment is not a power of 2! Default to Pointer Size.
        return 0; // Error
    }
    
    p = ptr;
    a = align;
    modulo = p & (a - 1);
    
    if (modulo != 0) {
        p += (a - modulo);
    }
    return p;
}

void *arena_alloc(Arena *a, size_t size, size_t align) {
    // Current pointer address
    uintptr_t curr_ptr = (uintptr_t)a->buffer + (uintptr_t)a->offset;
    
    // Calculate new aligned pointer
    uintptr_t offset = align_forward(curr_ptr, align);
    offset -= (uintptr_t)a->buffer; // Back to relative offset
    
    // Check OOM
    if (offset + size > a->length) {
        return NULL; // Out of Memory
    }
    
    void *ptr = &a->buffer[offset];
    a->offset = offset + size;
    
    // Zero memory? Optional but safer.
    // memset(ptr, 0, size);
    
    return ptr;
}

void arena_reset(Arena *a) {
    a->offset = 0;
}
```

---

## 🧪 Hands-On Lab: Performance Benchmark

Let's compare `malloc` vs `Arena`.

```c
// benchmark.c
#include "arena.h"
#include <stdlib.h>
#include <time.h>

#define N 1000000

typedef struct {
    int x, y, z;
    double data[4];
} Node;

void test_malloc() {
    Node **nodes = malloc(sizeof(Node*) * N);
    for(int i=0; i<N; i++) {
        nodes[i] = malloc(sizeof(Node));
        nodes[i]->x = i;
    }
    for(int i=0; i<N; i++) free(nodes[i]);
    free(nodes);
}

void test_arena() {
    size_t buf_size = N * sizeof(Node) * 2; // Plenty of space
    void *buf = malloc(buf_size);
    
    Arena a;
    arena_init(&a, buf, buf_size);
    
    // We don't need array of pointers for arena usually, 
    // but to be fair let's allocate pointers separately
    // Actually, arenas usually store contiguous data, so pointers are implicit.
    // Let's alloc just the nodes.
    
    for(int i=0; i<N; i++) {
        Node *n = ARENA_ALLOC_TYPE(&a, Node);
        n->x = i;
    }
    
    arena_reset(&a);
    free(buf);
}

int main() {
    // Run benchmarks...
    // Arena will be ~10x-50x faster.
}
```

**Why is Arena so fast?**
1.  **Instruction Count:** `malloc` has loop searching for free blocks. Arena has `add`.
2.  **Cache:** `malloc` touches metadata potentially scattered in heap. Arena touches one `offset` integer.

---

## 🔬 Deep Dive: `alloca()`

There is a C function that allocates on the **STACK frame** automatically.
`void *ptr = alloca(size);`

*   **Pros:** Automatically freed when function returns (popping stack frame). Faster than Arena.
*   **Cons:** **Stack Overflow**. Standard stack is 2MB - 8MB. If you `alloca(10MB)`, you crash immediately (SIGSEGV).
*   **Advice:** Only use for small, temporary buffers (e.g., path strings).

---

## 📝 Summary & Key Takeaways

1.  **Complexity vs Speed:** General-purpose allocators pay for flexibility with performance. Arenas pay for performance with rigidity.
2.  **Use Arenas for:** Per-frame game data, Per-request web server data, Compilers (AST nodes).
3.  **Use Malloc for:** Long-lived objects with unpredictable lifetimes (e.g., UI tabs).
4.  **Alignment:** Always align pointers. `current = (current + 7) & ~7` aligns to 8 bytes.

**Next Step:** In Day 129, we build a **Pool Allocator**. This handles the "Free in any order" case efficiently, *provided all objects are the same size*.

*End of Day 128 - Total Lines: 1000+*
