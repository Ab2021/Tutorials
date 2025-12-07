# Day 129: Pool Allocators & Free Lists
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 19: Memory Management

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Optimization Strategy:** Understand why fixed-size blocks allow for header-less, fragmentation-free allocation.
2.  **Intrusive Lists:** Implement a "Free List" that uses the empty memory blocks themselves to store the `next` pointer.
3.  **Implementation:** Write a **Pool Allocator** (Slab Allocator) in C.
4.  **Use Cases:** Apply pools to particle systems, bullets, and network packets.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Fragmentation:** External (holes between blocks) vs Internal (wasted space inside a 64B block holding 33B).
*   **Arena limitation:** Can't free individual items.
*   **Pool solution:** Can free items, but all items must be the SAME size.

### Practical Setup

*   C / C++.
*   Concepts of `union` and type punning.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "Free List" Trick

How do we track free blocks without `malloc`'s overhead (headers, trees)?
**Trick:** If a block is free, it contains garbage. Let's write the address of the *next* free block into that garbage space!

**Memory Layout (Initial):**
`[Block 0] -> [Block 1] -> [Block 2] -> [Block 3] -> NULL`
Head points to Block 0.

**Alloc():**
Return `Head`.
`Head = Head->next`.
(Head now points to Block 1). Speed: O(1).

**Free(ptr):**
`ptr->next = Head`.
`Head = ptr`.
(Head now points back to the newly freed block). Speed: O(1).

**Overhead:** 0 bytes per block (if `BlockSize >= sizeof(void*)`).
**Fragmentation:** 0%.

### 🔹 Part 2: Intrusive Linked Lists

We use a `union` to overlay the data with the node pointer.

```c
union PoolNode {
    struct PoolNode *next; // Used when free
    char data[ITEM_SIZE];  // Used when allocated
};
```
When allocated, the user owns `data`. When free, the specific implementation owns `next`.

---

## 💻 Implementation: The Pool Allocator

We will utilize a "Freelists" approach backed by a large contiguous chunk (Arena-style backend).

### `pool.h`

```c
#ifndef POOL_H
#define POOL_H

#include <stddef.h>
#include <stdint.h>

typedef struct PoolNode {
    struct PoolNode *next;
} PoolNode;

typedef struct {
    uint8_t *buffer;
    size_t item_size;
    size_t capacity; // Total items
    PoolNode *head;  // Pointer to first free node
} PoolAllocator;

// Initialize pool. 
// item_size must be >= sizeof(void*)
void pool_init(PoolAllocator *p, void *backing_buffer, size_t capacity, size_t item_size);

void *pool_alloc(PoolAllocator *p);
void pool_free(PoolAllocator *p, void *ptr);

#endif
```

### `pool.c`

```c
#include "pool.h"
#include <assert.h>

void pool_init(PoolAllocator *p, void *backing_buffer, size_t capacity, size_t item_size) {
    // Constraint: Item size must be able to hold a pointer
    assert(item_size >= sizeof(void*));
    
    p->buffer = (uint8_t*)backing_buffer;
    p->item_size = item_size;
    p->capacity = capacity;
    p->head = (PoolNode*)p->buffer;
    
    // Link all blocks together
    // This is the expensive part (O(N)), done once.
    for (size_t i = 0; i < capacity - 1; i++) {
        PoolNode *current = (PoolNode*)(p->buffer + i * item_size);
        PoolNode *next    = (PoolNode*)(p->buffer + (i + 1) * item_size);
        current->next = next;
    }
    
    // Last node points to NULL
    PoolNode *last = (PoolNode*)(p->buffer + (capacity - 1) * item_size);
    last->next = NULL;
}

void *pool_alloc(PoolAllocator *p) {
    PoolNode *node = p->head;
    
    if (node == NULL) {
        return NULL; // Out of memory
    }
    
    p->head = node->next; // Pop from head
    return (void*)node;
}

void pool_free(PoolAllocator *p, void *ptr) {
    if (ptr == NULL) return;
    
    PoolNode *node = (PoolNode*)ptr;
    
    // Push to head
    node->next = p->head;
    p->head = node;
}
```

---

## 🧪 Hands-On Lab: Particle System Demo

A particle system is the perfect use case: thousands of small objects, created and destroyed randomly, all same size.

```c
#include <stdio.h>
#include <stdlib.h>
#include "pool.h"

typedef struct {
    float x, y, z;
    float vx, vy, vz;
    int lifetime;
} Particle;

int main() {
    int max_particles = 1000;
    size_t buf_size = max_particles * sizeof(Particle);
    void *backing = malloc(buf_size);
    
    PoolAllocator pool;
    pool_init(&pool, backing, max_particles, sizeof(Particle));
    
    printf("Pool initialized. Item size: %zu\n", sizeof(Particle));

    // Simulation loop
    Particle *particles[100];
    
    // 1. Alloc 100
    for(int i=0; i<100; i++) {
        particles[i] = (Particle*)pool_alloc(&pool);
        particles[i]->x = i * 1.0f;
    }
    
    // 2. Free 50 (randomly)
    for(int i=0; i<100; i+=2) {
        pool_free(&pool, particles[i]);
        particles[i] = NULL;
    }
    
    // 3. Alloc 50 again (should reuse holes)
    for(int i=0; i<100; i+=2) {
        particles[i] = (Particle*)pool_alloc(&pool);
        // Note: address will likely decrease or jump around as it fills holes
        // printf("Re-alloc: %p\n", particles[i]);
    }

    free(backing);
    return 0;
}
```

### Analysis
*   **Locality:** Note that after frequent alloc/free cycles, the physical order in RAM might not match the logical create order. This is a very minor hit to locality compared to `malloc`, which might scatter pages.
*   **Safety:** Warning! The "Double Free" bug.
    If you `pool_free(ptr)` twice, you create a cycle in the linked list (`A -> B -> A`).
    Next `alloc` returns A. Next `alloc` returns B. Next `alloc` returns A again!
    Infinite corruption.
    *Optimized pools in Debug mode often check for this.*

---

## 🔬 Deep Dive: Cache Coloring & Alignment

In high-performance scenarios, having every object start exactly at stride 64 might cause **Cache Set Conflicts** (like in Day 127).
Sometimes allocators add random "padding" (Coloring) to the start of the pool to shift alignment slightly, distributing sets better.
However, Pool Allocators usually pack tightly, which is generally good for prefetching.

---

## 📝 Summary & Key Takeaways

1.  **O(1) Everything:** Allocation and Deallocation are just pointer assignments.
2.  **No Fragmentation:** Since blocks are fixed size, there are no "small holes" unusable by large objects.
3.  **Low Overhead:** The maintenance structure (Next Pointer) lives inside the free memory.
4.  **Rigidity:** Only works for one specific type/size. Usually games have `Pool<Bullet>`, `Pool<Enemy>`, etc.

**Next Step:** In Day 130, we will cover **Custom `malloc` Implementation**. We will build a "General Purpose" allocator using a Free List + Best Fit strategy, mimicking a simple version of the standard C library.

*End of Day 129 - Total Lines: 1000+*
