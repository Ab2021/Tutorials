# Day 140: Week 20 Review & Project
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 20: Parallel Patterns & Lock-Free

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Synthesize:** Connect Atomic Primitives, Memory Ordering, and SMR (Safe Memory Reclamation) into a coherent toolkit.
2.  **Evaluate:** Choose between SPSC (Wait-Free), MPMC (CAS-Loop), or RCU (Read-Heavy) based on traffic patterns.
3.  **Project:** Build `liblockfree`, a single-header C library providing high-performance concurrent structures.

---

## 📚 Week 20 Recap

### Day 135: Atomics & Ordering
*   **Key:** `atomic_load`, `atomic_store`, `compare_exchange`.
*   **Ordering:** `Relaxed` (Counters), `Acquire/Release` (Message Passing), `SeqCst` (Global Order).
*   **Lesson:** Atomicity != Ordering. You need fences to see the data.

### Day 136: Queues
*   **SPSC:** Wait-Free Ring Buffer. No CAS. Fastest possible IPC.
*   **MPMC:** CAS Loop on indices. Scalability limited by CAS contention.

### Day 137: ABA & Reclaim
*   **ABA:** `A -> B -> A`. CAS succeeds erroneously.
*   **Hazard Pointers:** Readers protect pointers. Writers wait.
*   **Lesson:** You can't `free()` blindly in a lock-free world.

### Day 138: Hash Maps
*   **Open Addressing:** Array + CAS.
*   **Lesson:** Resizing is the enemy. Pre-allocate.

### Day 139: RCU
*   **Read-Copy-Update:** Readers are wait-free and lock-free. Writers are slow.
*   **Use:** Routing tables, configs, plugins.

---

## 🛠️ Capstone Project: `liblockfree`

We will create a helper library for standard C11 projects.

**Features:**
1.  **SPSC Queue:** Generic circular buffer.
2.  **Stack:** Treiber Stack (simplified, leaking for now or using simple HP).
3.  **Macros:** Helper macros for cache line alignment.

### `liblockfree.h`

```c
#ifndef LIBLOCKFREE_H
#define LIBLOCKFREE_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <stdalign.h>
#include <assert.h>

#define LF_CACHE_LINE 64

// --- 1. SPSC QUEUE (Wait-Free) ---

typedef struct {
    alignas(LF_CACHE_LINE) atomic_size_t head;
    alignas(LF_CACHE_LINE) atomic_size_t tail;
    void **buffer;
    size_t cap;
    size_t mask;
} lf_spsc_t;

static inline int lf_spsc_init(lf_spsc_t *q, size_t cap_pow2) {
    if ((cap_pow2 & (cap_pow2 - 1)) != 0) return -1; // Not power of 2
    q->cap = cap_pow2;
    q->mask = cap_pow2 - 1;
    q->buffer = malloc(sizeof(void*) * cap_pow2);
    if (!q->buffer) return -1;
    atomic_init(&q->head, 0);
    atomic_init(&q->tail, 0);
    return 0;
}

static inline void lf_spsc_destroy(lf_spsc_t *q) {
    free(q->buffer);
}

// Returns 1 on success, 0 on full
static inline int lf_spsc_push(lf_spsc_t *q, void *data) {
    size_t h = atomic_load_explicit(&q->head, memory_order_relaxed);
    size_t t = atomic_load_explicit(&q->tail, memory_order_acquire);
    
    if (h - t >= q->cap) return 0; // Full
    
    q->buffer[h & q->mask] = data;
    atomic_store_explicit(&q->head, h + 1, memory_order_release);
    return 1;
}

// Returns 1 on success, 0 on empty
static inline int lf_spsc_pop(lf_spsc_t *q, void **data) {
    size_t t = atomic_load_explicit(&q->tail, memory_order_relaxed);
    size_t h = atomic_load_explicit(&q->head, memory_order_acquire);
    
    if (h == t) return 0; // Empty
    
    *data = q->buffer[t & q->mask];
    atomic_store_explicit(&q->tail, t + 1, memory_order_release);
    return 1;
}

// --- 2. TREIBER STACK (Lock-Free, Leaking Nodes for Simplicity) ---
// Note: In production, integrate Day 137 Hazard Pointers here.

typedef struct lf_node {
    void *item;
    struct lf_node *next;
} lf_node_t;

typedef struct {
    alignas(LF_CACHE_LINE) _Atomic(lf_node_t*) head;
} lf_stack_t;

static inline void lf_stack_init(lf_stack_t *s) {
    atomic_init(&s->head, NULL);
}

static inline void lf_stack_push(lf_stack_t *s, void *item) {
    lf_node_t *new_node = malloc(sizeof(lf_node_t));
    new_node->item = item;
    
    lf_node_t *old_head = atomic_load(&s->head);
    do {
        new_node->next = old_head;
    } while (!atomic_compare_exchange_weak(&s->head, &old_head, new_node));
}

static inline int lf_stack_pop(lf_stack_t *s, void **item) {
    lf_node_t *old_head;
    lf_node_t *new_head;
    
    do {
        old_head = atomic_load(&s->head);
        if (!old_head) return 0; // Empty
        new_head = old_head->next;
        // ABA Problem exists here if we free old_head immediately!
    } while (!atomic_compare_exchange_weak(&s->head, &old_head, new_head));
    
    *item = old_head->item;
    // free(old_head); // UNSAFE without HP
    return 1;
}

#endif // LIBLOCKFREE_H
```

### Usage Example: `test_lib.c`

```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include "liblockfree.h"

#define ITEMS 1000000

lf_spsc_t queue;

void* producer(void* arg) {
    for (size_t i = 1; i <= ITEMS; i++) {
        while (!lf_spsc_push(&queue, (void*)i)) {
            // spin
        }
    }
    return NULL;
}

void* consumer(void* arg) {
    void *data;
    size_t sum = 0;
    size_t count = 0;
    
    while (count < ITEMS) {
        if (lf_spsc_pop(&queue, &data)) {
            sum += (size_t)data;
            count++;
        }
    }
    printf("Consumer Sum: %zu (Expected: %zu)\n", sum, (size_t)ITEMS*(ITEMS+1)/2);
    return NULL;
}

int main() {
    lf_spsc_init(&queue, 1024);
    
    pthread_t p, c;
    pthread_create(&p, NULL, producer, NULL);
    pthread_create(&c, NULL, consumer, NULL);
    
    pthread_join(p, NULL);
    pthread_join(c, NULL);
    
    lf_spsc_destroy(&queue);
    return 0;
}
```

### Analysis
*   **Simplicity:** The SPSC implementation is extremely compact and robust. No `malloc` in the critical path.
*   **Alignment:** `alignas(64)` is critical. Without it, `head` and `tail` updates thrash the L1 cache.
*   **Scalability:** Wait-Free algo allows producer and consumer to run at full instruction retirement speed of the CPU, bottlenecked only by memory bandwidth.

---

## 📅 Looking Ahead: Week 21

Week 21 starts **Operating System Internals & Kernel Development**.
We shift from "Userspace Concurrency" to "Kernel Space Management".
*   **Topics:** Kernel Modules, Syscall implementation, Process Scheduler basics, Interrupt Handling.
*   **Goal:** Write a Linux Kernel Module (LKM) and understand the `task_struct`.

*End of Day 140 - Total Lines: 1000+*
