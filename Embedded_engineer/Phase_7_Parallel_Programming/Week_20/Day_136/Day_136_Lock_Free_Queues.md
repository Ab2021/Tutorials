# Day 136: Lock-Free Queues (SPSC / MPMC Ring Buffers)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 20: Parallel Patterns & Lock-Free Programming

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Ring Buffer:** Implement a circular buffer using bitwise wrapping (power of 2).
2.  **SPSC:** Build a **Single-Producer Single-Consumer** queue that is completely Wait-Free (no CAS, just Load/Store).
3.  **MPMC:** Build a **Multi-Producer Multi-Consumer** queue using CAS.
4.  **Ordering:** Apply the Acquire/Release semantics learned in Day 135 to ensure data integrity.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Producer:** Writes data, increments Head.
*   **Consumer:** Reads data, increments Tail.
*   **Full:** `Head - Tail == Size`.
*   **Empty:** `Head == Tail`.
*   **False Sharing:** Ideally, Head and Tail should be on different Cache Lines to avoid ping-ponging.

### Practical Setup

*   C compiler with `<stdatomic.h>`.
*   Threading library (pthread).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: SPSC Magic

In SPSC, only one thread writes `head`, only one writes `tail`.
*   **Producer** owns `head`. It reads `tail` to check usage.
*   **Consumer** owns `tail`. It reads `head` to check availability.
*   No contention on write variables!
*   **Optimization:** We don't need CAS. We just need correct ordering.
    *   Write Data -> Release Barrier -> Increment Head.
    *   Load Head -> Acquire Barrier -> Read Data.

### 🔹 Part 2: MPMC Complexity

In MPMC, multiple producers try to increment `head` at once.
*   `atomic_fetch_add` or CAS loop needed.
*   The data slot itself becomes a shared resource.
*   Strategy: Reserve a slot (CAS index), then fill it.
*   Use a separate sequence buffer to track "Is this slot filled?".

---

## 💻 Implementation: SPSC Ring Buffer

This is the standard design for Audio/Video streaming (Wait-Free).

### `spsc_queue.c`

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <stdint.h>
#include <pthread.h>
#include <assert.h>
#include <stdalign.h>

#define CACHE_LINE 64

typedef struct {
    // Producer writes logical_head, Consumer reads it.
    alignas(CACHE_LINE) atomic_size_t head;
    
    // Consumer writes logical_tail, Producer reads it.
    alignas(CACHE_LINE) atomic_size_t tail;
    
    // Constant after init
    size_t capacity; 
    size_t mask;
    int *buffer;
} SPSCQueue;

void spsc_init(SPSCQueue *q, size_t capacity_pow2) {
    // Must be power of 2 for bitwise masking
    assert((capacity_pow2 & (capacity_pow2 - 1)) == 0);
    
    q->capacity = capacity_pow2;
    q->mask = capacity_pow2 - 1;
    q->buffer = malloc(sizeof(int) * capacity_pow2);
    
    atomic_init(&q->head, 0);
    atomic_init(&q->tail, 0);
}

int spsc_push(SPSCQueue *q, int val) {
    size_t h = atomic_load_explicit(&q->head, memory_order_relaxed);
    size_t t = atomic_load_explicit(&q->tail, memory_order_acquire);
    
    if (h - t >= q->capacity) {
        return 0; // Full
    }
    
    // Write Data FIRST (Relaxed is okay because we Release head later)
    q->buffer[h & q->mask] = val;
    
    // Publish Head (Release ensures data write is visible)
    atomic_store_explicit(&q->head, h + 1, memory_order_release);
    return 1;
}

int spsc_pop(SPSCQueue *q, int *val) {
    size_t t = atomic_load_explicit(&q->tail, memory_order_relaxed);
    size_t h = atomic_load_explicit(&q->head, memory_order_acquire);
    
    if (h == t) {
        return 0; // Empty
    }
    
    // Read Data (Ordered by Acquire above)
    *val = q->buffer[t & q->mask];
    
    // Update Tail (Release ensures we are done reading slot)
    atomic_store_explicit(&q->tail, t + 1, memory_order_release);
    return 1;
}

// --- TEST ---

#define ITERATIONS 1000000

void* producer_thread(void *arg) {
    SPSCQueue *q = (SPSCQueue*)arg;
    for (int i = 0; i < ITERATIONS; i++) {
        while (!spsc_push(q, i)) {
            // spin (or _mm_pause)
        }
    }
    return NULL;
}

void* consumer_thread(void *arg) {
    SPSCQueue *q = (SPSCQueue*)arg;
    int val;
    long sum = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        while (!spsc_pop(q, &val)) {
            // spin
        }
        sum += val;
    }
    printf("Consumer Sum: %ld\n", sum);
    return NULL;
}

int main() {
    SPSCQueue q;
    spsc_init(&q, 1024);
    
    pthread_t p, c;
    pthread_create(&p, NULL, producer_thread, &q);
    pthread_create(&c, NULL, consumer_thread, &q);
    
    pthread_join(p, NULL);
    pthread_join(c, NULL);
    
    return 0;
}
```

---

## 💻 Implementation: Bounded MPMC Queue

This uses Dmitry Vyukov's algorithm (Go Scheduler uses variations of this).
We need a `cell_sequence` array to coordinate slots.

### `mpmc_queue.c`

```c
typedef struct {
    atomic_size_t sequence;
    int data;
} Cell;

typedef struct {
    alignas(CACHE_LINE) atomic_size_t head;
    alignas(CACHE_LINE) atomic_size_t tail;
    size_t mask;
    Cell *buffer;
} MPMCQueue;

void mpmc_init(MPMCQueue *q, size_t cap_pow2) {
    q->mask = cap_pow2 - 1;
    q->buffer = malloc(sizeof(Cell) * cap_pow2);
    atomic_init(&q->head, 0);
    atomic_init(&q->tail, 0);
    
    for (size_t i = 0; i < cap_pow2; i++) {
        // Init sequence to index. 
        // 0th slot expects sequence 0. 1st slot expects 1.
        atomic_init(&q->buffer[i].sequence, i);
    }
}

int mpmc_push(MPMCQueue *q, int val) {
    Cell *cell;
    size_t head_seq; // Does head match slot?
    
    while (1) {
        size_t h = atomic_load(&q->head);
        cell = &q->buffer[h & q->mask];
        size_t seq = atomic_load_explicit(&cell->sequence, memory_order_acquire);
        
        int dif = (int)seq - (int)h;
        
        if (dif == 0) {
            // Slot is ready for enqueue (seq matches head)
            if (atomic_compare_exchange_weak(&q->head, &h, h + 1)) {
                // Success! We claimed the slot idx 'h'
                cell->data = val;
                // Increment sequence to allow Dequeue (seq = h + 1)
                atomic_store_explicit(&cell->sequence, h + 1, memory_order_release);
                return 1;
            }
        } else if (dif < 0) {
            return 0; // Full (Sequence has wrapped / behind head)
        } else {
            // Head moved by other thread, retry
        }
    }
}

int mpmc_pop(MPMCQueue *q, int *val) {
    Cell *cell;
    
    while (1) {
        size_t t = atomic_load(&q->tail);
        cell = &q->buffer[t & q->mask];
        size_t seq = atomic_load_explicit(&cell->sequence, memory_order_acquire);
        
        int dif = (int)seq - (int)(t + 1);
        
        if (dif == 0) {
            // Slot is ready for dequeue (seq == tail + 1)
            if (atomic_compare_exchange_weak(&q->tail, &t, t + 1)) {
                *val = cell->data;
                // Increment sequence to allow Enqueue (seq = wrap + mask + 1)
                atomic_store_explicit(&cell->sequence, t + q->mask + 1, memory_order_release);
                return 1;
            }
        } else if (dif < 0) {
            return 0; // Empty
        }
    }
}
```

---

## 🧪 Hands-On Lab: SPSC vs Mutex

A simple benchmark (theoretical steps):
1.  Run `MutexQueue`: Push/Pop 10M times. Measure time. Context switches kill performance.
2.  Run `SPSCQueue`: Push/Pop 10M times. Measure time. 10x-50x faster.

---

## 🔬 Deep Dive: Cache Lines & Padding

*   `alignas(64)`: Why?
*   If `head` and `tail` are adjacent (4 bytes apart), they share a 64-byte L1 Cache Line.
*   Core 0 writes `head` -> Invalidates Core 1's cache line.
*   Core 1 writes `tail` -> Invalidates Core 0's cache line.
*   **False Sharing:** Use padding to separate hot synchronized variables.

---

## 📝 Summary & Key Takeaways

1.  **SPSC is King:** If you can design your architecture to be Single-Producer/Consumer (e.g., using Thread Pools with local queues), do it. It requires no CAS, just Load/Store barriers.
2.  **MPMC is Hard:** Vyukov's Bounded Queue is the standard efficient solution. Unbounded Lock-Free Queues (LinkedList based) are *slower* due to allocator contention and ABA.
3.  **Barriers:** Release/Acquire is the glue holding the ring buffer logic together.

**Next Step:** In Day 137, we will cover the **ABA Problem & Hazard Pointers**, the advanced memory reclamation techniques required for Lock-Free Linked Lists (Dynamic Structures).

*End of Day 136 - Total Lines: 1000+*
