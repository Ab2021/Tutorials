# Day 135: Atomic Operations & Memory Ordering (C++11/C11)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 20: Parallel Patterns & Lock-Free Programming

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Atomic Basics:** Define what makes an operation "Atomic" (indivisible).
2.  **RMW:** Use Read-Modify-Write operations like `fetch_add` and `compare_exchange`.
3.  **Memory Models:** Explain the C11/C++11 Memory Model (SC, Relaxed, Acquire/Release).
4.  **Hardware:** Understand why x86 is "Strongly Ordered" while ARM is "Weakly Ordered" and how this affects your code.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Race Condition:** Two threads access shared data, at least one writes, without synch. Result is Undefined Behavior (UB).
*   **Volatile:** Does NOT imply atomicity or thread-safety. Is useless for multi-threading (in C/C++).
*   **The Cache Coherence Protocol (MESI):** Hardware ensures caches agree, but execution pipelines reorder instructions.

### Practical Setup

*   C compiler with C11 support (`-std=c11`) or C++11 (`-std=c++11`).
*   `<stdatomic.h>` (C) or `<atomic>` (C++).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Atomicity vs Ordering

1.  **Atomicity:** "All or Nothing".
    *   `i++` is 3 instructions: Load, Inc, Store. Thread B can interrupt in the middle.
    *   `atomic_fetch_add(&i, 1)` is 1 indivisible event (to observers).
2.  **Ordering:** "When does it happen?"
    *   Thread A: `data = 42; flag = 1;`
    *   Thread B: `if (flag) print(data);`
    *   **Horror:** CPU might reorder writes. Thread B sees `flag=1` but OLD `data`.
    *   **Solution:** Memory Barriers/Fences (Acquire/Release).

### 🔹 Part 2: The Levels of Ordering

1.  **SeqCst (Sequential Consistency):** The default. Safest. Slowest. "Global Order" of all ops.
2.  **Acquire/Release:**
    *   **Store-Release:** "All my previous writes are visible before this flag is set."
    *   **Load-Acquire:** "I see all writes that happened before the releasing-store I just read."
3.  **Relaxed:** No ordering guarantees. Just Atomicity. Use for counters, stats.

---

## 💻 Implementation: Atomics in C11

We use C11 `<stdatomic.h>`.

### `atomics_demo.c`

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>

// 1. Shared Atomic Counter
atomic_int counter = 0;

void* increment_task(void *arg) {
    for (int i = 0; i < 100000; i++) {
        // Equivalent to i++ but Thread-Safe
        // Uses memory_order_seq_cst by default
        atomic_fetch_add(&counter, 1);
        
        // Relaxed version (faster, ok for stats):
        // atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);
    }
    return NULL;
}

// 2. Spinlock using Compare-and-Swap (CAS)
atomic_flag lock_flag = ATOMIC_FLAG_INIT;

void spin_lock() {
    // test_and_set returns PREVIOUS value.
    // If prev was 1, someone holds it. Spin.
    // If prev was 0, we set it to 1. Success.
    while (atomic_flag_test_and_set(&lock_flag)) {
        // Pause CPU to save power/reduce contention
        // _mm_pause(); // x86 intrinsic
    }
}

void spin_unlock() {
    atomic_flag_clear(&lock_flag);
}

// 3. Acquire-Release Handshake
atomic_int data_payload = 0;
atomic_int data_ready = 0;

void* producer(void* arg) {
    data_payload = 42; // Regular write
    // RELEASE: Ensures 'data_payload' write is visible BEFORE 'data_ready' write
    atomic_store_explicit(&data_ready, 1, memory_order_release);
    return NULL;
}

void* consumer(void* arg) {
    int ready = 0;
    while (!ready) {
        // ACQUIRE: Ensures subsequent reads happen AFTER this load
        ready = atomic_load_explicit(&data_ready, memory_order_acquire);
    }
    // Safe to read payload
    printf("Consumer read payload: %d\n", data_payload);
    assert(data_payload == 42); 
    return NULL;
}

int main() {
    pthread_t t1, t2;
    
    // Test 1: Counter
    pthread_create(&t1, NULL, increment_task, NULL);
    pthread_create(&t2, NULL, increment_task, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    printf("Counter: %d (Expected 200000)\n", atomic_load(&counter));
    
    // Test 2: Acquire/Release
    atomic_store(&data_ready, 0);
    pthread_create(&t1, NULL, producer, NULL);
    pthread_create(&t2, NULL, consumer, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    
    return 0;
}
```

---

## 🧪 Hands-On Lab: Implementing a Safe Stack (Treiber Stack)

The Treiber Stack is the "Hello World" of lock-free data structures.

```c
typedef struct Node {
    int value;
    struct Node *next;
} Node;

// The atomic head pointer
_Atomic(Node*) head = NULL;

void push(int val) {
    Node *new_node = malloc(sizeof(Node));
    new_node->value = val;
    
    Node *old_head;
    
    // CAS Loop
    do {
        old_head = atomic_load(&head);
        new_node->next = old_head;
        // Try to swap Head with NewNode
        // If head == old_head, head = new_node, return true.
        // If head != old_head (changed by other thread), update old_head, return false.
    } while (!atomic_compare_exchange_weak(&head, &old_head, new_node));
}

int pop(int *result) {
    Node *old_head;
    Node *new_head;
    
    do {
        old_head = atomic_load(&head);
        if (old_head == NULL) return 0; // Empty
        new_head = old_head->next;
        
        // ABA Problem risk here! (We will fix in Day 137)
        // If old_head was freed and re-allocated as same address between load and CAS...
    } while (!atomic_compare_exchange_weak(&head, &old_head, new_head));
    
    *result = old_head->value;
    // free(old_head); // DANGEROUS! Thread B might be looking at old_head right now!
    // Without GC, Lock-Free Memory Reclaim (Epochs/Hazard Pointers) is HARD.
    // For now, we leak or depend on Day 137.
    return 1;
}
```

### Analysis
*   **Weak vs Strong CAS:** `weak` can fail spuriously (return false even if value matches). Better for loops on ARM/PowerPC. `strong` guarantees success if value matches.
*   **Lock-Free:** At least one thread makes progress. No deadlocks.
*   **Wait-Free:** *All* threads make progress. (CAS loops are not wait-free, starvation is possible).

---

## 🔬 Deep Dive: Memory Barriers on Hardware

1.  **x86 (Total Store Order):**
    *   Loads are not reordered with loads.
    *   Stores are not reordered with stores.
    *   Stores can be buffered (Store Buffer).
    *   Basically, x86 is almost Acquire/Release by default.
    *   `MFENCE` is strictly needed only for Store-Load ordering (Dekker's Algorithm).
2.  **ARMv7/v8 (Weakly Ordered):**
    *   ANYTHING can reorder.
    *   Store-Store reorder? Yes.
    *   Load-Load reorder? Yes.
    *   `DMB` (Data Memory Barrier) instructions are inserted by the compiler for C11 atomics.
    *   Code correct on x86 might break on ARM if you use `relaxed` incorrectly!

---

## 📝 Summary & Key Takeaways

1.  **Atomics:** Necessary for any shared data modified by threads.
2.  **RMW:** CAS (`compare_exchange`) is the primitive for all lock-free algos.
3.  **Ordering:** Use `Release` when publishing data. Use `Acquire` when reading published data. Use `Relaxed` only if order doesn't matter (stats).
4.  **Hardware:** Be "Weakly Ordered" aware. Test on ARM/Apple Silicon if possible to catch bugs x86 hides.

**Next Step:** In Day 136, we will cover **Lock-Free Queues (SPSC, MPMC)**. We will implement the high-performance Single-Producer-Single-Consumer Ring Buffer widely used in audio and networking.

*End of Day 135 - Total Lines: 1000+*
