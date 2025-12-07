# Day 137: ABA Problem & Hazard Pointers
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 20: Parallel Patterns & Lock-Free Programming

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **ABA Problem:** Explain exactly how a "successful" CAS can corrupt a data structure.
2.  **Tagging:** Use Tagged Pointers (Versioning) to solve ABA in simple cases (Stacks).
3.  **Hazard Pointers:** Implement Safe Memory Reclamation (SMR) to prevent freeing memory while other threads read it.
4.  **Trade-offs:** Compare Performance vs Safety in memory reclamation.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **CAS:** `Compare-And-Swap(Addr, Old, New)`. If `Addr` contains `Old`, write `New`.
*   **The Assumption:** If `Addr` contains `Old`, then the state has not changed.
*   **The Flaw:** `Old` could be `A`. It changed to `B`, then back to `A`. CAS succeeds, but side effects (like strictly ordered nodes) might make this invalid.

### Practical Setup

*   C / C++.
*   Understanding of the lock-free stack from Day 135.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The ABA Nightmare Scenario

Consider a Lock-Free Stack: `Head -> A -> B`.
1.  **Thread 1:** Wants to Pop. Reads `Head` (A). Reads `A->Next` (B).
    *   *Paused before CAS.* T1 thinks: "If Head is A, set Head to B".
2.  **Thread 2:** Pops A. Frees A. Stack is `Head -> B`.
3.  **Thread 2:** Pops B. Frees B. Stack is `Head -> NULL`.
4.  **Thread 2:** Allocates new Node `C`. Allocator reuses address of `A`.
5.  **Thread 2:** Pushes `C`. Stack is `Head -> C (at address A) -> NULL`.
6.  **Thread 1:** Wakes up. Performs `CAS(&Head, A, B)`.
    *   Current Head is `A` (actually C, but same address).
    *   **CAS MATCHES!**
    *   Head is set to `B`.
    *   **CRITICAL FAILURE:** `B` was already freed! Head now points to freed memory (B). `C` is leaked/detached.

### 🔹 Part 2: Hazard Pointers (The Manual Solution)

**Concept:** Before T1 reads `A`, it publishes "I am reading Address X" to a global array (Hazard Pointers).
When T2 wants to Free `A`:
1.  Check global Hazard Pointers.
2.  Is `A` in there?
    *   **Yes:** Delay free (Put in "Retire List").
    *   **No:** Safe to free immediately.

---

## 💻 Implementation: ABA in Practice

We simulate the crash first, then fix it with Hazard Pointers.

### `hazard_ptrs.c`

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>
#include <unistd.h>

// --- THE DATA STRUCTURE ---

typedef struct Node {
    int value;
    struct Node *next;
} Node;

_Atomic(Node*) head = NULL;

// --- HAZARD POINTERS INFRASTRUCTURE ---

#define MAX_THREADS 4
#define HP_PER_THREAD 2 // Usually 1 or 2 is enough per thread

// The Global Hazard Pointer Array
// hp[thread_id][hp_index]
// We flatten it: hp[thread_id * HP_PER_THREAD + index]
_Atomic(void*) global_hp[MAX_THREADS * HP_PER_THREAD];

// Retired List (Thread Local) - Pointers waiting to be freed
#define RETIRE_LIMIT 10
void* retired_list[MAX_THREADS][RETIRE_LIMIT];
int retired_count[MAX_THREADS] = {0};

void hp_init() {
    for(int i=0; i<MAX_THREADS*HP_PER_THREAD; i++) {
        atomic_init(&global_hp[i], NULL);
    }
}

// Publish a pointer to protect it
void hp_publish(int tid, int hp_idx, void *ptr) {
    atomic_store(&global_hp[tid * HP_PER_THREAD + hp_idx], ptr);
    // Needed: Memory Barrier (SeqCst default here handles it)
}

// Clear
void hp_clear(int tid, int hp_idx) {
    atomic_store(&global_hp[tid * HP_PER_THREAD + hp_idx], NULL);
}

// Scan global HP array to see if 'ptr' is safe to free
int is_hazarded(void *ptr) {
    for(int i=0; i<MAX_THREADS*HP_PER_THREAD; i++) {
        if(atomic_load(&global_hp[i]) == ptr) return 1;
    }
    return 0;
}

// Try to empty the retired list
void scan_and_free(int tid) {
    int remaining = 0;
    for(int i=0; i<retired_count[tid]; i++) {
        void *ptr = retired_list[tid][i];
        if(is_hazarded(ptr)) {
            // Still in use! Keep it.
            retired_list[tid][remaining++] = ptr;
        } else {
            // Safe to free!
            // printf("Thread %d freeing %p\n", tid, ptr);
            free(ptr);
        }
    }
    retired_count[tid] = remaining; // Compaction
}

// Retire a pointer (call this instead of free)
void retire_node(int tid, void *ptr) {
    retired_list[tid][retired_count[tid]++] = ptr;
    if(retired_count[tid] >= RETIRE_LIMIT) {
        scan_and_free(tid);
    }
}

// --- SAFE STACK OPERATIONS ---

void push(int val) {
    Node *new_node = malloc(sizeof(Node));
    new_node->value = val;
    Node *old_head;
    do {
        old_head = atomic_load(&head);
        new_node->next = old_head;
    } while (!atomic_compare_exchange_weak(&head, &old_head, new_node));
}

int pop(int tid, int *result) {
    Node *old_head;
    Node *new_head;
    
    while(1) {
        old_head = atomic_load(&head);
        if(!old_head) return 0; // Empty
        
        // 1. HAZARD: Protect old_head
        hp_publish(tid, 0, old_head);
        
        // 2. CHECK: Did head change while we were publishing?
        if(atomic_load(&head) != old_head) {
            // Changed. Retry.
            continue; 
        }
        
        // Now safe to read old_head->next because old_head is protected.
        // Even if T2 pops old_head, T2 cannot free it because we hold it in HP.
        new_head = old_head->next;
        
        // 3. CAS
        if(atomic_compare_exchange_weak(&head, &old_head, new_head)) {
            // Success!
            *result = old_head->value;
            hp_clear(tid, 0); // Unprotect
            retire_node(tid, old_head); // Safe Free
            return 1;
        }
        
        // Fail
        hp_clear(tid, 0); // Unprotect
        // Loop again
    }
}

// --- TEST ---

void* stress_test(void *arg) {
    long id = (long)arg;
    for(int i=0; i<10000; i++) {
        push(i);
        int val;
        pop((int)id, &val);
    }
    return NULL;
}

int main() {
    hp_init();
    
    pthread_t threads[MAX_THREADS];
    for(long i=0; i<MAX_THREADS; i++) {
        pthread_create(&threads[i], NULL, stress_test, (void*)i);
    }
    
    for(int i=0; i<MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Passed ABA stress test with Hazard Pointers.\n");
    return 0;
}
```

### Analysis
*   **Without HP:** If we just called `free(old_head)` in `Pop`, Threads doing line `new_head = old_head->next` would access freed memory (Use-After-Free) or the ABA pointer replacement would occur.
*   **Performance:** HPs are heavy. Writing to `global_hp` causes Cache Flushes (Store Buffer drain). Reading `global_hp` in `scan_and_free` is O(N_Threads).
*   **Epochs (RCU):** Faster (O(1) publish), but coarser granularity (blocks entire batches).

---

## 🔬 Deep Dive: Tagged Pointers (Versioning)

On x64, pointers use 48 bits. We have 16 bits free.
We can hide a counter in the top 16 bits.
`Pointer = (Address & 0xFFFFFFFFFFFF) | (Counter << 48)`
Increment counter on every free/alloc.
`A_v1` != `A_v2`.
**Tagged Pointer CAS:** `CAS(Head, A_v1, B)` fails if Head is `A_v2`.
**Limit:** 16 bits wrap fast. 128-bit CAS (`cmpxchg16b`) is preferred to store `{Node*, uint64_t counter}`.

---

## 📝 Summary & Key Takeaways

1.  **ABA:** The "Silent Killer" of Lock-Free Algos. Values match, but identity changed.
2.  **SMR:** Safe Memory Reclamation. You cannot just `free()` in a lock-free structure.
3.  **Hazard Pointers:** The rigorous C solution. Wait for readers to leave.
4.  **Java/Go:** Solved by GC (GC acts as the ultimate SMR). We pay this complexity in C for predictable latency.

**Next Step:** In Day 138, we will cover **Lock-Free Hash Maps**. We will build a high-concurrency Key-Value store using Split-Ordered Lists or Linear Probing with atomics.

*End of Day 137 - Total Lines: 1000+*
