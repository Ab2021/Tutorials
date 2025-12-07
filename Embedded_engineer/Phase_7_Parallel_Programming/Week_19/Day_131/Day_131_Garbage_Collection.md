# Day 131: Garbage Collection (Mark & Sweep)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 19: Memory Management

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **GC Theory:** Explain the **Mark & Sweep** algorithm and "Stop the World" pauses.
2.  **Roots:** Understand what "Roots" are (Stack pointers, Global variables) and how they anchor the live object graph.
3.  **Implementation:** Build a simple Garbage Collector in C that manages a graph of objects.
4.  **Trade-offs:** Contrast Manual Memory Management (speed, risk) vs GC (safety, pauses).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Mutator:** Your program code (it changes/mutates the graph of objects).
*   **Collector:** The GC code (it cleans up).
*   **Reachability:** An object is "Live" if it can be reached from a Root. Ideally, everything else is garbage.

### Practical Setup

*   C compiler.
*   Basic linked list / graph structure knowledge.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Algorithm

**Phase 1: Mark**
1.  Start from **Roots** (variables currently on stack or globals).
2.  DFS/BFS traverse the graph of objects.
3.  Set `obj->marked = 1` for every visited node.
4.  Stop when no new nodes is found.

**Phase 2: Sweep**
1.  Iterate linearly through the **entire heap** (all allocated blocks).
2.  If `obj->marked == 1`:
    *   It is unreachable? No, it's live!
    *   Reset `obj->marked = 0` (prepare for next GC).
3.  If `obj->marked == 0`:
    *   It is unreachable! Garbage!
    *   Free the memory.

### 🔹 Part 2: What is a Root?

In C/C++, finding roots is hard because the compiler optimizes variables into registers or spills them to stack unpredictably.
"Conservative GC" (like Boehm GC) scans the stack memory and assumes *anything that looks like a pointer* IS a pointer.
For our lab, we will **explicitly register roots** to understand the logic cleanly.

---

## 💻 Implementation: Simple Mark & Sweep GC

We define a generic `Object` that can hold pairs of integers or references to other objects.

### `gc.c`

```c
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define MAX_OBJECTS 256
#define STACK_MAX 256

typedef enum {
    OBJ_INT,
    OBJ_PAIR
} ObjectType;

typedef struct Object {
    ObjectType type;
    unsigned char marked;
    struct Object *next; // Linked list of ALL objects (simulating heap)
    
    union {
        int value; // OBJ_INT
        struct {   // OBJ_PAIR
            struct Object *head;
            struct Object *tail;
        };
    };
} Object;

// The Heap (Linked List of all allocs)
Object *heap_head = NULL;
int num_objects = 0;

// The Virtual Machine Stack (Roots)
Object* vm_stack[STACK_MAX];
int stack_size = 0;

// Push Root
void push(Object *obj) {
    assert(stack_size < STACK_MAX);
    vm_stack[stack_size++] = obj;
}

// Pop Root
Object* pop() {
    assert(stack_size > 0);
    return vm_stack[--stack_size];
}

// Allocation
Object* new_object(ObjectType type) {
    if (num_objects >= MAX_OBJECTS) {
        printf("Out of memory triggers GC!\n");
        // gc_collect(); // Ideally call GC here
    }
    
    Object *obj = malloc(sizeof(Object));
    obj->type = type;
    obj->marked = 0;
    
    // Add to global heap list
    obj->next = heap_head;
    heap_head = obj;
    num_objects++;
    
    return obj;
}

// ======= GC LOGIC =======

void mark(Object *obj) {
    // If NULL or already marked, return
    if (obj == NULL || obj->marked) return;
    
    obj->marked = 1;
    
    if (obj->type == OBJ_PAIR) {
        mark(obj->head);
        mark(obj->tail);
    }
}

void mark_all() {
    for (int i = 0; i < stack_size; i++) {
        mark(vm_stack[i]);
    }
}

void sweep() {
    Object **object = &heap_head;
    while (*object) {
        if (!(*object)->marked) {
            // Unreached! Garbage!
            Object *unreached = *object;
            *object = unreached->next; // Unlink
            free(unreached);
            num_objects--;
        } else {
            // Reached! Unmark for next time
            (*object)->marked = 0;
            object = &(*object)->next;
        }
    }
}

void gc_collect() {
    int prev_count = num_objects;
    mark_all();
    sweep();
    printf("GC: Collected %d objects, %d remaining.\n", 
           prev_count - num_objects, num_objects);
}
```

---

## 🧪 Hands-On Lab: Simulating a Graph

We will build a graph, drop references, and see GC clean it up.

```c
// test_gc.c (Append to above)

int main() {
    printf("--- GC Test ---\n");
    
    // 1. Create some distinct objects (Roots)
    printf("Pushing 2 ints to stack.\n");
    Object *o1 = new_object(OBJ_INT); o1->value = 10;
    Object *o2 = new_object(OBJ_INT); o2->value = 20;
    push(o1);
    push(o2);
    
    // 2. Create a tracked usage (Roots -> Pair -> Ints)
    printf("Creating pair.\n");
    Object *pair = new_object(OBJ_PAIR);
    pair->head = o1;
    pair->tail = o2;
    push(pair);
    
    // Stack: [Int, Int, Pair]
    // Heap: 3 objects. All reachable.
    
    gc_collect(); // Should collect 0
    
    // 3. Drop references
    printf("Popping Pair and one Int.\n");
    pop(); // Drop Pair (but Pair is still pointed to by... oh wait, nothing points to Pair)
           // Actually, vm_stack held the ref.
    pop(); // Drop o2.
    
    // Stack: [o1]
    // Heap: o1 is reachable directly.
    // o2 is ONLY reachable via Pair? No, Pair points to o2.
    // Is Pair reachable? No.
    // So Pair is garbage. o2 is garbage (unless o1 points to it, which it doesn't).
    
    // Wait, o2 is garbage. But mark() on Pair recurses to o2.
    // Since Pair is unreachable, mark(Pair) is never called.
    // So mark(o2) is never called.
    // So both collected.
    
    gc_collect(); // Should collect Pair and o2 (2 objects)
    
    return 0;
}
```

### Analysis of Logic
*   **Reference Counting** would free `Pair` immediately when popped (count=0). Then `Pair` destructor would decrement `o2`.
*   **Mark & Sweep** waits until memory pressure (or explicit call) to do a full scan.
*   **Cycles:** If Pair A -> Pair B -> Pair A, and Stack drops A:
    *   Ref Count: Fails (cycles keep count > 0). Leak!
    *   Mark & Sweep: Succeeds. Roots don't reach A. Garbage. Mark & Sweep handles cycles naturally.

---

## 🔬 Deep Dive: Generational GC

Scanning the *entire* heap (Sweep) is slow for huge heaps.
**Generational Hypothesis:** Most objects die young (temporary vars).
**Strategy:**
*   **Gen 0 (Nursery):** Where new objs are alloc'd. Small. GC'd frequently.
*   **Gen 1 / Old:** If obj survives ~3 GCs, move it here. GC this rarely.
*   Write Barriers are needed to track if an Old obj points to a New obj.

---

## 📝 Summary & Key Takeaways

1.  **Roots are Key:** If you lose the root, you lose the object (GC frees it). If you keep a root unintentionally (stale pointer in global list), you get a Memory Leak (GC *won't* free it).
2.  **Cycles:** Mark & Sweep solves the Circular Reference problem that plagues Reference Counting (e.g., `std::shared_ptr`).
3.  **Pauses:** Stop-the-world is unacceptable for real-time systems (games, audio), leading to **Incremental GC**.
4.  **Java/C#/Go:** All use sophisticated variations of this (Generational, Concurrent, Compacting).

**Next Step:** In Day 132, we will cover **Reference Counting & Smart Pointers**, implementing our own `shared_ptr` and `weak_ptr` to understand C++ RAII.

*End of Day 131 - Total Lines: 1000+*
