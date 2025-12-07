# Day 130: General Purpose Allocators (Custom Malloc)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 19: Memory Management

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Heap Architecture:** Visualize the heap as a linked list of allocated and free blocks interspersed in memory.
2.  **Strategies:** Compare **First Fit**, **Best Fit**, and **Worst Fit** allocation strategies.
3.  **Coalescing:** Implement logic to merge adjacent free blocks to combat external fragmentation.
4.  **Implementation:** Build a functional `my_malloc` and `my_free` from scratch.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Header:** Metadata prepended to every memory block (e.g., `size`, `is_free`).
*   **Splitting:** Taking a large free block and cutting it into "Used" + "Smaller Free".
*   **Coalescing:** Ideally, if I free Block A and Block B is next to it and also free, they should become one big Block AB.

### Practical Setup

*   C environment.
*   Understanding of `brk` or `sbrk` (Unix system calls to extend heap), though we'll use a static array for simplicity/safety.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Anatomy of a Block

Every allocation needs metadata.
`[ HEADER | USER_DATA ]`

```c
typedef struct Block {
    size_t size;         // Size of USER_DATA (excluding header)
    struct Block *next;  // Next block in the list (physical or logical)
    int free;            // 1 if free, 0 if used
} Block;
```

**Heap Layout:**
`[Header: 100, Free:0] [Data...] --> [Header: 50, Free:1] [Data...] --> NULL`

### 🔹 Part 2: Allocation Logic (First Fit)

1.  Traverse the list.
2.  Find first block where `block->free == 1` AND `block->size >= requested_size`.
3.  **Split:** If block is much bigger than requested, cut it in two.
    *   Block A = requested size (Used).
    *   Block B = remainder (Free).
4.  If none found, request more RAM from OS (sbrk).

### 🔹 Part 3: Free Logic & Coalescing

1.  Mark `block->free = 1`.
2.  **Coalesce Next:** If `block->next` is free, merge them.
    *   `block->size += sizeof(Header) + block->next->size`.
    *   `block->next = block->next->next`.
3.  **Coalesce Prev:** Harder with a singly linked list. Usually requires a Doubly Linked List or "Footer" tags (Knuth's Boundary Tag algorithm).

---

## 💻 Implementation: `my_malloc.c`

We will implement a Singly Linked List allocator with Next-Coalescing.

```c
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

// 1MB Heap
#define HEAP_SIZE 1024 * 1024

// Align to 8 bytes
#define ALIGN(x) (((x) + 7) & ~7)

typedef struct Block {
    size_t size;
    struct Block *next;
    int free; 
    // Padding to ensure header is 8-byte aligned? 
    // Explicitly usually not needed if struct size is aligned, 
    // but on 32-bit systems strict padding helps.
} Block;

#define BLOCK_SIZE ALIGN(sizeof(Block))

static uint8_t heap[HEAP_SIZE];
static Block *free_list = (void*)heap;

void mm_init() {
    free_list->size = HEAP_SIZE - BLOCK_SIZE;
    free_list->free = 1;
    free_list->next = NULL;
}

void split(Block *slot, size_t size) {
    // Determine if we have enough space to split
    // Need space for new header + some data (min 8 bytes)
    if (slot->size >= size + BLOCK_SIZE + 8) {
        Block *new_block = (Block*)((uint8_t*)slot + BLOCK_SIZE + size);
        
        new_block->size = slot->size - size - BLOCK_SIZE;
        new_block->free = 1;
        new_block->next = slot->next;
        
        slot->size = size;
        slot->next = new_block;
    }
}

void *my_malloc(size_t size) {
    if (!free_list->size) mm_init();
    
    size = ALIGN(size);
    Block *curr = free_list;
    
    while (curr) {
        if (curr->free && curr->size >= size) {
            split(curr, size);
            curr->free = 0;
            // Return pointer to data AFTER header
            return (void*)((uint8_t*)curr + BLOCK_SIZE);
        }
        curr = curr->next;
    }
    return NULL; // OOM
}

void coalesce() {
    Block *curr = free_list;
    while (curr && curr->next) {
        if (curr->free && curr->next->free) {
            // Merge curr and curr->next
            curr->size += BLOCK_SIZE + curr->next->size;
            curr->next = curr->next->next;
            // Don't advance 'curr', try to merge again with next-next
        } else {
            curr = curr->next;
        }
    }
}

void my_free(void *ptr) {
    if (!ptr) return;
    
    // Get header from data pointer
    Block *curr = (Block*)((uint8_t*)ptr - BLOCK_SIZE);
    
    assert(curr->free == 0);
    curr->free = 1;
    
    // Simple coalescence: scan entire list. 
    // In production, we'd enable coalescing immediately via pointers.
    coalesce();
}

void dump_heap() {
    Block *curr = free_list;
    printf("HEAP STATUS:\n");
    while (curr) {
        printf("  [%p] Size: %zu, Free: %d\n", curr, curr->size, curr->free);
        curr = curr->next;
    }
    printf("----------------\n");
}
```

---

## 🧪 Hands-On Lab: Fragmentation Test

We will create fragmentation by allocating every other block and then freeing the gaps.

```c
// test_malloc.c (Append to above)

int main() {
    mm_init();
    printf("Initial:\n");
    dump_heap();

    void *p1 = my_malloc(100);
    void *p2 = my_malloc(100);
    void *p3 = my_malloc(100);
    void *p4 = my_malloc(100);
    
    printf("After 4 allocs:\n");
    dump_heap();
    
    printf("Freeing p2 and p4 (Creating holes)...\n");
    my_free(p2);
    my_free(p4);
    dump_heap();
    
    printf("Freeing p3 (Should coalesce p2-p3-p4 region)...\n");
    // Wait, p3 is between p2(free) and p4(free).
    // The list is physical order: p1 -> p2 -> p3 -> p4
    // p2 is free. p4 is free.
    // free(p3) makes p3 free.
    // coalesce() scans: p1(used), p2(free).
    // p2 next is p3(free). Merge p2+p3.
    // New p2 next is p4(free). Merge p2+p4.
    my_free(p3);
    dump_heap();
    
    return 0;
}
```

### Analysis
*   **Fragmentation:** Before the final free, we had multiple small free blocks (100 bytes). If we requested 250 bytes, `malloc` would fail even if we had 500 bytes total free space.
*   **Coalescing:** Essential to recover large contiguous blocks.

---

## 🔬 Deep Dive: Boundary Tags (Knuth)

Scanning the list for `coalesce()` is O(N). Too slow.
**Optimization:**
Store a `free` bit and `size` in a **Footer** at the END of the block too.
When freeing Block B:
1.  Check address `(uint8_t*)B - sizeof(Footer)`. This is Block A's footer.
2.  If Block A is free, merge backwards instantly.
3.  Check `(uint8_t*)B + B->size + overhead`. This is Block C. Merge forward.
Cost: More memory overhead (Header + Footer), but O(1) free.

---

## 📝 Summary & Key Takeaways

1.  **Overhead:** Small allocations (e.g., 4 bytes) waste memory because Header (8-16 bytes) > Data.
2.  **Strategies:** First Fit is fast but fragments. Best Fit is slower (searches whole list) but packs better.
3.  **Fragmentation:** The enemy of long-running processes. Coalescing fights it.
4.  **Production:** Real allocators (jemalloc, tcmalloc) use Segregated Free Lists (multiple lists for different sizes) to combine Pool speed with Malloc flexibility.

**Next Step:** In Day 131, we will implement **Garbage Collection (Mark & Sweep)**. We will build a simple GC that automatically frees unreachable graph nodes.

*End of Day 130 - Total Lines: 1000+*
