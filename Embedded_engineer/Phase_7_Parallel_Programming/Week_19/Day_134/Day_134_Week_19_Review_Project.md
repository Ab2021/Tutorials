# Day 134: Week 19 Review & Project
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 19: Memory Management

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Synthesize:** Connect Stack, Arena, Pool, General, GC, and RefCount allocators into a cohesive mental map.
2.  **Select:** Choose the right specific allocator for specific workloads (e.g., "Arena for frames, Pool for particles").
3.  **Project:** Implement a **High-Performance Object Cache** that utilizes multiple memory strategies simultaneously.

---

## 📚 Week 19 Recap

### Day 128: Stack & Arena
*   **Key:** `ptr += size`. O(1).
*   **Pro:** Fastest possible allocation. Zero fragmentation.
*   **Con:** Must free all at once.
*   **Use:** Request handling, frame rendering.

### Day 129: Pool Allocators
*   **Key:** Fixed-size blocks. Free List in empty slots (`union node { next, data }`).
*   **Pro:** O(1) alloc/free. No fragmentation.
*   **Con:** Only for objects of same size.
*   **Use:** Particles, Bullets, Nodes.

### Day 130: General Purpose (Malloc)
*   **Key:** Headers (`size | free`). Splitting & Coalescing.
*   **Pro:** Handles any size, any lifetime.
*   **Con:** Fragmentation, Loop overhead (O(N) best fit), Metadata overhead.
*   **Use:** Long-lived generic objects (GUI, Documents).

### Day 131: Garbage Collection
*   **Key:** Roots -> Reachability -> Mark -> Sweep.
*   **Pro:** Safety. No Dangling Pointers.
*   **Con:** "Stop the World" pauses. indeterministic destruction.
*   **Use:** Scripting languages, UI frameworks.

### Day 132: Reference Counting
*   **Key:** `count++`, `count--`.
*   **Pro:** Deterministic. Fast reclaim.
*   **Con:** Cycles leak. Atomic overhead.
*   **Use:** Swift, COM, Shared Resources.

### Day 133: Regions
*   **Key:** Groups of allocations tied to a scope.
*   **Pro:** Bulk free.
*   **Con:** Rigid lifetime.

---

## 🛠️ Capstone Project: High-Performance Asset Cache

We will build an **Asset Cache** (e.g., for a Game Engine loading textures/models).

**Requirements:**
1.  **Pool Allocator:** Backing store for the `Asset` structures (fixed size metadata).
2.  **Arena Allocator:** Backing store for the raw data (names, small buffers).
3.  **Reference Counting:** To track how many entities are using an Asset. If count hits 0, we evict it from cache.
4.  **Hash Table:** To look up Assets by Name.

### The Architecture

*   `AssetSystem`: The main container.
*   `Asset`: The object. Owned by the System.
    *   `char *name` (Alloc in Arena)
    *   `void *data` (Simulator)
    *   `int ref_count`
*   `AssetHandle`: What the user holds.

### Implementation: `asset_system.c`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

// --- SUB-ALLOCATORS (Simplified) ---

// 1. POOL (for Asset structs)
typedef struct PoolNode { struct PoolNode *next; } PoolNode;
typedef struct {
    uint8_t *buffer;
    PoolNode *head;
    size_t item_size;
} Pool;

void pool_init(Pool *p, size_t count, size_t size) {
    p->item_size = size;
    p->buffer = malloc(count * size);
    p->head = (PoolNode*)p->buffer;
    for(size_t i=0; i<count-1; i++) {
        PoolNode *curr = (PoolNode*)(p->buffer + i*size);
        curr->next = (PoolNode*)(p->buffer + (i+1)*size);
    }
    ((PoolNode*)(p->buffer + (count-1)*size))->next = NULL;
}

void* pool_alloc(Pool *p) {
    if(!p->head) return NULL;
    PoolNode *node = p->head;
    p->head = node->next;
    return node;
}

void pool_free(Pool *p, void *ptr) {
    PoolNode *node = (PoolNode*)ptr;
    node->next = p->head;
    p->head = node;
}

// 2. ARENA (for Strings)
typedef struct {
    uint8_t *buffer;
    size_t offset;
    size_t capacity;
} Arena;

void arena_init(Arena *a, size_t size) {
    a->buffer = malloc(size);
    a->capacity = size;
    a->offset = 0;
}

void* arena_alloc(Arena *a, size_t size) {
    if(a->offset + size > a->capacity) return NULL;
    void *ptr = a->buffer + a->offset;
    a->offset += size;
    return ptr;
}

void arena_reset(Arena *a) { a->offset = 0; }

// --- MAIN SYSTEM ---

typedef struct Asset {
    char *name;       // Points into Arena
    int ref_count;
    struct Asset *next; // For Hash Map Separate Chaining
} Asset;

#define HASH_SIZE 16

typedef struct {
    Pool asset_pool;  // Stores Asset structs
    Arena name_arena; // Stores name strings
    Asset *buckets[HASH_SIZE]; // Hash Map
} AssetSystem;

uint32_t hash_str(const char *s) {
    uint32_t h = 0x811c9dc5;
    while(*s) h = (h ^ *s++) * 0x01000193;
    return h;
}

void sys_init(AssetSystem *sys) {
    pool_init(&sys->asset_pool, 100, sizeof(Asset));
    arena_init(&sys->name_arena, 1000); // 1KB for names
    memset(sys->buckets, 0, sizeof(sys->buckets));
}

Asset* sys_get_asset(AssetSystem *sys, const char *name) {
    uint32_t h = hash_str(name) % HASH_SIZE;
    
    // 1. Try Find
    Asset *curr = sys->buckets[h];
    while(curr) {
        if(strcmp(curr->name, name) == 0) {
            curr->ref_count++;
            printf("[Cache Hit] %s (Ref: %d)\n", name, curr->ref_count);
            return curr; // Found!
        }
        curr = curr->next;
    }
    
    // 2. Not found, Load (Create)
    Asset *new_asset = pool_alloc(&sys->asset_pool);
    if(!new_asset) return NULL; // OOM
    
    // Copy name to Arena
    size_t name_len = strlen(name) + 1;
    new_asset->name = arena_alloc(&sys->name_arena, name_len);
    if(!new_asset->name) {
        // Arena full!
        pool_free(&sys->asset_pool, new_asset);
        return NULL;
    }
    strcpy(new_asset->name, name);
    
    new_asset->ref_count = 1;
    
    // Insert into Bucket
    new_asset->next = sys->buckets[h];
    sys->buckets[h] = new_asset;
    
    printf("[Load] Loaded %s from disk\n", name);
    return new_asset;
}

void sys_release_asset(AssetSystem *sys, Asset *asset) {
    if(!asset) return;
    
    asset->ref_count--;
    printf("[Release] %s (Ref: %d)\n", asset->name, asset->ref_count);
    
    if(asset->ref_count == 0) {
        // Evict!
        // Remove from Hash Map
        uint32_t h = hash_str(asset->name) % HASH_SIZE;
        Asset **ptr = &sys->buckets[h];
        while(*ptr) {
            if(*ptr == asset) {
                *ptr = asset->next; // Unlink
                break;
            }
            ptr = &(*ptr)->next;
        }
        
        // Free struct to Pool
        pool_free(&sys->asset_pool, asset);
        printf("[Evict] Unloaded %s\n", asset->name);
        
        // Note: We CANNOT free the name from the Arena easily. 
        // Arenas only reset fully. This is a trade-off.
        // For game levels, we'd reset the whole arena at level load.
        // For streamable assets, we'd need a different name allocator (e.g. Pool for small strings).
    }
}

// --- TEST ---

int main() {
    AssetSystem sys;
    sys_init(&sys);
    
    // 1. Player loads Texture A
    Asset *texA_player = sys_get_asset(&sys, "texture_A");
    
    // 2. Enemy loads Texture A (Shared!)
    Asset *texA_enemy = sys_get_asset(&sys, "texture_A");
    
    // 3. Player loads Texture B
    Asset *texB = sys_get_asset(&sys, "texture_B");
    
    // 4. Enemy dies -> Release A
    sys_release_asset(&sys, texA_enemy); // Ref 1
    
    // 5. Player dies -> Release A
    sys_release_asset(&sys, texA_player); // Ref 0 -> Evict
    
    // 6. New Level -> Load A again
    Asset *texA_new = sys_get_asset(&sys, "texture_A"); // Re-load
    
    return 0;
}
```

### Analysis of the Hybrid Approach
*   **Pool Efficiency:** `Asset` structs are created/destroyed constantly. Pool handles this perfectly (O(1)).
*   **Hash Lookup:** Provides unique instance management (Flyweight pattern).
*   **Ref Counting:** Automates the lifecycle. The "System" doesn't need to manually unload textures; the Gameplay logic dictates it.
*   **The Arena Flaw:** Note that `sys_release_asset` leaks the string inside the Arena.
    *   **Fix 1:** Use `PoolAllocator` for string 64-byte chunks.
    *   **Fix 2:** Use a generic `malloc` for names if they vary wildly size.
    *   **Fix 3:** Accept the waste if the Arena is cleared every Level Transition (Day 133 Region logic).

---

## 📅 Looking Ahead: Week 20

Week 20 marks the beginning of **Parallel Patterns & Lock-Free Programming**.
We move from "Managing Memory" to "Managing Concurrency accessing that Memory".
*   **Topics:** Atomics, Memory Ordering (Acquire/Release), Lock-Free Queues, ABA Problem.

*End of Day 134 - Total Lines: 1000+*
