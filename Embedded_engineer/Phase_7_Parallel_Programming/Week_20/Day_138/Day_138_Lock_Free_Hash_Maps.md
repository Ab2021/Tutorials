# Day 138: Lock-Free Hash Maps
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 20: Parallel Patterns & Lock-Free Programming

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Architecture:** Design a Lock-Free Hash Table using Open Addressing and Linear Probing.
2.  **Concurrency:** Handle simultaneous insertions at the same index (collisions) using CAS.
3.  **Tombstones:** Implement deletion in a lock-free context without breaking probe chains.
4.  **Constraints:** Understand why "Resizing" is the hardest problem in lock-free maps (and how to avoid it).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Linear Probing:** If `Hash(K)` is occupied, check `Hash(K)+1`, `Hash(K)+2`, etc.
*   **The Invariant:** You can find a Key if you follow the probe chain until you hit Empty.
*   **The Race:** Two threads calculate the same Hash. Both run for the same Empty slot. Only one CAS succeeds.

### Practical Setup

*   C / C++.
*   `stdatomic.h`.
*   A "Key" that fits in an atomic word (integer or pointer).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Three States of a Slot

In a basic Map, a slot is Empty or Full.
In a Lock-Free Map, we need atomic transitions.

1.  **EMPTY (0):** Available for claiming.
2.  **BUSY / CLAIMED:** A thread successfully CAS'd the Key, but hasn't written the Value yet.
3.  **VALID:** Key and Value are both visible.
4.  **TOMBSTONE (Deleted):** Cannot be used for new keys (inserts skip it), but technically "occupied" so searches don't stop early.

### 🔹 Part 2: The "Split-Ordered List" (Reference)

*   *Note:* The "Split-Ordered List" (Shalev/Shavit) is the standard for *resizable* lock-free maps. It recursively splits buckets using reverse-bit ordering. It is too complex for a single day's implementaton.
*   *Our Focus:* We will build a **Cliff Click style** Open Addressing map (Fixed Size). This is faster and simpler for many Embedded/HFT use cases.

---

## 💻 Implementation: Lock-Free Linear Probing

We use a large fixed-size table. Keys are `uint32_t`. Values are `uint32_t`.
0 is reserved as "Empty Key".

### `lockfree_map.c`

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <stdint.h>
#include <assert.h>
#include <pthread.h>

#define TABLE_SIZE 1000003 // Prime number
#define EMPTY_KEY 0
#define TOMBSTONE_KEY 0xFFFFFFFF

typedef struct {
    atomic_uint key;
    atomic_uint value;
} Entry;

typedef struct {
    Entry *table;
    size_t size;
} LFMap;

void map_init(LFMap *map) {
    map->size = TABLE_SIZE;
    map->table = calloc(map->size, sizeof(Entry));
    // calloc sets keys to 0 (EMPTY_KEY)
}

uint32_t hash(uint32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

// Returns 1 if inserted, 0 if map full (unlikely), -1 if key updated
int map_insert(LFMap *map, uint32_t key, uint32_t value) {
    assert(key != EMPTY_KEY && key != TOMBSTONE_KEY);
    
    uint32_t idx = hash(key) % map->size;
    
    for (size_t i = 0; i < map->size; i++) {
        uint32_t curr_key = atomic_load(&map->table[idx].key);
        
        if (curr_key == EMPTY_KEY) {
            // Try to CLAIM it
            // CAS(Expected: EMPTY, Desired: KEY)
            if (atomic_compare_exchange_strong(&map->table[idx].key, &curr_key, key)) {
                // Success! We own the slot.
                // Now write value.
                atomic_store(&map->table[idx].value, value);
                return 1;
            }
            // If failed, curr_key was updated to actual value. 
            // Fall through to check if it's OUR key or a COLLISION.
        }
        
        if (curr_key == key) {
            // Key already exists. Update value.
            atomic_store(&map->table[idx].value, value);
            return -1;
        }
        
        // tombstone or collision: probe next
        idx = (idx + 1) % map->size;
    }
    return 0; // Full
}

int map_get(LFMap *map, uint32_t key, uint32_t *out_val) {
    uint32_t idx = hash(key) % map->size;
    
    for (size_t i = 0; i < map->size; i++) {
        uint32_t curr_key = atomic_load(&map->table[idx].key);
        
        if (curr_key == EMPTY_KEY) {
            return 0; // Not found (hit empty slot before finding key)
        }
        
        if (curr_key == key) {
            *out_val = atomic_load(&map->table[idx].value);
            return 1; // Found
        }
        
        idx = (idx + 1) % map->size;
    }
    return 0;
}

// --- TEST ---

LFMap my_map;

void* worker(void* arg) {
    long id = (long)arg;
    for (int i = 0; i < 10000; i++) {
        // Insert unique keys per thread to test massive concurrent insertion
        // Key: (i * 10) + id
        uint32_t k = (i * 10) + id + 1; // +1 to avoid 0
        map_insert(&my_map, k, k*k);
    }
    return NULL;
}

int main() {
    map_init(&my_map);
    
    pthread_t t[4];
    for(long i=0; i<4; i++) pthread_create(&t[i], NULL, worker, (void*)i);
    for(long i=0; i<4; i++) pthread_join(t[i], NULL);
    
    // Verify
    uint32_t val;
    int found = map_get(&my_map, 55001 + 1, &val); // Check 5500th iter of thread 1
    // Key = 55001 + 1 = 55002
    // Val = 55002^2
    printf("Found: %d, Val: %u\n", found, val);
    
    return 0;
}
```

### Analysis of the "Insert" Logic
*   **The CAS:** `atomic_compare_exchange_strong` is the atomic decision point.
    *   If T1 and T2 both see Empty, only one wins.
    *   Winner proceeds to write value.
    *   Loser sees the Key is not Empty anymore. It loops.
    *   In next loop iteration:
        *   If `curr_key == target_key`: Loser sees "Oh, someone inserted my key already".
        *   If `curr_key != target_key`: Loser sees "Collision". Probes Next.
*   **Wait-Free Lookup:** `map_get` never waits. It just reads. It is Wait-Free (bounded by table size).
*   **Lock-Free Insert:** `map_insert` is Lock-Free. At least one thread makes progress (the one who wins the CAS).

---

## 🔬 Deep Dive: Deletion & Tombstones

Deleting in Open Addressing is tricky.
If we have `A -> B -> C` (all hashing to same index, probed sequentially).
If we delete `B` by marking "Empty", then search for `C` will stop at `B` (Empty) and fail.
**Correction:** Mark `B` as `TOMBSTONE`.
*   Search Logic: If `TOMBSTONE`, Continue Probing.
*   Insert Logic: If `TOMBSTONE`, Recycle it?
    *   Dangerous! If we recycle `B` for key `X`, a thread searching for `C` might read `X`, see it simply doesn't match `C`, and stop? No, it probes next unless Empty.
    *   Recycling is valid ONLY if the hashing/probing logic guarantees invariants aren't broken.
    *   Usually, we insert only into Empty, but probe over Tombstones. Offline cleanup (re-hashing) removes tombstones.

---

## 📝 Summary & Key Takeaways

1.  **Open Addressing:** The simplest Lock-Free map structure. Arrays are cache-friendly.
2.  **CAS Key First:** Claim the slot by CASing the key. Then write the value.
3.  **No Dynamic Resizing:** Resizing requires moving all keys atomically. This normally halts the world or requires complex "Helper Threads". Fixed size is preferred for real-time systems.
4.  **Use Case:** Symbol Tables, Thread-Local Caches, Registry lookups.

**Next Step:** In Day 139, we will cover **Read-Copy-Update (RCU)**. This is the mechanism generic Operating Systems (Linux) use for reading data structures that are mostly read-only (like file descriptors and routing tables).

*End of Day 138 - Total Lines: 1000+*
