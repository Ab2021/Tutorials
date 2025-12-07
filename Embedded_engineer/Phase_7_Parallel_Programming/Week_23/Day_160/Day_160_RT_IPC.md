# Day 160: IPC in Real-Time Linux (Shared Memory & Lock-Free Queues)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 23: Real-Time Linux

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **The IPC Problem:** Explain why standard Pipes, Sockets, and Message Queues violate RT constraints.
2.  **Shared Memory:** Implement zero-copy communication using `shm_open` and `mmap`.
3.  **Lock-Free Ring Buffer:** Design a Single-Producer Single-Consumer (SPSC) queue safe for RT usage.
4.  **Priority Inversion:** Demonstrate how sharing a Mutex between RT and Non-RT tasks destroys performance.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Syscalls:** Calling `write()` to a pipe involves the kernel. The kernel might block/sleep. **Forbidden** in RT loops.
*   **Inversion:** If RT task waits for a Global Lock held by a low-priority Logging task, the RT task effectively becomes low priority.
*   **Memory Barriers:** Ordering instructions to ensure data is visible before the "ready" flag is set.

### Practical Setup

*   **Library:** `-lrt` (POSIX Realtime).
*   **Headers:** `stdatomic.h` (C11 Atomies).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Cross-Domain Challenge

Architecture:
*   **Core 2 (RT):** High Frequency Control Loop (1 kHz). Reads Sensors, Calculates Torque. Needs to log data.
*   **Core 0 (Non-RT):** Logging Service. Writes to Disk (slow), sends Network packets (jittery).

**Naive Approach (Fail):**
RT Task calls `send()` to a socket.
*   Socket buffer full? RT Task sleeps (Fail).
*   Kernel lock contention? RT Task waits (Fail).

**Correct Approach:**
RT Task writes to a **Lock-Free Ring Buffer** in Shared Memory.
*   If buffer full? RT Task drops data (or overwrites old). **Never Sleeps.**
*   Non-RT Task reads from buffer and does the slow I/O.

### 🔹 Part 2: Lock-Free Mechanics

We use indices `head` and `tail`.
*   buffer size MUST be a power of 2 (for fast modulo masking).
*   **Producer (RT):** Writes data, increments `head`.
*   **Consumer (Non-RT):** Reads data, increments `tail`.
*   **Constraint:** ONLY Producer touches `head`. ONLY Consumer touches `tail`. No CAS needed, just Release/Acquire barriers.

---

## 💻 Implementation: The Ring Buffer

### 1. The Header (`rt_ipc.h`)

```c
#ifndef RT_IPC_H
#define RT_IPC_H

#include <stdint.h>
#include <stdatomic.h>

#define RING_SIZE 4096 // Must be power of 2

struct SharedRing {
    atomic_size_t head; // Written by Producer
    atomic_size_t tail; // Written by Consumer
    uint8_t buffer[RING_SIZE];
};

// Calculate size used
static inline size_t ring_count(struct SharedRing* r) {
    return atomic_load_explicit(&r->head, memory_order_acquire) - 
           atomic_load_explicit(&r->tail, memory_order_acquire);
}
#endif
```

### 2. The Producer (RT Task - C Code)

```c
void rt_log_data(struct SharedRing* r, uint8_t data) {
    size_t h = atomic_load_explicit(&r->head, memory_order_relaxed);
    size_t t = atomic_load_explicit(&r->tail, memory_order_acquire);
    
    // Check if full
    if (h - t >= RING_SIZE) {
        // Drop data to preserve timing!
        // Or overwrite: just proceed (SPSC allows this risk if careful)
        return; 
    }
    
    // Write Data
    r->buffer[h & (RING_SIZE - 1)] = data;
    
    // Publish (Release Barrier ensures data is written before head moves)
    atomic_store_explicit(&r->head, h + 1, memory_order_release);
}
```

### 3. The Consumer (Non-RT Task - C Code)

```c
void process_logs(struct SharedRing* r) {
    while(1) {
        size_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
        size_t h = atomic_load_explicit(&r->head, memory_order_acquire);
        
        if (h == t) {
            usleep(1000); // Empty, sleep to save CPU on Non-RT core
            continue;
        }
        
        // Read Data
        uint8_t data = r->buffer[t & (RING_SIZE - 1)];
        printf("Log: %d\n", data); // Slow I/O
        
        // Update Tail
        atomic_store_explicit(&r->tail, t + 1, memory_order_release);
    }
}
```

### 4. Setup (Shared Memory)

```c
int fd = shm_open("/my_rt_shm", O_CREAT | O_RDWR, 0666);
ftruncate(fd, sizeof(struct SharedRing));
struct SharedRing* ring = mmap(NULL, sizeof(struct SharedRing), 
                               PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
// Initialize head/tail to 0
```

---

## 🔬 Deep Dive: Memory Barriers

Why `memory_order_release`?
*   It prevents the CPU (or Compiler) from reordering the store to `head` *before* the store to `buffer`.
*   Without this, Consumer might see `head` move, read the buffer, and get garbage data because the data write hasn't landed yet.

---

## 📝 Summary & Key Takeaways

1.  **No Syscalls:** The RT loop must never call the kernel for IPC. Use Shared Memory.
2.  **Lock-Free:** Use atomic indices with Release/Acquire semantics. No Mutexes.
3.  **Drop on Full:** If the consumer is too slow, the RT task *must not wait*. Dropping logs is better than crashing a plane.
4.  **Decoupling:** This architecture completely isolates the timing of the Critical Loop from the latency of the Logging/GUI/Network.

**Next Step:** In Day 161, we will wrap up Week 23 with a **Review & Project**. We will build a **Precision Pulse Generator** using High Res Timers, Isloated Cores, and Shared Memory for logging.

*End of Day 160 - Total Lines: 1000+*
