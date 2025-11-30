# Day 130: Concurrency & Race Conditions
## Phase 2: Linux Kernel & Device Drivers | Week 19: Character Device Drivers

---

> **📝 Content Creator Instructions:**
> This document is designed to produce **comprehensive, industry-grade educational content**. 
> - **Target Length:** The final filled document should be approximately **1000+ lines** of detailed markdown.
> - **Depth:** Do not skim over details. Explain *why*, not just *how*.
> - **Structure:** If a topic is complex, **DIVIDE IT INTO MULTIPLE PARTS** (Part 1, Part 2, etc.).
> - **Code:** Provide complete, compilable code examples, not just snippets.
> - **Visuals:** Use Mermaid diagrams for flows, architectures, and state machines.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Identify** Race Conditions in kernel drivers.
2.  **Implement** Mutual Exclusion using `struct mutex`.
3.  **Implement** Fast Locking using `spinlock_t`.
4.  **Distinguish** between Sleeping Locks (Mutex) and Atomic Locks (Spinlock).
5.  **Use** Atomic Variables (`atomic_t`) for lock-free counters.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 128 (Char Drivers).
    *   Basic Threading concepts (pthreads).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Concurrency Problem
Linux is a **Symmetric Multi-Processing (SMP)** OS.
*   **Scenario:** Two CPU cores (Core A and Core B) run code that accesses the *same* global variable in your driver at the *same* time.
*   **The Race:**
    1.  Core A reads `count` (val = 5).
    2.  Core B reads `count` (val = 5).
    3.  Core A increments (val = 6) and writes back.
    4.  Core B increments (val = 6) and writes back.
    *   **Result:** `count` is 6, but it should be 7. Data Corruption!

### 🔹 Part 2: Synchronization Primitives

#### 1. Atomic Variables (`atomic_t`)
*   **Use Case:** Simple integer counters.
*   **Mechanism:** CPU hardware instructions (`LOCK` prefix on x86, `LDREX/STREX` on ARM).
*   **Cost:** Very cheap. No locking overhead.

#### 2. Mutex (`struct mutex`)
*   **Use Case:** Protecting large critical sections where you might **sleep** (e.g., `copy_from_user`, `kmalloc`).
*   **Mechanism:** If lock is taken, the calling process is put to **sleep** (Context Switch) until lock is free.
*   **Rule:** NEVER use in Interrupt Context.

#### 3. Spinlock (`spinlock_t`)
*   **Use Case:** Protecting small critical sections in **Interrupt Context**.
*   **Mechanism:** If lock is taken, the CPU **spins** (busy-wait loop) until lock is free.
*   **Rule:** NEVER sleep while holding a spinlock. Fast, but wastes CPU cycles if held too long.

---

## 💻 Implementation: The Race Condition Demo

> **Instruction:** We will create a driver with a shared counter and simulate a race.

### 👨‍💻 Code Implementation

#### Step 1: The Vulnerable Driver (`race_driver.c`)

```c
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/delay.h> // For msleep

#define DEVICE_NAME "race_dev"

static int shared_resource = 0;

static ssize_t my_read(struct file *file, char __user *user_buf, size_t count, loff_t *offset) {
    int local_copy;
    
    // CRITICAL SECTION START
    local_copy = shared_resource;
    printk(KERN_INFO "Race: Read %d\n", local_copy);
    
    // Simulate complex processing to widen the race window
    msleep(100); 
    
    local_copy++;
    shared_resource = local_copy;
    // CRITICAL SECTION END

    printk(KERN_INFO "Race: Wrote %d\n", shared_resource);
    return 0;
}

// ... Standard cdev init/exit ...
```

#### Step 2: The Attacker Script (`test_race.sh`)
Run two processes in parallel that read the device.
```bash
#!/bin/bash
cat /dev/race_dev &
cat /dev/race_dev &
wait
```

#### Step 3: Observe Failure
If run sequentially:
1.  Read 0 -> Write 1.
2.  Read 1 -> Write 2.
Final: 2.

If run in parallel (Race):
1.  P1 Reads 0.
2.  P2 Reads 0.
3.  P1 Writes 1.
4.  P2 Writes 1.
Final: 1. **Data Lost!**

---

## 💻 Implementation: Fixing with Mutex

> **Instruction:** Protect the critical section.

### 👨‍💻 Code Implementation

#### Step 1: Update Driver (`mutex_driver.c`)

```c
#include <linux/mutex.h> // Header

static DEFINE_MUTEX(my_mutex); // Declare and Init

static ssize_t my_read(struct file *file, char __user *user_buf, size_t count, loff_t *offset) {
    int local_copy;
    
    // Lock
    if (mutex_lock_interruptible(&my_mutex)) {
        return -ERESTARTSYS; // Interrupted by signal
    }

    // CRITICAL SECTION START
    local_copy = shared_resource;
    msleep(100); 
    local_copy++;
    shared_resource = local_copy;
    // CRITICAL SECTION END

    // Unlock
    mutex_unlock(&my_mutex);

    return 0;
}
```

#### Step 2: Verify
Run the script again.
1.  P1 takes lock.
2.  P2 tries to take lock -> Sleeps.
3.  P1 finishes, unlocks.
4.  P2 wakes up, takes lock.
Final: 2. **Correct!**

---

## 💻 Implementation: Fixing with Spinlock

> **Instruction:** Use spinlock (Only if we don't sleep!).
> *Note: We must remove `msleep` because we cannot sleep inside a spinlock.*

### 👨‍💻 Code Implementation

```c
#include <linux/spinlock.h>

static DEFINE_SPINLOCK(my_lock);

static ssize_t my_read(...) {
    unsigned long flags;
    
    // Lock and Disable Local Interrupts (Safest)
    spin_lock_irqsave(&my_lock, flags);

    // CRITICAL SECTION (Must be fast!)
    shared_resource++;

    spin_unlock_irqrestore(&my_lock, flags);
    
    return 0;
}
```

---

## 🔬 Lab Exercise: Lab 130.1 - Atomic Counters

### 1. Lab Objectives
- Replace the `int shared_resource` with `atomic_t`.
- Verify that no locks are needed for simple increments.

### 2. Step-by-Step Guide
1.  Declare: `static atomic_t my_counter = ATOMIC_INIT(0);`
2.  In `read`:
    ```c
    atomic_inc(&my_counter);
    int val = atomic_read(&my_counter);
    printk("Counter: %d\n", val);
    ```
3.  Run the parallel script. The counter should be accurate.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Reader-Writer Semaphores (`rw_semaphore`)
- **Goal:** Optimize for "Many Readers, Few Writers".
- **Task:**
    1.  Use `down_read()` for reading (allows multiple concurrent readers).
    2.  Use `down_write()` for writing (exclusive access).
    3.  Demonstrate that 10 readers can access simultaneously, but a writer blocks them.

### Lab 3: Deadlock Simulation
- **Goal:** Understand the "Deadly Embrace".
- **Task:**
    1.  Create two mutexes: `lockA` and `lockB`.
    2.  Thread 1: Takes A, sleeps, takes B.
    3.  Thread 2: Takes B, sleeps, takes A.
    4.  Run them. System hangs.
    5.  Enable `CONFIG_LOCKDEP` in kernel config to see the kernel detect and report the deadlock.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "BUG: scheduling while atomic"
*   **Cause:** You called a sleeping function (like `msleep`, `copy_from_user`, `kmalloc(GFP_KERNEL)`) while holding a **Spinlock**.
*   **Fix:** Use a Mutex instead, or use `GFP_ATOMIC` for allocation.

#### 2. System Freeze (Deadlock)
*   **Cause:** You took a lock and forgot to unlock it in an error path.
*   **Fix:** Use `goto out;` pattern where `out:` label handles unlocking.

---

## ⚡ Optimization & Best Practices

### Lock Granularity
- **Coarse Grained:** One big lock for the whole driver. Simple, but poor performance on many cores.
- **Fine Grained:** One lock per object/queue. Better parallelism, but complex to avoid deadlocks.

### Per-CPU Variables
- If each CPU has its own counter, you don't need locks at all!
- Use `get_cpu_var()` and `put_cpu_var()`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can I use a Mutex in an Interrupt Handler?
    *   **A:** NO. Interrupt handlers cannot sleep. Mutexes might sleep. Use Spinlocks.
2.  **Q:** What is `spin_lock_irqsave`?
    *   **A:** It takes the lock AND disables interrupts on the local CPU. This prevents an interrupt handler from running and trying to take the same lock (which would cause a deadlock).

### Challenge Task
> **Task:** "The Bank Account". Create a driver managing a `balance`.
> 1.  `ioctl` DEPOSIT.
> 2.  `ioctl` WITHDRAW.
> 3.  Ensure that if two people withdraw simultaneously, the balance never goes negative or becomes corrupt. Use a Mutex.

---

## 📚 Further Reading & References
- [Linux Device Drivers 3, Chapter 5: Concurrency and Race Conditions](https://lwn.net/Kernel/LDD3/)
- [Kernel Locking Guide](https://www.kernel.org/doc/Documentation/locking/locking-explained.txt)

---
