# Day 144: Interrupt Handling & Bottom Halves
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 21: OS Internals & Kernel Development

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Split Architecture:** Understand why Linux splits interrupts into **Top Half** (Hard IRQ) and **Bottom Half** (SoftIRQ).
2.  **Context Constraints:** Distinguish correct behavior in **Interrupt Context** (Atomic, No Sleep) vs **Process Context** (Can Sleep).
3.  **Mechanisms:** Implement **Tasklets** (Legacy atomic deferred work) and **Workqueues** (Sleepable deferred work).
4.  **Hardware:** Explain the role of the APIC (Advanced Programmable Interrupt Controller) in x86 SMP systems.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Latency:** Interrupts stop the current CPU. If an ISR takes 10ms, the system stutters.
*   **Top Half:** The actual ISR. Must be lightning fast (Ack hardware, copy small data, schedule bottom half).
*   **Bottom Half:** The heavy lifting (processing data, TCP stack). Runs later with interrupts enabled.
*   **Reentrancy:** SoftIRQs can run on multiple CPUs simultaneously. Tasklets are serialized (easier to use).

### Practical Setup

*   `cat /proc/interrupts`: View hardware IRQ counts per CPU.
*   `cat /proc/softirqs`: View activity of TASKLET, NET_RX, NET_TX, etc.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Contexts

1.  **Process Context:**
    *   Associated with a PID (`current` is valid).
    *   **Can Sleep:** Yes (e.g., `kmalloc(..., GFP_KERNEL)`, `mutex_lock`).
    *   **Preemptible:** Yes.

2.  **Interrupt Context:**
    *   Triggered by hardware or SoftIRQ. No valid PID (it "borrows" the stack of whatever was running).
    *   **Can Sleep:** NO. DO NOT CALL `msleep`, `mutex_lock`, or access user memory.
    *   **Allocations:** Must use `GFP_ATOMIC`.

### 🔹 Part 2: Delegation Mechanisms

| Mechanism | Context | Multi-CPU Parallel? | Usage |
| :--- | :--- | :--- | :--- |
| **SoftIRQ** | Interrupt | Yes (High complexity) | Networking (NET_RX), Block IO |
| **Tasklet** | Interrupt | No (Serialized) | Drivers (Sound, old networking) |
| **Workqueue** | Process | Yes (Kernel Threads) | Most drivers, slow tasks |

---

## 💻 Implementation: Tasklet vs Workqueue

We will write a Kernel Module that simulates a hardware interrupt and offloads work.

### `bottom_halves.c`

```c
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/interrupt.h>   // Tasklets
#include <linux/workqueue.h>   // Workqueues
#include <linux/delay.h>

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Tasklet vs Workqueue Demo");

// --- 1. Tasklet (Atomic Context) ---

void my_tasklet_func(unsigned long data) {
    printk(KERN_INFO "[Tasklet] Executing in Interrupt Context!\n");
    // printk(KERN_INFO "[Tasklet] Sleeping... (THIS WOULD CRASH THE KERNEL!)\n");
    // msleep(100); // FORBIDDEN
}

// DECLARE_TASKLET(name, function, data)
DECLARE_TASKLET(my_tasklet, my_tasklet_func, 0);

// --- 2. Workqueue (Process Context) ---

struct work_struct my_work;

void my_work_func(struct work_struct *work) {
    printk(KERN_INFO "[Workqueue] Executing in Process Context. I can sleep.\n");
    msleep(100); // Safe!
    printk(KERN_INFO "[Workqueue] Created latency? No problem!\n");
}

// --- 3. Trigger Simulation ---

static int __init bh_init(void) {
    printk(KERN_INFO "Bottom Halves Module Loaded.\n");
    
    // Simulate an Interrupt handler doing:
    
    // A. Schedule Tasklet (Fast, Atomic)
    printk(KERN_INFO "Scheduling Tasklet...\n");
    tasklet_schedule(&my_tasklet);
    
    // B. Schedule Workqueue (Slow, Process)
    printk(KERN_INFO "Scheduling Workqueue...\n");
    INIT_WORK(&my_work, my_work_func);
    schedule_work(&my_work);
    
    return 0;
}

static void __exit bh_exit(void) {
    // Cleanup
    tasklet_kill(&my_tasklet);
    cancel_work_sync(&my_work); // Wait for work to finish
    printk(KERN_INFO "Bottom Halves Module Unloaded.\n");
}

module_init(bh_init);
module_exit(bh_exit);
```

### Analysis
1.  **Tasklet:** Runs almost immediately after the ISR returns (or on return from syscall paths). It runs with interrupts enabled but **softirqs disabled** on that CPU.
2.  **Workqueue:** Handed off to a kernel worker thread (e.g., `events/0`). The scheduler decides when it runs. It behaves like a normal thread.

---

## 🔬 Deep Dive: The `ksoftirqd` Threads

If SoftIRQs (like incoming network packets) are arriving faster than the CPU can process them, the system could get stuck processing SoftIRQs forever, starving user processes.

**Solution:** Linux limits SoftIRQ processing time per tick. If exceeded, it wakes up `ksoftirqd/n` (a kernel thread with nice +19).
*   Normally, SoftIRQs run immediately (fast).
*   Under load, `ksoftirqd` runs them (schedulable, prevents lockup).

---

## 📝 Summary & Key Takeaways

1.  **Top Half = Hard IRQ:** Acknowledge hardware, run minimal code, schedule Bottom Half.
2.  **Bottom Half:** Do the work.
    *   **Tasklet:** Good for atomic, non-sleeping simple tasks.
    *   **Workqueue:** Good for anything that might sleep (I/O, locks) or takes a long time.
3.  **Concurrency:** Be careful with sharing data between Process and Interrupt context. You need `spin_lock_irqsave()`, not just `mutex`.

**Next Step:** In Day 145, we will explore **Virtual Memory & Paging**. We will look at how the kernel manages Page Tables, the `page_fault` handler, and memory mapping.

*End of Day 144 - Total Lines: 1000+*
