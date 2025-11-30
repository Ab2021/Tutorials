# Day 133: Deferred Work - Tasklets & Workqueues
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
1.  **Explain** the need for Deferred Work (Bottom Halves).
2.  **Implement** a **Tasklet** (Atomic Context) for high-priority deferred work.
3.  **Implement** a **Workqueue** (Process Context) for sleepable deferred work.
4.  **Choose** the correct mechanism based on latency and sleeping requirements.
5.  **Use** Kernel Timers (`timer_list`) for scheduling future events.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 132 (Interrupts).
    *   Atomic vs Process Context.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "Top Half" Limitation
The ISR (Top Half) must be fast. It blocks other interrupts.
*   **Problem:** What if an interrupt needs to write to an I2C device (slow, might sleep) or decrypt a packet (CPU intensive)?
*   **Solution:** Do the bare minimum in ISR (Ack IRQ, copy data to buffer), then schedule the rest for later.

### 🔹 Part 2: Tasklets (The Old Way)
*   **Context:** Atomic (Interrupt Context).
*   **Sleeping:** NO.
*   **Priority:** Higher than normal processes.
*   **Usage:** Quick data processing that shouldn't block the CPU for too long.
*   *Note: Tasklets are being phased out in modern kernels in favor of threaded IRQs, but are still widely used.*

### 🔹 Part 3: Workqueues (The Standard Way)
*   **Context:** Process Context (Kernel Thread).
*   **Sleeping:** YES.
*   **Priority:** Normal scheduling.
*   **Usage:** Anything that needs to sleep (I/O, Mutexes) or takes a long time.

---

## 💻 Implementation: The Workqueue Demo

> **Instruction:** We will modify the Button Driver from Day 132.
> *   **ISR:** Acknowledges button press.
> *   **Workqueue:** Prints a long message and "simulates" heavy work (msleep).

### 👨‍💻 Code Implementation

#### Step 1: Source Code (`wq_driver.c`)

```c
#include <linux/module.h>
#include <linux/init.h>
#include <linux/interrupt.h>
#include <linux/workqueue.h>
#include <linux/slab.h>
#include <linux/delay.h>

#define IRQ_NO 1 // Fake IRQ for demo (or use GPIO logic from Day 132)

// 1. Declare the Work Structure
static struct work_struct my_work;

// 2. The Work Handler (Bottom Half)
// This runs in Process Context. We CAN sleep here.
void my_work_handler(struct work_struct *work) {
    printk(KERN_INFO "Workqueue: Starting heavy task...\n");
    msleep(1000); // Sleep for 1 second (Simulate I/O)
    printk(KERN_INFO "Workqueue: Task finished.\n");
}

// 3. The ISR (Top Half)
// This runs in Interrupt Context. NO sleeping.
static irqreturn_t my_isr(int irq, void *dev_id) {
    printk(KERN_INFO "ISR: Interrupt received. Scheduling work.\n");
    
    // Schedule the work
    schedule_work(&my_work);
    
    return IRQ_HANDLED;
}

static int __init wq_init(void) {
    // Initialize Work
    INIT_WORK(&my_work, my_work_handler);
    
    // Request IRQ (Simulated)
    // In real code, use gpio_to_irq()
    // request_irq(IRQ_NO, my_isr, IRQF_SHARED, "wq_test", (void *)(my_isr));
    
    printk(KERN_INFO "Workqueue Driver Loaded.\n");
    
    // Simulate an interrupt trigger for testing without hardware
    my_isr(0, NULL); 
    
    return 0;
}

static void __exit wq_exit(void) {
    // Ensure work is done before unloading
    cancel_work_sync(&my_work);
    // free_irq(IRQ_NO, (void *)(my_isr));
    printk(KERN_INFO "Workqueue Driver Unloaded.\n");
}

module_init(wq_init);
module_exit(wq_exit);
MODULE_LICENSE("GPL");
```

---

## 💻 Implementation: Kernel Timers

> **Instruction:** Sometimes you don't need an interrupt, you just need to run something 5 seconds from now.

### 👨‍💻 Code Implementation

#### Step 1: Source Code (`timer_driver.c`)

```c
#include <linux/module.h>
#include <linux/timer.h>

static struct timer_list my_timer;

// Timer Callback (Atomic Context!)
void my_timer_callback(struct timer_list *t) {
    printk(KERN_INFO "Timer: Tick Tock!\n");
    
    // Re-arm timer for 1 second later (Periodic)
    mod_timer(&my_timer, jiffies + msecs_to_jiffies(1000));
}

static int __init timer_init(void) {
    printk(KERN_INFO "Timer: Module Loaded.\n");
    
    // Setup Timer
    timer_setup(&my_timer, my_timer_callback, 0);
    
    // Start Timer (1 second from now)
    mod_timer(&my_timer, jiffies + msecs_to_jiffies(1000));
    
    return 0;
}

static void __exit timer_exit(void) {
    del_timer(&my_timer); // Delete timer
    printk(KERN_INFO "Timer: Module Unloaded.\n");
}

module_init(timer_init);
module_exit(timer_exit);
MODULE_LICENSE("GPL");
```

---

## 🔬 Lab Exercise: Lab 133.1 - Delayed Work

### 1. Lab Objectives
- Use `delayed_work` instead of standard `work_struct`.
- Schedule work to run 5 seconds *after* the interrupt.

### 2. Step-by-Step Guide
1.  Declare: `static struct delayed_work my_delayed_work;`
2.  Init: `INIT_DELAYED_WORK(&my_delayed_work, handler_fn);`
3.  Schedule: `schedule_delayed_work(&my_delayed_work, msecs_to_jiffies(5000));`
4.  Exit: `cancel_delayed_work_sync(&my_delayed_work);`

---

## 🧪 Additional / Advanced Labs

### Lab 2: Custom Workqueue
- **Goal:** Don't use the system-wide default queue.
- **Task:**
    1.  `create_workqueue("my_queue")`.
    2.  `queue_work(my_wq, &work)`.
    3.  `destroy_workqueue(my_wq)`.
    4.  **Why?** Prevents your driver from blocking other system tasks if your work takes a long time.

### Lab 3: Tasklets (Legacy)
- **Goal:** Implement a tasklet.
- **Task:**
    1.  `DECLARE_TASKLET(my_tasklet, tasklet_fn, data);`
    2.  ISR: `tasklet_schedule(&my_tasklet);`
    3.  Verify that `tasklet_fn` runs in atomic context (try `msleep` -> Crash).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Module unload hang"
*   **Cause:** You unloaded the module while a work item was still pending or running.
*   **Fix:** ALWAYS call `cancel_work_sync` or `del_timer_sync` in the `__exit` function.

#### 2. "Kernel Panic: scheduling while atomic" in Timer
*   **Cause:** You tried to sleep inside the timer callback.
*   **Fact:** Timers run in Software Interrupt context (Atomic). No sleeping allowed. Use a Workqueue if you need to sleep.

---

## ⚡ Optimization & Best Practices

### Concurrency
*   **Reentrancy:** If you schedule work, and the interrupt fires *again* before the work runs, the work is **not** queued twice. It runs once.
*   If you need it to run twice, you need to manage your own data queue.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the main difference between Tasklet and Workqueue?
    *   **A:** Tasklet = Atomic (No Sleep), Higher Priority. Workqueue = Process (Can Sleep), Lower Priority.
2.  **Q:** When should I use a Custom Workqueue?
    *   **A:** When your work is performance-critical or very long-running, and you don't want to contend with the shared system workqueue.

### Challenge Task
> **Task:** "The Morse Code Flasher".
> 1.  User writes string "SOS" to `/dev/morse`.
> 2.  Driver parses string.
> 3.  Uses a **Workqueue** to blink an LED (GPIO) with correct timing (Dot=200ms, Dash=600ms).
> 4.  Must handle `msleep` for timing.

---

## 📚 Further Reading & References
- [Linux Device Drivers 3, Chapter 7: Time, Delays, and Deferred Work](https://lwn.net/Kernel/LDD3/)
- [Kernel API: workqueue.h](https://www.kernel.org/doc/html/latest/core-api/workqueue.html)

---
