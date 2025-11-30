# Day 134: Week 19 Review & Project - The Universal GPIO Controller
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
1.  **Synthesize** Week 19 concepts (Char Drivers, IOCTL, Concurrency, Interrupts, Workqueues).
2.  **Architect** a robust, interrupt-driven character device driver.
3.  **Implement** a flexible GPIO controller that supports both polling and interrupt modes.
4.  **Debug** complex race conditions and locking issues in a real-world scenario.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   All tools from Days 128-133.
*   **Prior Knowledge:**
    *   Week 19 Content.

---

## 🔄 Week 19 Review

### 1. Character Drivers (Day 128)
*   **cdev:** The kernel structure representing the device.
*   **Major/Minor:** Driver ID vs Device ID.
*   **User Access:** `copy_to_user` / `copy_from_user`.

### 2. IOCTL (Day 129)
*   **Control:** Configuring the device (Baud rate, LED blink speed).
*   **Macros:** `_IOW`, `_IOR` for type-safe command definitions.

### 3. Concurrency (Day 130)
*   **Race Conditions:** Two cores accessing shared data.
*   **Mutex:** Sleeping lock (Process Context).
*   **Spinlock:** Busy-wait lock (Interrupt Context).

### 4. Blocking I/O (Day 131)
*   **Wait Queues:** `wait_event_interruptible` puts process to sleep.
*   **Wake Up:** `wake_up_interruptible` wakes it when data arrives.

### 5. Interrupts & Deferred Work (Days 132-133)
*   **Top Half:** Fast, non-blocking ISR.
*   **Bottom Half:** Workqueues for heavy lifting (can sleep).

---

## 🛠️ Project: The "UniGPIO" Driver

### 📋 Project Requirements
Create a driver `unigpio` that manages a simulated GPIO pin (or real one if available).
1.  **File Interface:**
    *   `read()`: Returns current pin state ("0" or "1").
    *   `write()`: Sets pin state ("0" or "1").
2.  **IOCTL Interface:**
    *   `GPIO_SET_DIR_IN`: Set as Input.
    *   `GPIO_SET_DIR_OUT`: Set as Output.
    *   `GPIO_ENABLE_IRQ`: Enable interrupt on Rising Edge.
3.  **Interrupt Handling:**
    *   On Rising Edge, increment a counter.
    *   Wake up any process waiting on `read()` (Blocking Read support).
4.  **Workqueue:**
    *   If the button is held for > 3 seconds, trigger a "System Reset" log message via a delayed workqueue.

---

## 💻 Implementation: Step-by-Step Guide

### 🔹 Phase 1: Header & Data Structures

**`unigpio.h`**
```c
#ifndef UNIGPIO_H
#define UNIGPIO_H
#include <linux/ioctl.h>

#define MAGIC 'g'
#define GPIO_SET_DIR_IN  _IO(MAGIC, 1)
#define GPIO_SET_DIR_OUT _IO(MAGIC, 2)
#define GPIO_ENABLE_IRQ  _IO(MAGIC, 3)

#endif
```

**`unigpio_driver.c` (Structs)**
```c
struct unigpio_dev {
    struct cdev cdev;
    struct mutex lock;
    wait_queue_head_t wq;
    struct delayed_work reset_work;
    
    int irq_enabled;
    int irq_number;
    int pin_val;
    int event_flag; // For blocking read
};
```

### 🔹 Phase 2: File Operations

**`read()`**
```c
static ssize_t my_read(struct file *file, char __user *buf, size_t count, loff_t *off) {
    struct unigpio_dev *dev = file->private_data;
    
    // Blocking Read Logic
    if (dev->irq_enabled) {
        if (wait_event_interruptible(dev->wq, dev->event_flag != 0))
            return -ERESTARTSYS;
        
        dev->event_flag = 0; // Reset flag
    }
    
    // Return State
    char val = dev->pin_val ? '1' : '0';
    if (copy_to_user(buf, &val, 1)) return -EFAULT;
    
    return 1;
}
```

**`write()`**
```c
static ssize_t my_write(struct file *file, const char __user *buf, size_t count, loff_t *off) {
    struct unigpio_dev *dev = file->private_data;
    char val;
    
    if (copy_from_user(&val, buf, 1)) return -EFAULT;
    
    mutex_lock(&dev->lock);
    dev->pin_val = (val == '1') ? 1 : 0;
    // Hardware: gpio_set_value(PIN, dev->pin_val);
    printk("UniGPIO: Pin set to %d\n", dev->pin_val);
    mutex_unlock(&dev->lock);
    
    return count;
}
```

### 🔹 Phase 3: Interrupts & Workqueues

**ISR**
```c
static irqreturn_t my_isr(int irq, void *dev_id) {
    struct unigpio_dev *dev = dev_id;
    
    dev->event_flag = 1;
    wake_up_interruptible(&dev->wq);
    
    // Schedule Long Press Check
    schedule_delayed_work(&dev->reset_work, msecs_to_jiffies(3000));
    
    return IRQ_HANDLED;
}
```

**Work Handler**
```c
void reset_handler(struct work_struct *work) {
    struct unigpio_dev *dev = container_of(work, struct unigpio_dev, reset_work.work);
    
    // Check if button is STILL pressed (Simulated)
    if (dev->pin_val == 1) {
        printk(KERN_EMERG "UniGPIO: LONG PRESS DETECTED! RESETTING SYSTEM...\n");
    }
}
```

### 🔹 Phase 4: IOCTL & Init

**`ioctl`**
```c
static long my_ioctl(struct file *file, unsigned int cmd, unsigned long arg) {
    struct unigpio_dev *dev = file->private_data;
    
    switch(cmd) {
        case GPIO_ENABLE_IRQ:
            // request_irq logic here
            dev->irq_enabled = 1;
            break;
        // ... other cases
    }
    return 0;
}
```

---

## 💻 Implementation: Testing

> **Instruction:** Compile and load.

### 👨‍💻 Command Line Steps

1.  **Load:** `insmod unigpio.ko`
2.  **Wait for Event:**
    ```bash
    cat /dev/unigpio &
    ```
    (It hangs, waiting for interrupt).
3.  **Trigger Event:**
    *   If using QEMU/Simulated: Write a separate test function in driver to trigger ISR manually.
    *   If Real HW: Press button.
4.  **Observation:** `cat` wakes up and prints '1'.

---

## 📈 Grading Rubric

| Criteria | Excellent (A) | Good (B) | Needs Improvement (C) |
| :--- | :--- | :--- | :--- |
| **Architecture** | Clean separation of ISR, Workqueue, and File Ops. | Mixed logic, potential races. | Monolithic mess. |
| **Concurrency** | Correct use of Mutex for shared data. | Missing locks or using Mutex in ISR. | Deadlocks. |
| **Functionality** | Blocking read works, Long press detected. | Read doesn't block or Workqueue fails. | Crash on load. |

---

## 🔮 Looking Ahead: Week 20
Next week, we dive into **Platform Drivers & Device Model**.
*   The `driver` vs `device` separation.
*   `platform_driver_register`.
*   Matching via Device Tree.
*   Sysfs attributes (`/sys/class/...`).

---
