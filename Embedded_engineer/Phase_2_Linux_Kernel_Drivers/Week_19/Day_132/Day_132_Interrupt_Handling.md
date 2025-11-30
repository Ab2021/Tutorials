# Day 132: Interrupt Handling in Drivers
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
1.  **Request** and **Free** Interrupt Lines (IRQs) using `request_irq` and `free_irq`.
2.  **Implement** an Interrupt Service Routine (ISR) in Kernel Space.
3.  **Map** GPIO pins to IRQ numbers using `gpiod_to_irq`.
4.  **Explain** the constraints of Interrupt Context (No sleeping!).
5.  **Debounce** buttons using software timers (jiffies).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
    *   (Optional) Raspberry Pi or similar for real GPIO testing.
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 125 (Device Tree - GPIOs).
    *   Interrupt concepts (Vector table, Context save).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Interrupt Context vs Process Context
*   **Process Context:** Code running on behalf of a user process (e.g., `read()`, `ioctl()`). Can sleep. Has a valid `current` pointer (PID).
*   **Interrupt Context:** Code running because hardware screamed "Attention!".
    *   **NO Sleeping:** Cannot call `msleep`, `mutex_lock`, `copy_from_user`.
    *   **Priority:** Preempts everything else.
    *   **Goal:** Be fast. Acknowledge hardware, copy data, schedule work for later.

### 🔹 Part 2: The Top Half vs Bottom Half
*   **Top Half (Hard IRQ):** The actual ISR (`irq_handler_t`). Runs with interrupts disabled (on local CPU). Must be extremely fast (microseconds).
*   **Bottom Half (Soft IRQ / Tasklet / Workqueue):** Deferred work. Runs later with interrupts enabled. (Covered in Day 133).

### 🔹 Part 3: Requesting an IRQ
```c
int request_irq(unsigned int irq, irq_handler_t handler, unsigned long flags, const char *name, void *dev);
```
*   `irq`: The number (e.g., 42).
*   `handler`: Your function.
*   `flags`: `IRQF_SHARED` (shared line), `IRQF_TRIGGER_RISING` (edge).
*   `dev`: Cookie passed back to handler (usually your device struct).

---

## 💻 Implementation: The Button Driver

> **Instruction:** We will create a driver that listens for a GPIO button press and logs it.
> *Note: In QEMU, simulating GPIO interrupts is tricky without custom patches. We will write the code as if for a Raspberry Pi or generic ARM board, but explain how to test logic.*

### 👨‍💻 Code Implementation

#### Step 1: Source Code (`button_irq.c`)

```c
#include <linux/module.h>
#include <linux/init.h>
#include <linux/gpio.h>
#include <linux/interrupt.h>
#include <linux/jiffies.h>

#define GPIO_BUTTON 17  // Example: GPIO 17 on RPi
#define DEVICE_NAME "button_irq"

static unsigned int irq_number;
static unsigned int press_count = 0;

// Debounce logic
static unsigned long last_interrupt_time = 0;
static unsigned long debounce_time = 200; // 200ms

// --- Interrupt Handler (Top Half) ---
static irqreturn_t button_isr(int irq, void *dev_id) {
    unsigned long current_time = jiffies;

    // Simple Debounce
    if ((current_time - last_interrupt_time) < msecs_to_jiffies(debounce_time)) {
        return IRQ_HANDLED; // Ignore noise
    }
    last_interrupt_time = current_time;

    press_count++;
    printk(KERN_INFO "Button: Interrupt! Count = %d\n", press_count);

    return IRQ_HANDLED;
}

// --- Init ---
static int __init button_init(void) {
    int result;

    // 1. Request GPIO
    if (!gpio_is_valid(GPIO_BUTTON)) {
        printk(KERN_ERR "Button: Invalid GPIO %d\n", GPIO_BUTTON);
        return -ENODEV;
    }
    gpio_request(GPIO_BUTTON, "sysfs");
    gpio_direction_input(GPIO_BUTTON);
    gpio_set_debounce(GPIO_BUTTON, 200); // Hardware debounce if supported
    gpio_export(GPIO_BUTTON, false);

    // 2. Map GPIO to IRQ Number
    irq_number = gpio_to_irq(GPIO_BUTTON);
    printk(KERN_INFO "Button: GPIO %d mapped to IRQ %d\n", GPIO_BUTTON, irq_number);

    // 3. Request IRQ
    result = request_irq(irq_number,             // IRQ Number
                         (irq_handler_t) button_isr, // Handler
                         IRQF_TRIGGER_RISING,    // Trigger on Rising Edge
                         "button_handler",       // Name in /proc/interrupts
                         NULL);                  // Dev ID (Cookie)

    if (result) {
        printk(KERN_ERR "Button: Failed to request IRQ: %d\n", result);
        gpio_free(GPIO_BUTTON);
        return result;
    }

    printk(KERN_INFO "Button: Module Loaded.\n");
    return 0;
}

// --- Exit ---
static void __exit button_exit(void) {
    free_irq(irq_number, NULL);
    gpio_free(GPIO_BUTTON);
    printk(KERN_INFO "Button: Module Unloaded.\n");
}

module_init(button_init);
module_exit(button_exit);
MODULE_LICENSE("GPL");
```

---

## 💻 Implementation: Testing on Hardware (or Sim)

> **Instruction:** If you have a Raspberry Pi, compile this against the Pi kernel headers.

### 👨‍💻 Command Line Steps

#### Step 1: Load
```bash
insmod button_irq.ko
```

#### Step 2: Verify Registration
Check `/proc/interrupts`.
```bash
cat /proc/interrupts | grep button
# Output: 188:          0  pinctrl-bcm2835  17 Edge      button_handler
```
This confirms the kernel knows about our handler.

#### Step 3: Trigger
Press the physical button connected to GPIO 17.
Check `dmesg`.
```text
Button: Interrupt! Count = 1
Button: Interrupt! Count = 2
```

---

## 🔬 Lab Exercise: Lab 132.1 - Shared Interrupts

### 1. Lab Objectives
- Understand how multiple devices share one IRQ line (common in PCI).
- Modify `request_irq` to use `IRQF_SHARED`.

### 2. Step-by-Step Guide
1.  Change flags to `IRQF_SHARED | IRQF_TRIGGER_RISING`.
2.  **Crucial:** You MUST pass a unique `dev_id` (not NULL).
    ```c
    request_irq(..., &my_dev_struct);
    ```
3.  In the ISR, check if the interrupt was actually for you. (For GPIO, you can't really tell easily without reading a status register, but for PCI you read the device's status register).
4.  If not for you, return `IRQ_NONE`. The kernel will then try the next handler in the chain.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Threaded IRQs (`request_threaded_irq`)
- **Goal:** Reduce latency impact of Top Halves.
- **Task:**
    1.  Use `request_threaded_irq(irq, handler, thread_fn, ...)`
    2.  `handler` returns `IRQ_WAKE_THREAD`.
    3.  `thread_fn` runs in a kernel thread (Process Context!).
    4.  Now you can sleep (e.g., read I2C) inside `thread_fn`.

### Lab 3: Polling vs Interrupts
- **Goal:** Compare CPU usage.
- **Task:**
    1.  Write a driver that polls the GPIO in a loop (using `while` and `cpu_relax`).
    2.  Run `top`. CPU usage will be 100% on one core.
    3.  Load the Interrupt driver. CPU usage ~0%.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Scheduling while atomic"
*   **Cause:** You called `printk` (which is usually safe, but slow) or `kmalloc` or `mutex_lock` inside the ISR.
*   **Fix:** Move heavy logic to Bottom Half (Day 133).

#### 2. Interrupt Storm
*   **Cause:** You requested `IRQF_TRIGGER_LEVEL_LOW` but didn't clear the interrupt source in the hardware. The line stays Low.
*   **Result:** As soon as ISR exits, it triggers again. System hangs.
*   **Fix:** Always acknowledge/clear the interrupt in the hardware registers (if applicable).

---

## ⚡ Optimization & Best Practices

### Reentrancy
*   An ISR on CPU 0 can be interrupted by a higher priority interrupt.
*   An ISR for IRQ X will generally *not* be interrupted by IRQ X (masked), but can be interrupted by IRQ Y.
*   **Spinlocks:** If your ISR shares data with process context, the process context must use `spin_lock_irqsave` to disable IRQs locally while accessing the data.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What does `IRQ_HANDLED` mean?
    *   **A:** It tells the kernel "This was my interrupt, I handled it." If you return `IRQ_NONE`, the kernel thinks the interrupt is spurious (or belongs to another shared device).
2.  **Q:** Why can't I use `copy_to_user` in an ISR?
    *   **A:** `copy_to_user` might cause a Page Fault. Handling a Page Fault requires sleeping (waiting for disk). You cannot sleep in ISR.

### Challenge Task
> **Task:** "The Morse Code Decoder". Connect a button. Measure the time between Press and Release (using `jiffies`).
> *   Short press (< 200ms) = Dot.
> *   Long press (> 200ms) = Dash.
> *   Accumulate dots/dashes and print the character to dmesg.

---

## 📚 Further Reading & References
- [Linux Device Drivers 3, Chapter 10: Interrupt Handling](https://lwn.net/Kernel/LDD3/)
- [Kernel API: interrupt.h](https://www.kernel.org/doc/html/latest/core-api/genericirq.html)

---
