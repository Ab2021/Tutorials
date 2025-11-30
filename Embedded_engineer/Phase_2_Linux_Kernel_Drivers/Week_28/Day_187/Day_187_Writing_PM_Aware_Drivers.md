# Day 187: Writing PM-Aware Drivers
## Phase 2: Linux Kernel & Device Drivers | Week 28: Power Management

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
1.  **Convert** a legacy driver to support Runtime PM and System Suspend.
2.  **Handle** Wake-up sources (`device_init_wakeup`, `enable_irq_wake`).
3.  **Manage** Concurrency between PM callbacks and Interrupts.
4.  **Implement** "Remote Wakeup" logic.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Development Board with a Button/GPIO.
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Week 28 Content.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Wakeup Problem
*   **Scenario:** System is suspended (S3). User presses a button.
*   **Requirement:** The button's IRQ must be configured to wake the CPU.
*   **API:** `enable_irq_wake(irq)`.

### 🔹 Part 2: Concurrency
*   **Race Condition:**
    *   `suspend()` is running.
    *   IRQ fires.
    *   Driver tries to access hardware that is half-off.
*   **Solution:**
    *   Disable IRQ in `suspend`.
    *   Use `spin_lock_irqsave`.
    *   Check `pm_runtime_active` status.

---

## 💻 Implementation: Wake-up Capable Button Driver

> **Instruction:** A GPIO button driver that can wake the system from suspend.

### 👨‍💻 Code Implementation

#### Step 1: Probe
```c
static int my_probe(struct platform_device *pdev) {
    // ... Request GPIO, IRQ ...
    
    // Mark device as wakeup capable
    device_init_wakeup(&pdev->dev, true);
    
    // Enable Runtime PM
    pm_runtime_set_active(&pdev->dev);
    pm_runtime_enable(&pdev->dev);
    
    return 0;
}
```

#### Step 2: Suspend
```c
static int my_suspend(struct device *dev) {
    struct my_data *data = dev_get_drvdata(dev);
    
    if (device_may_wakeup(dev)) {
        // Configure IRQ to wake system
        enable_irq_wake(data->irq);
    } else {
        // Not a wakeup source? Disable IRQ to save power
        disable_irq(data->irq);
    }
    
    return 0;
}
```

#### Step 3: Resume
```c
static int my_resume(struct device *dev) {
    struct my_data *data = dev_get_drvdata(dev);
    
    if (device_may_wakeup(dev)) {
        disable_irq_wake(data->irq);
    } else {
        enable_irq(data->irq);
    }
    
    return 0;
}
```

#### Step 4: Runtime PM (Auto-suspend)
For a button, "Runtime Suspend" might mean disabling the debounce clock or setting the GPIO controller to low-power mode.
```c
static int my_runtime_suspend(struct device *dev) {
    // Low power mode
    return 0;
}

static int my_runtime_resume(struct device *dev) {
    // Active mode
    return 0;
}
```

---

## 🔬 Lab Exercise: Lab 187.1 - Wake from Suspend

### 1. Lab Objectives
- Configure a GPIO button as a wakeup source.
- Suspend the system.
- Press the button to wake it.

### 2. Step-by-Step Guide
1.  **Load Driver:** `insmod my_button.ko`.
2.  **Enable Wakeup (Userspace):**
    ```bash
    echo enabled > /sys/bus/platform/devices/my_button/power/wakeup
    ```
3.  **Suspend:**
    ```bash
    echo mem > /sys/power/state
    ```
4.  **Action:** Press the button.
5.  **Result:** System should resume. `dmesg` should show "Resume called".

---

## 🧪 Additional / Advanced Labs

### Lab 2: Remote Wakeup (USB)
- **Goal:** Understand USB Remote Wakeup.
- **Task:**
    1.  Plug in a USB mouse.
    2.  `echo enabled > /sys/bus/usb/devices/.../power/wakeup`.
    3.  Suspend.
    4.  Click mouse.
    5.  System wakes.

### Lab 3: Debugging Wake Sources
- **Goal:** Identify what woke the system.
- **Task:**
    1.  `cat /sys/kernel/debug/wakeup_sources`.
    2.  Look for the "active_since" or "event_count" incrementing.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. IRQ storm on resume
*   **Cause:** Level-triggered IRQ was asserted while suspended, but handler didn't run until resume.
*   **Fix:** Ensure handler clears the interrupt immediately on resume.

#### 2. `enable_irq_wake` fails
*   **Cause:** Hardware (PIC/GIC) doesn't support waking from that specific IRQ line.

---

## ⚡ Optimization & Best Practices

### `pm_wakeup_event`
*   If an event happens (e.g., packet received) that should prevent suspend for a short time, call `pm_wakeup_event(dev, ms)`.
*   This resets the autosuspend timer and aborts any pending system suspend if using "Opportunistic Suspend" (Android style).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `device_init_wakeup`?
    *   **A:** Helper that sets the `can_wakeup` flag and optionally enables it by default. Creates the `power/wakeup` sysfs file.
2.  **Q:** Can any IRQ be a wakeup source?
    *   **A:** No. It depends on the hardware wiring. The IRQ controller must remain powered or have a special "Wakeup Logic" block.

### Challenge Task
> **Task:** "The Knock Wake".
> *   Use an accelerometer (I2C).
> *   Configure it to generate an IRQ on "Double Tap".
> *   Make that IRQ wake the system.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/pm/devices.rst](https://www.kernel.org/doc/html/latest/driver-api/pm/devices.html)

---
