# Day 183: Linux Power Management Overview
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
1.  **Explain** the Linux Power Management (PM) architecture.
2.  **Differentiate** between System Sleep (Suspend) and Runtime PM.
3.  **Understand** the ACPI vs Device Tree PM models.
4.  **Identify** the key kernel structures (`struct dev_pm_ops`).
5.  **Control** system state via `/sys/power/state`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC or Dev Board (Raspberry Pi/BeagleBone).
*   **Software Required:**
    *   `pm-utils` (optional).
*   **Prior Knowledge:**
    *   Basic Driver Model (Probe/Remove).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Goal of Power Management
*   **Energy Efficiency:** Critical for battery-powered devices (Phones, IoT).
*   **Thermal Management:** Reducing power reduces heat.
*   **Noise Reduction:** Fan control.

### 🔹 Part 2: Types of Power Management
1.  **System-Wide Suspend (Static):**
    *   The whole system goes to sleep.
    *   **Freeze (S0ix):** Idle, everything stopped.
    *   **Standby (S1):** CPU halted.
    *   **Suspend-to-RAM (S3/STR):** RAM self-refresh, CPU off.
    *   **Suspend-to-Disk (S4/Hibernate):** RAM saved to disk, power off.
2.  **Runtime Power Management (Dynamic):**
    *   Individual devices power down when idle *while the system is running*.
    *   Example: USB Camera turns off when not recording.

### 🔹 Part 3: The Kernel Architecture
*   **PM Core:** Orchestrates the transition.
*   **Bus Subsystem:** (PCI, I2C, Platform) handles bus-specific logic.
*   **Device Driver:** Saves/Restores device registers.

#### The `dev_pm_ops` Structure
```c
struct dev_pm_ops {
    int (*suspend)(struct device *dev);
    int (*resume)(struct device *dev);
    int (*freeze)(struct device *dev);
    int (*thaw)(struct device *dev);
    int (*poweroff)(struct device *dev);
    int (*restore)(struct device *dev);
    int (*runtime_suspend)(struct device *dev);
    int (*runtime_resume)(struct device *dev);
    // ...
};
```

---

## 💻 Implementation: A Simple PM-Aware Driver

> **Instruction:** We will modify a "Dummy" Platform Driver to handle Suspend and Resume events.

### 👨‍💻 Code Implementation

#### Step 1: The Driver Structure
```c
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/pm.h>

static int my_probe(struct platform_device *pdev) {
    pr_info("MyPM: Device Probed\n");
    return 0;
}

static int my_remove(struct platform_device *pdev) {
    pr_info("MyPM: Device Removed\n");
    return 0;
}
```

#### Step 2: Suspend and Resume Callbacks
```c
static int my_suspend(struct device *dev) {
    pr_info("MyPM: Suspend called. Saving state...\n");
    // 1. Stop hardware (disable interrupts, DMA)
    // 2. Save registers to memory
    return 0;
}

static int my_resume(struct device *dev) {
    pr_info("MyPM: Resume called. Restoring state...\n");
    // 1. Restore registers
    // 2. Restart hardware
    return 0;
}

// Macro to populate the structure automatically
static const struct dev_pm_ops my_pm_ops = {
    .suspend = my_suspend,
    .resume  = my_resume,
};
```

#### Step 3: Registration
```c
static struct platform_driver my_driver = {
    .probe = my_probe,
    .remove = my_remove,
    .driver = {
        .name = "my_pm_driver",
        .pm = &my_pm_ops, // Link PM ops here
    },
};

module_platform_driver(my_driver);
```

---

## 🔬 Lab Exercise: Lab 183.1 - Triggering Suspend

### 1. Lab Objectives
- Load the driver.
- Trigger a system suspend (Suspend-to-RAM or Freeze).
- Verify the callbacks are called.

### 2. Step-by-Step Guide
1.  **Load:** `insmod my_pm.ko`.
2.  **Create Device:** (If not using DT)
    *   You might need a small helper module to register a `platform_device` named "my_pm_driver".
    *   *Or just use `sysfs` to bind if possible, but platform devices usually need registration.*
    *   *Alternative:* Use `dummy_driver` approach where you register the device in `init`.
3.  **Trigger Suspend:**
    *   **Warning:** This will suspend your machine! If accessing via SSH, you will lose connection until you wake it (Power button/Keyboard).
    *   `echo freeze > /sys/power/state` (Safer, just idles).
    *   `echo mem > /sys/power/state` (Deep sleep).
4.  **Wake Up:** Press Power button or Key.
5.  **Check Log:** `dmesg | grep MyPM`.
    *   Should see "Suspend called" then "Resume called".

---

## 🧪 Additional / Advanced Labs

### Lab 2: Saving State
- **Goal:** Simulate register saving.
- **Task:**
    1.  Add a global variable `int counter = 0`.
    2.  In `probe`, start a timer that increments it.
    3.  In `suspend`, stop the timer and print `counter`.
    4.  In `resume`, restart the timer.
    5.  Verify `counter` didn't increment while suspended.

### Lab 3: Preventing Suspend
- **Goal:** Abort suspend.
- **Task:**
    1.  In `my_suspend`, return `-EBUSY`.
    2.  Try to suspend.
    3.  System should fail to suspend and log the error.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. System doesn't wake up
*   **Cause:** Driver crashed during resume (e.g., accessing hardware before clock is on).
*   **Fix:** Use `no_console_suspend` kernel parameter to see logs during suspend.

#### 2. Callbacks not called
*   **Cause:** `CONFIG_PM` or `CONFIG_PM_SLEEP` not enabled.
*   **Fix:** Check `.config`.

---

## ⚡ Optimization & Best Practices

### `SET_SYSTEM_SLEEP_PM_OPS`
*   Use macros like `SET_SYSTEM_SLEEP_PM_OPS(suspend_fn, resume_fn)` to handle `#ifdef CONFIG_PM_SLEEP` automatically.
*   Prevents "unused function" warnings if PM is disabled.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `suspend` and `runtime_suspend`?
    *   **A:** `suspend` is for system-wide sleep (user requested). `runtime_suspend` is for auto-suspend when idle (OS managed).
2.  **Q:** Why do we need `freeze`?
    *   **A:** For "Suspend-to-Idle" (S0ix). It's faster than S3 but saves less power. Used heavily in modern laptops/phones.

### Challenge Task
> **Task:** "The Insomniac Driver".
> *   Create a driver that refuses to suspend if a specific file `/sys/kernel/insomniac` contains "1".
> *   Otherwise, it allows suspend.

---

## 📚 Further Reading & References
- [Kernel Documentation: power/interface.rst](https://www.kernel.org/doc/html/latest/power/interface.html)
- [Kernel Documentation: driver-api/pm/devices.rst](https://www.kernel.org/doc/html/latest/driver-api/pm/devices.html)

---
