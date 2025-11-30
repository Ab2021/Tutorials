# Day 184: Runtime Power Management (RPM)
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
1.  **Explain** the concept of Runtime PM (RPM).
2.  **Implement** `runtime_suspend` and `runtime_resume` callbacks.
3.  **Use** the RPM API (`pm_runtime_get_sync`, `pm_runtime_put`).
4.  **Configure** Autosuspend delay.
5.  **Debug** RPM transitions via Sysfs.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC or Dev Board.
*   **Software Required:**
    *   `powertop` (optional, for visualization).
*   **Prior Knowledge:**
    *   Day 183 (PM Basics).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Runtime PM?
*   **Granularity:** System suspend is "all or nothing". RPM allows individual components (e.g., GPU, Audio Codec, USB Hub) to sleep while the CPU is running.
*   **Reference Counting:** The kernel keeps a usage count (`usage_count`) for each device.
    *   Count > 0: Device must be Active.
    *   Count == 0: Device *can* be Suspended (after a delay).

### 🔹 Part 2: The API
*   `pm_runtime_enable(dev)`: Enable RPM for this device.
*   `pm_runtime_get_sync(dev)`: Increment count. Wake up if suspended. (Call before accessing HW).
*   `pm_runtime_put(dev)`: Decrement count. (Call after done).
*   `pm_runtime_put_autosuspend(dev)`: Decrement count, but wait `N` ms before suspending.

---

## 💻 Implementation: RPM-Aware Driver

> **Instruction:** Create a driver that "wakes up" when a file is read, and "sleeps" after 5 seconds of inactivity.

### 👨‍💻 Code Implementation

#### Step 1: Callbacks
```c
#include <linux/pm_runtime.h>

static int my_runtime_suspend(struct device *dev) {
    pr_info("MyRPM: Going to sleep (Runtime Suspend)...\n");
    // Turn off clocks, regulators
    return 0;
}

static int my_runtime_resume(struct device *dev) {
    pr_info("MyRPM: Waking up (Runtime Resume)...\n");
    // Turn on clocks, regulators
    return 0;
}

static const struct dev_pm_ops my_pm_ops = {
    .runtime_suspend = my_runtime_suspend,
    .runtime_resume  = my_runtime_resume,
};
```

#### Step 2: Probe (Initialization)
```c
static int my_probe(struct platform_device *pdev) {
    struct device *dev = &pdev->dev;
    
    // 1. Set autosuspend delay (e.g., 2000ms)
    pm_runtime_set_autosuspend_delay(dev, 2000);
    pm_runtime_use_autosuspend(dev);
    
    // 2. Enable RPM
    pm_runtime_enable(dev);
    
    pr_info("MyRPM: Probed\n");
    return 0;
}

static int my_remove(struct platform_device *pdev) {
    pm_runtime_disable(&pdev->dev);
    return 0;
}
```

#### Step 3: Usage (e.g., in Sysfs Read)
```c
static ssize_t my_show(struct device *dev, struct device_attribute *attr, char *buf) {
    int ret;
    
    // 1. Wake up / Keep awake
    ret = pm_runtime_get_sync(dev);
    if (ret < 0) {
        pm_runtime_put_noidle(dev);
        return ret;
    }
    
    // 2. Access Hardware (Simulated)
    pr_info("MyRPM: Accessing Hardware...\n");
    
    // 3. Release
    pm_runtime_mark_last_busy(dev);
    pm_runtime_put_autosuspend(dev);
    
    return sprintf(buf, "Active\n");
}
static DEVICE_ATTR_RO(my_show);
```

---

## 🔬 Lab Exercise: Lab 184.1 - Observing Transitions

### 1. Lab Objectives
- Load the driver.
- Monitor `dmesg`.
- Trigger reads and watch it wake/sleep.

### 2. Step-by-Step Guide
1.  **Load:** `insmod my_rpm.ko`.
2.  **Check Status:**
    ```bash
    cat /sys/bus/platform/devices/my_rpm_device/power/runtime_status
    # Output: suspended (initially)
    ```
3.  **Trigger Access:**
    ```bash
    cat /sys/bus/platform/devices/my_rpm_device/my_show
    ```
    *   **Log:** "Waking up...", "Accessing Hardware...".
4.  **Wait:** Wait 2 seconds.
    *   **Log:** "Going to sleep...".
5.  **Control via Sysfs:**
    *   Force On: `echo on > .../power/control`.
    *   Auto: `echo auto > .../power/control`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Nested Calls
- **Goal:** Understand refcounting.
- **Task:**
    1.  Call `pm_runtime_get_sync` twice.
    2.  Call `pm_runtime_put` once.
    3.  Verify it does *not* suspend.
    4.  Call `pm_runtime_put` again.
    5.  Verify it suspends.

### Lab 3: Universal PM
- **Goal:** Combine System Sleep and RPM.
- **Task:**
    1.  Implement `suspend` as well.
    2.  In `suspend`, check `pm_runtime_suspended(dev)`.
    3.  If already runtime suspended, do nothing (or minimal).
    4.  If active, do full suspend.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Device never sleeps
*   **Cause:** Unbalanced `get`/`put`. `usage_count` > 0.
*   **Fix:** Check `/sys/.../power/runtime_usage`.

#### 2. "Device busy"
*   **Cause:** Child devices are active. (Parents cannot sleep if children are active).

---

## ⚡ Optimization & Best Practices

### `pm_runtime_resume_and_get`
*   Newer helper that combines `get_sync` and error handling.
*   `ret = pm_runtime_resume_and_get(dev); if (ret) return ret;`

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What happens if I call `pm_runtime_get_sync` in an interrupt handler?
    *   **A:** It might sleep (if device needs time to wake). Use `pm_runtime_get_noresume` or ensure it's already active.
2.  **Q:** What is `autosuspend`?
    *   **A:** A hysteresis mechanism. Prevents rapid on/off cycling by keeping the device active for a short window after the last usage.

### Challenge Task
> **Task:** "The Efficient Logger".
> *   Create a char driver.
> *   On `open`, wake up.
> *   On `write`, log to hardware.
> *   On `close`, sleep.
> *   Use `autosuspend` so that rapid open/close doesn't toggle power constantly.

---

## 📚 Further Reading & References
- [Kernel Documentation: power/runtime_pm.rst](https://www.kernel.org/doc/html/latest/power/runtime_pm.rst)

---
