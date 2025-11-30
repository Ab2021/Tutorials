# Day 189: Week 28 Review and Project - The Smart Power Manager
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
1.  **Synthesize** Week 28 concepts (System Sleep, Runtime PM, QoS, GenPD).
2.  **Architect** a power-efficient subsystem.
3.  **Implement** a driver that aggressively manages power.
4.  **Verify** power savings using `powertop` or logs.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC or Board.
*   **Software Required:**
    *   `powertop`.
*   **Prior Knowledge:**
    *   Week 28 Content.

---

## 🔄 Week 28 Review

### 1. PM Basics (Day 183)
*   `dev_pm_ops`.
*   Suspend/Resume.

### 2. Runtime PM (Day 184)
*   `pm_runtime_get/put`.
*   Autosuspend.

### 3. Sleep States (Day 185)
*   S0ix vs S3 vs S4.
*   `freeze`, `thaw`, `restore`.

### 4. Domains & QoS (Day 186)
*   GenPD.
*   PM QoS Constraints.

### 5. Implementation (Day 187-188)
*   Wakeup sources.
*   Debugging with `no_console_suspend`.

---

## 🛠️ Project: The "Smart Power Manager"

### 📋 Project Requirements
1.  **Driver:** `smart_pm.ko`.
2.  **Devices:**
    *   Creates 3 virtual devices: `sensor_hub`, `display`, `modem`.
3.  **Behavior:**
    *   **Sensor Hub:** Always active (Runtime PM disabled), but allows System Suspend.
    *   **Display:** Aggressive Runtime PM (Autosuspend 100ms).
    *   **Modem:** High Latency allowed (QoS).
4.  **Interaction:**
    *   Expose a Sysfs file `/sys/class/smart_pm/mode`.
    *   Writing "performance" disables Runtime PM for all.
    *   Writing "powersave" enables it.

---

## 💻 Implementation: Step-by-Step Guide

### 🔹 Phase 1: The Device Hierarchy
Create a platform driver that registers 3 platform devices in `init`.

```c
struct platform_device *pdev_sensor, *pdev_display, *pdev_modem;

// In init:
pdev_sensor = platform_device_register_simple("smart_sensor", -1, NULL, 0);
// ...
```

### 🔹 Phase 2: The Driver Logic
Implement `probe` and `dev_pm_ops`.

```c
static int smart_probe(struct platform_device *pdev) {
    if (strcmp(pdev->name, "smart_display") == 0) {
        pm_runtime_set_autosuspend_delay(&pdev->dev, 100);
        pm_runtime_use_autosuspend(&pdev->dev);
        pm_runtime_enable(&pdev->dev);
    }
    // ...
    return 0;
}
```

### 🔹 Phase 3: The Manager Interface
Create a separate class or sysfs entry to control the policy.

```c
static ssize_t mode_store(struct class *class, struct class_attribute *attr,
                          const char *buf, size_t count) {
    if (sysfs_streq(buf, "performance")) {
        // Forbid sleep
        pm_runtime_forbid(&pdev_display->dev);
    } else if (sysfs_streq(buf, "powersave")) {
        // Allow sleep
        pm_runtime_allow(&pdev_display->dev);
    }
    return count;
}
```

### 🔹 Phase 4: Testing
1.  **Load:** `insmod smart_pm.ko`.
2.  **Check Display:**
    *   Wait 100ms.
    *   Check status: `cat /sys/bus/platform/devices/smart_display/power/runtime_status`.
    *   Should be "suspended".
3.  **Set Performance:**
    *   `echo performance > /sys/class/smart_pm/mode`.
    *   Check status. Should be "active".

---

## 📈 Grading Rubric

| Criteria | Excellent (A) | Good (B) | Needs Improvement (C) |
| :--- | :--- | :--- | :--- |
| **Functionality** | All devices behave as specified. | Some devices don't suspend. | Crash on load. |
| **Control** | Mode switch works instantly. | Mode switch requires manual trigger. | No control. |
| **Code Quality** | Clean, modular, commented. | Spaghetti code. | No comments. |

---

## 🔮 Looking Ahead: Week 29
Next week, we dive into **Kernel Debugging**.
You will learn how to trace function calls, analyze performance bottlenecks, and debug crashes using Ftrace, Perf, and KGDB.

---
