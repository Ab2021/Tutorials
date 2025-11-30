# Day 185: System Sleep States (S2RAM, S2Disk)
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
1.  **Distinguish** between S0ix (Freeze), S3 (Mem), and S4 (Disk).
2.  **Implement** `freeze`, `thaw`, `poweroff`, and `restore` callbacks.
3.  **Understand** the Hibernation workflow (Snapshot -> Save -> Power Off -> Boot -> Restore).
4.  **Debug** resume failures using `pm_test`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC (Virtual Machine might not support all states).
*   **Software Required:**
    *   `swsusp` or `systemd-suspend`.
*   **Prior Knowledge:**
    *   Day 183 (PM Basics).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Sleep Spectrum
*   **S0 (Active):** Running.
*   **S0ix (Suspend-to-Idle / Modern Standby):** CPU in low power, RAM active. Fast wake.
*   **S3 (Suspend-to-RAM):** CPU off, RAM self-refresh. Slower wake.
*   **S4 (Hibernation / Suspend-to-Disk):** RAM image written to swap. Zero power. Slowest wake.

### 🔹 Part 2: The Callback Matrix
Different events trigger different callbacks in `struct dev_pm_ops`:

| Event | Callbacks Invoked |
| :--- | :--- |
| **Suspend (S3)** | `suspend` -> `resume` |
| **Freeze (S0ix)** | `freeze` -> `thaw` |
| **Hibernate (S4)** | `freeze` -> `thaw` (Create Image) -> `poweroff` -> `restore` (On Boot) |

*   **Note:** Hibernation is complex. It freezes tasks, creates an image (atomic snapshot), thaws tasks to write the image, then powers off. On boot, it loads the image and calls `restore`.

---

## 💻 Implementation: Handling Hibernation

> **Instruction:** Extend our PM driver to handle Hibernation explicitly.

### 👨‍💻 Code Implementation

```c
static int my_freeze(struct device *dev) {
    pr_info("MyPM: Freeze (Quiescing hardware for snapshot)...\n");
    // Stop DMA, Interrupts. DO NOT power down device yet.
    return 0;
}

static int my_thaw(struct device *dev) {
    pr_info("MyPM: Thaw (Snapshot done, or failed)...\n");
    // Restart DMA/Interrupts.
    return 0;
}

static int my_poweroff(struct device *dev) {
    pr_info("MyPM: Poweroff (Writing image done, shutting down)...\n");
    // Power down device.
    return 0;
}

static int my_restore(struct device *dev) {
    pr_info("MyPM: Restore (Booted from image)...\n");
    // Reset device, load state from saved memory.
    return 0;
}

static const struct dev_pm_ops my_pm_ops = {
    .suspend = my_suspend,
    .resume  = my_resume,
    .freeze  = my_freeze,
    .thaw    = my_thaw,
    .poweroff = my_poweroff,
    .restore  = my_restore,
};
```

---

## 🔬 Lab Exercise: Lab 185.1 - Testing Hibernation

### 1. Lab Objectives
- Trigger Hibernation (if supported).
- Observe the sequence of callbacks.

### 2. Step-by-Step Guide
1.  **Check Support:** `cat /sys/power/state`. Should see `disk`.
2.  **Load Driver:** `insmod my_pm.ko`.
3.  **Trigger Disk:**
    ```bash
    echo disk > /sys/power/state
    ```
4.  **Observation:**
    *   System will save image to swap and turn off.
5.  **Wake:**
    *   Power on.
    *   Kernel detects image and restores.
6.  **Check Log:**
    *   You should see: `freeze` -> `thaw` (Snapshot creation) -> `poweroff` (Final shutdown).
    *   On boot: `restore`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: PM Test Modes
- **Goal:** Debug suspend without actually sleeping (saves time/reboots).
- **Task:**
    1.  `cat /sys/power/pm_test`. (Shows: `[none] core processors platform devices freez`).
    2.  `echo devices > /sys/power/pm_test`.
    3.  `echo mem > /sys/power/state`.
    4.  **Result:** Kernel freezes tasks, suspends devices, waits 5 seconds, resumes devices, thaws tasks. Screen never goes black.
    5.  Great for testing driver suspend/resume logic quickly.

### Lab 3: Wake-on-LAN (WoL)
- **Goal:** Configure a device to wake the system.
- **Task:**
    1.  Use `ethtool -s eth0 wol g`.
    2.  Check `/proc/acpi/wakeup`.
    3.  Suspend.
    4.  Send Magic Packet from another PC.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Hibernate fails with "Not enough swap"
*   **Cause:** Swap partition is smaller than RAM.
*   **Fix:** Increase swap or use swap file.

#### 2. Driver crashes during `restore`
*   **Cause:** Assuming hardware is in a specific state.
*   **Fix:** In `restore`, treat the hardware as if it was just probed (reset it fully).

---

## ⚡ Optimization & Best Practices

### `SIMPLE_DEV_PM_OPS`
*   Macro to simplify assignment if `suspend` == `freeze` == `poweroff`.
*   `SIMPLE_DEV_PM_OPS(name, suspend_fn, resume_fn)` maps `suspend` to `freeze` and `poweroff`, and `resume` to `thaw` and `restore`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why does Hibernation call `thaw` after `freeze`?
    *   **A:** To write the memory image to disk, the kernel needs drivers (Disk, Filesystem) to be active. So it freezes to take a snapshot, then thaws to write it, then powers off.
2.  **Q:** What is `pm_test`?
    *   **A:** A debug facility to test suspend transitions up to a certain point (e.g., `devices`) and then automatically resume, avoiding a full power cycle.

### Challenge Task
> **Task:** "The Time Traveler".
> *   Store the system time (`ktime_get`) in a variable during `suspend`.
> *   In `resume`, calculate how long the system was asleep.
> *   Print "I slept for X seconds".

---

## 📚 Further Reading & References
- [Kernel Documentation: power/basic-pm-debugging.rst](https://www.kernel.org/doc/html/latest/power/basic-pm-debugging.html)

---
