# Day 190: Introduction to Kernel Debugging & Tracing
## Phase 2: Linux Kernel & Device Drivers | Week 29: Kernel Debugging

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
1.  **Master** `printk` formats and log levels.
2.  **Use** Dynamic Debug (`dyndbg`) to enable logs at runtime.
3.  **Expose** internal state via `debugfs`.
4.  **Analyze** the Kernel Ring Buffer (`dmesg`).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC or Board.
*   **Software Required:**
    *   `debugfs` mounted (usually at `/sys/kernel/debug`).
*   **Prior Knowledge:**
    *   Basic Kernel Modules.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Humble `printk`
*   **Log Levels:**
    *   `KERN_EMERG` (0): System is unusable.
    *   `KERN_ALERT` (1): Action must be taken immediately.
    *   `KERN_CRIT` (2): Critical conditions.
    *   `KERN_ERR` (3): Error conditions.
    *   `KERN_WARNING` (4): Warning conditions.
    *   `KERN_NOTICE` (5): Normal but significant.
    *   `KERN_INFO` (6): Informational.
    *   `KERN_DEBUG` (7): Debug-level messages.
*   **Helpers:** `pr_info()`, `pr_err()`, `dev_info()`, `dev_err()`.
    *   *Always* use `dev_*` variants when you have a `struct device *`. It prefixes the log with the device name (e.g., `[  12.345] my_driver 1-1:1.0: Error...`).

### 🔹 Part 2: Dynamic Debug
*   **Problem:** `pr_debug()` is compiled out unless `DEBUG` is defined. Recompiling is slow.
*   **Solution:** `CONFIG_DYNAMIC_DEBUG`.
    *   `pr_debug` is compiled in but disabled by default (NOP).
    *   Can be enabled at runtime via `/sys/kernel/debug/dynamic_debug/control`.

### 🔹 Part 3: Debugfs
*   A RAM-based filesystem for debugging.
*   Unlike `sysfs` (strict one-value-per-file rule), `debugfs` has no rules. You can dump blobs, huge text files, registers, etc.

---

## 💻 Implementation: Debugfs Interface

> **Instruction:** Create a module that exposes a variable and a "reset" button via debugfs.

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <linux/debugfs.h>

static struct dentry *my_debug_dir;
static u32 my_counter = 0;

static int my_reset_set(void *data, u64 val) {
    my_counter = 0;
    return 0;
}
DEFINE_SIMPLE_ATTRIBUTE(reset_fops, NULL, my_reset_set, "%llu\n");

static int my_probe(void) {
    my_debug_dir = debugfs_create_dir("my_debug_module", NULL);
    
    // 1. Expose a u32 variable (Read/Write)
    debugfs_create_u32("counter", 0644, my_debug_dir, &my_counter);
    
    // 2. Expose a "Reset" file (Write only)
    debugfs_create_file("reset", 0200, my_debug_dir, NULL, &reset_fops);
    
    return 0;
}

static void my_remove(void) {
    debugfs_remove_recursive(my_debug_dir);
}

module_init(my_probe);
module_exit(my_remove);
MODULE_LICENSE("GPL");
```

---

## 🔬 Lab Exercise: Lab 190.1 - Dynamic Debug

### 1. Lab Objectives
- Use `pr_debug` in a module.
- Enable it at runtime without recompiling.

### 2. Step-by-Step Guide
1.  **Code:**
    ```c
    // In loop or timer
    pr_debug("MyDebug: This is a hidden message %d\n", count++);
    ```
2.  **Load:** `insmod my_debug.ko`.
3.  **Check dmesg:** No output.
4.  **Enable:**
    ```bash
    echo 'module my_debug +p' > /sys/kernel/debug/dynamic_debug/control
    ```
5.  **Check dmesg:** Output appears!
6.  **Disable:**
    ```bash
    echo 'module my_debug -p' > /sys/kernel/debug/dynamic_debug/control
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Debugfs Blob
- **Goal:** Dump a binary structure (e.g., registers).
- **Task:**
    1.  Define a struct `regs { u32 r1; u32 r2; }`.
    2.  Use `debugfs_create_blob("regs", 0444, dir, &wrapper)`.
    3.  Read with `hexdump -C /sys/kernel/debug/.../regs`.

### Lab 3: Rate Limiting
- **Goal:** Prevent log flooding.
- **Task:**
    1.  Use `pr_info_ratelimited("Error!\n");` in a tight loop.
    2.  Verify it only prints once every few seconds (default configuration).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Debugfs not found
*   **Cause:** Not mounted.
*   **Fix:** `mount -t debugfs none /sys/kernel/debug`.

#### 2. `pr_debug` still not showing
*   **Cause:** `CONFIG_DYNAMIC_DEBUG` disabled in kernel config.
*   **Fix:** Recompile kernel or use `#define DEBUG` at the top of your C file (before includes).

---

## ⚡ Optimization & Best Practices

### `dev_dbg` vs `pr_debug`
*   Always use `dev_dbg` if possible. Dynamic Debug can filter by device name, not just module name.
*   Example: `echo 'device 1-1.2 +p' > control`. Useful if you have 10 identical USB devices and only want to debug one.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `sysfs` and `debugfs`?
    *   **A:** `sysfs` is part of the ABI (stable interface), strictly structured (one value per file). `debugfs` is unstable, for developers only, and allows arbitrary data.
2.  **Q:** How do I change the console log level at runtime?
    *   **A:** `echo "8" > /proc/sys/kernel/printk` (to see everything).

### Challenge Task
> **Task:** "The Hex Dumper".
> *   Write a function `print_hex_dump_debug` wrapper.
> *   It should take a buffer and length.
> *   It should use `print_hex_dump` but only if Dynamic Debug is enabled for that call site.

---

## 📚 Further Reading & References
- [Kernel Documentation: admin-guide/dynamic-debug-howto.rst](https://www.kernel.org/doc/html/latest/admin-guide/dynamic-debug-howto.html)

---
