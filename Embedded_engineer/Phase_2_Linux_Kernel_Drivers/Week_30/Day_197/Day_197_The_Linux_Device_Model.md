# Day 197: The Linux Device Model (Kobjects, Ksets)
## Phase 2: Linux Kernel & Device Drivers | Week 30: Device Model & Sysfs

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
1.  **Explain** the role of `kobject` as the base class of kernel objects.
2.  **Understand** the relationship between `kobject`, `kset`, and `ktype`.
3.  **Visualize** how the Device Model creates the `/sys` filesystem.
4.  **Implement** a raw `kobject` module to create custom directories in Sysfs.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Basic Sysfs usage.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "Object Oriented" Kernel
*   **Problem:** The kernel has many types of objects (Devices, Drivers, Buses, Classes) that share common features:
    *   Reference Counting (`kref`).
    *   Sysfs representation (Directory).
    *   Hotplug events (Uevents).
*   **Solution:** `struct kobject`. It is the "Base Class" embedded in other structures (like `struct device`, `struct cdev`).

### 🔹 Part 2: The Hierarchy
*   **Kobject:** A directory in Sysfs.
*   **Kset:** A collection of Kobjects (a subdirectory). It handles events for the group.
*   **Ktype:** Defines the default attributes (files) and destructor for a Kobject.

### 🔹 Part 3: Sysfs Mapping
*   `/sys/` is literally a visualization of the Kobject tree.
*   `/sys/devices/` -> A Kset.
*   `/sys/devices/pci0000:00/` -> A Kobject (embedded in `struct device`).

---

## 💻 Implementation: Raw Kobject

> **Instruction:** Create a module that creates `/sys/kernel/my_kobject/` and a file inside it.

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <linux/kobject.h>
#include <linux/sysfs.h>

static struct kobject *my_kobj;
static int my_value = 0;

// 1. Show Function (Read)
static ssize_t my_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
    return sprintf(buf, "%d\n", my_value);
}

// 2. Store Function (Write)
static ssize_t my_store(struct kobject *kobj, struct kobj_attribute *attr,
                        const char *buf, size_t count) {
    sscanf(buf, "%d", &my_value);
    return count;
}

// 3. Attribute Definition
static struct kobj_attribute my_attribute =
    __ATTR(my_value, 0664, my_show, my_store);

static int __init my_init(void) {
    // Create kobject named "my_kobject" under /sys/kernel/
    my_kobj = kobject_create_and_add("my_kobject", kernel_kobj);
    if (!my_kobj) return -ENOMEM;

    // Create file
    int ret = sysfs_create_file(my_kobj, &my_attribute.attr);
    if (ret) kobject_put(my_kobj);

    return ret;
}

static void __exit my_exit(void) {
    kobject_put(my_kobj); // Decrement refcount. If 0, it is freed.
}

module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
```

---

## 🔬 Lab Exercise: Lab 197.1 - Kobject Hierarchy

### 1. Lab Objectives
- Create a nested hierarchy: `/sys/kernel/my_parent/my_child/`.

### 2. Step-by-Step Guide
1.  **Create Parent:**
    ```c
    parent = kobject_create_and_add("my_parent", kernel_kobj);
    ```
2.  **Create Child:**
    ```c
    child = kobject_create_and_add("my_child", parent);
    ```
3.  **Verify:**
    ```bash
    tree /sys/kernel/my_parent
    ```
4.  **Unload:**
    *   Call `kobject_put(child)` then `kobject_put(parent)`.
    *   **Order matters!** If you free parent first, child becomes an orphan (or crashes if accessing parent).

---

## 🧪 Additional / Advanced Labs

### Lab 2: Reference Counting
- **Goal:** Prove that `kobject` stays alive as long as refcount > 0.
- **Task:**
    1.  In `my_show`, call `kobject_get(kobj)`.
    2.  Try to `rmmod`. It should hang or fail (because module holds the code for the destructor).
    3.  (Actually, `rmmod` checks module refcount, not kobject refcount. But if you `kobject_put` too many times, you get Use-After-Free).

### Lab 3: Custom Ktype
- **Goal:** Use `kzalloc` + `kobject_init`.
- **Task:**
    1.  Don't use `kobject_create_and_add`.
    2.  Allocate your own struct.
    3.  Initialize it with a custom `ktype` that has a `.release` function.
    4.  In `.release`, `kfree` the struct.
    5.  This is how real drivers work (embedding kobj).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Duplicate name" error
*   **Cause:** Creating a kobject with a name that already exists in that directory.
*   **Fix:** Check return value of `kobject_create_and_add`.

#### 2. Memory Leak
*   **Cause:** Forgetting `kobject_put`.
*   **Fix:** `kmemleak` tool.

---

## ⚡ Optimization & Best Practices

### `struct device`
*   In 99% of cases, you should use `struct device` (which wraps `kobject`) instead of raw `kobject`.
*   `device_register` handles the kobject logic, symlinks, and uevents for you.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What happens when `kobject_put` reduces refcount to 0?
    *   **A:** The `ktype->release` callback is invoked to free the memory.
2.  **Q:** Why is `kernel_kobj` used as a parent?
    *   **A:** It corresponds to `/sys/kernel/`. It's a convenient place for miscellaneous kernel modules to expose data.

### Challenge Task
> **Task:** "The Kset Explorer".
> *   Write a module that iterates over `module_kset` (which lists all loaded modules).
> *   Print the name of every module loaded in the system by walking the kset list.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/driver-model/overview.rst](https://www.kernel.org/doc/html/latest/driver-api/driver-model/overview.html)
- [LWN: The Zen of Kobjects](https://lwn.net/Articles/51437/)

---
