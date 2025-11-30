# Day 198: Sysfs Attributes & Binary Attributes
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
1.  **Use** `DEVICE_ATTR_RW`, `DEVICE_ATTR_RO` macros.
2.  **Organize** attributes into `attribute_group`.
3.  **Implement** Binary Attributes (`bin_attribute`) for large data.
4.  **Avoid** race conditions in Sysfs callbacks.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Day 197 (Kobjects).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Standard Attributes
*   **One Value Per File:** Sysfs files should contain a single value (string or number).
*   **Text Based:** Easy to read/write with shell scripts.
*   **Macros:**
    *   `DEVICE_ATTR_RW(name)` -> creates `dev_attr_name` with `name_show` and `name_store`.
    *   `DEVICE_ATTR_RO(name)` -> creates `dev_attr_name` with `name_show`.

### 🔹 Part 2: Attribute Groups
*   **Problem:** Calling `device_create_file` 10 times is tedious and race-prone (userspace might see the device before all files are created).
*   **Solution:** `struct attribute_group`. Pass it to `device_register` (or `sysfs_create_group`). The core creates all files atomically *before* firing the Uevent.

### 🔹 Part 3: Binary Attributes
*   **Use Case:** EEPROM dumps, Firmware images, VPD (Vital Product Data).
*   **Structure:** `struct bin_attribute`.
*   **Callbacks:** `read`, `write`, `mmap`. (Similar to file operations).

---

## 💻 Implementation: Attribute Groups

> **Instruction:** Create a platform driver with a group of attributes.

### 👨‍💻 Code Implementation

```c
#include <linux/platform_device.h>
#include <linux/sysfs.h>

static int val_a = 0;
static int val_b = 0;

// 1. Callbacks
static ssize_t a_show(struct device *dev, struct device_attribute *attr, char *buf) {
    return sprintf(buf, "%d\n", val_a);
}
static ssize_t a_store(struct device *dev, struct device_attribute *attr, const char *buf, size_t count) {
    sscanf(buf, "%d", &val_a);
    return count;
}
static DEVICE_ATTR_RW(a);

static ssize_t b_show(struct device *dev, struct device_attribute *attr, char *buf) {
    return sprintf(buf, "%d\n", val_b);
}
static DEVICE_ATTR_RO(b);

// 2. Array of Pointers
static struct attribute *my_attrs[] = {
    &dev_attr_a.attr,
    &dev_attr_b.attr,
    NULL,
};

// 3. Group
static const struct attribute_group my_group = {
    .name = "my_vars", // Optional: Creates a subdirectory
    .attrs = my_attrs,
};

static const struct attribute_group *my_groups[] = {
    &my_group,
    NULL,
};

// 4. Driver
static struct platform_driver my_driver = {
    .driver = {
        .name = "my_attr_driver",
        .dev_groups = my_groups, // Auto-created on probe
    },
};
```

---

## 💻 Implementation: Binary Attribute (EEPROM)

> **Instruction:** Simulate a 256-byte EEPROM.

### 👨‍💻 Code Implementation

```c
static char eeprom_data[256];

static ssize_t eeprom_read(struct file *filp, struct kobject *kobj,
                           struct bin_attribute *bin_attr,
                           char *buf, loff_t off, size_t count) {
    if (off >= 256) return 0;
    if (off + count > 256) count = 256 - off;
    
    memcpy(buf, eeprom_data + off, count);
    return count;
}

static ssize_t eeprom_write(struct file *filp, struct kobject *kobj,
                            struct bin_attribute *bin_attr,
                            char *buf, loff_t off, size_t count) {
    if (off >= 256) return -ENOSPC;
    if (off + count > 256) count = 256 - off;
    
    memcpy(eeprom_data + off, buf, count);
    return count;
}

static struct bin_attribute eeprom_attr = {
    .attr = { .name = "eeprom", .mode = 0664 },
    .size = 256,
    .read = eeprom_read,
    .write = eeprom_write,
};

// In probe:
sysfs_create_bin_file(&pdev->dev.kobj, &eeprom_attr);
```

---

## 🔬 Lab Exercise: Lab 198.1 - Testing Attributes

### 1. Lab Objectives
- Load the driver.
- Read/Write text attributes.
- Read/Write binary attribute using `hexdump` and `dd`.

### 2. Step-by-Step Guide
1.  **Text:**
    ```bash
    echo 123 > /sys/bus/platform/devices/.../my_vars/a
    cat /sys/bus/platform/devices/.../my_vars/a
    ```
2.  **Binary Write:**
    ```bash
    echo "Hello World" | dd of=/sys/bus/platform/devices/.../eeprom bs=1 seek=0
    ```
3.  **Binary Read:**
    ```bash
    hexdump -C /sys/bus/platform/devices/.../eeprom
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Visibility Callback
- **Goal:** Hide attributes based on condition.
- **Task:**
    1.  Implement `.is_visible` in `attribute_group`.
    2.  Return 0 (hidden) or mode (visible).
    3.  Example: Hide `debug_reg` if user is not root.

### Lab 3: Notify Userspace
- **Goal:** Trigger `poll` on sysfs file.
- **Task:**
    1.  `sysfs_notify(&dev->kobj, NULL, "a");`.
    2.  Userspace: `poll()` on the file. It returns when value changes.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Permission denied"
*   **Cause:** `.mode` is set to 0444 (Read Only) but you tried to write.
*   **Fix:** Change to 0644 or 0664.

#### 2. Race Conditions
*   **Cause:** Accessing freed memory in `show`/`store` because device was removed.
*   **Fix:** Sysfs handles this mostly (it holds a lock), but be careful with global pointers.

---

## ⚡ Optimization & Best Practices

### `PAGE_SIZE` Limit
*   Standard attributes (`show`) usually return at most `PAGE_SIZE` (4KB).
*   Do not try to return huge strings. Use binary attributes for that.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why use `dev_groups` in `struct driver` instead of `sysfs_create_group` in `probe`?
    *   **A:** `dev_groups` guarantees the files exist *before* the `ADD` uevent is sent to userspace. If you create them manually in probe, udev might run before the files are ready.
2.  **Q:** Can I use `mmap` on a binary attribute?
    *   **A:** Yes! This is how PCI BARs are exposed to userspace.

### Challenge Task
> **Task:** "The Secure Storage".
> *   Create a binary attribute "secret".
> *   Implement `.is_visible` so it only shows up if a module parameter `unlock=1` is passed.

---

## 📚 Further Reading & References
- [Kernel Documentation: filesystems/sysfs.rst](https://www.kernel.org/doc/html/latest/filesystems/sysfs.html)

---
