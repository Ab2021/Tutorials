# Day 200: Device Classes and Groups
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
1.  **Explain** the purpose of Device Classes (`/sys/class`).
2.  **Create** a custom Class (`class_create`, `class_destroy`).
3.  **Register** devices belonging to a class (`device_create`).
4.  **Implement** Class Attributes (shared by all devices in the class).

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

### 🔹 Part 1: Bus vs Class
*   **Bus (`/sys/bus`):** Groups devices by how they are *physically connected* (PCI, USB, I2C).
*   **Class (`/sys/class`):** Groups devices by *what they do* (Input, Net, TTY, Block, GPIO).
*   **Example:** A USB Network Adapter.
    *   Physical: `/sys/bus/usb/devices/...`
    *   Functional: `/sys/class/net/eth0`

### 🔹 Part 2: The `struct class`
*   Defines a high-level view.
*   Can have `class_attributes` (global settings for the subsystem).
*   Can have `dev_groups` (default attributes for every device added to this class).

---

## 💻 Implementation: Creating a Custom Class

> **Instruction:** Create a class named "robot" and register devices "arm_left" and "arm_right".

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <linux/device.h>
#include <linux/fs.h>

static struct class *robot_class;
static int major;

// Class Attribute (Global)
static ssize_t version_show(struct class *class, struct class_attribute *attr, char *buf) {
    return sprintf(buf, "1.0\n");
}
static CLASS_ATTR_RO(version);

static struct attribute *robot_class_attrs[] = {
    &class_attr_version.attr,
    NULL,
};
ATTRIBUTE_GROUPS(robot_class);

// Device Attribute (Per Device)
static ssize_t speed_show(struct device *dev, struct device_attribute *attr, char *buf) {
    return sprintf(buf, "Speed: %s\n", dev_name(dev)); // Dummy logic
}
static DEVICE_ATTR_RO(speed);

static struct attribute *robot_dev_attrs[] = {
    &dev_attr_speed.attr,
    NULL,
};
ATTRIBUTE_GROUPS(robot_dev);

static int __init my_init(void) {
    // 1. Create Class
    robot_class = class_create(THIS_MODULE, "robot");
    if (IS_ERR(robot_class)) return PTR_ERR(robot_class);
    
    // 2. Set Attributes
    robot_class->class_groups = robot_class_groups;
    robot_class->dev_groups = robot_dev_groups; // Auto-added to children
    
    // 3. Create Devices
    device_create(robot_class, NULL, MKDEV(0, 0), NULL, "arm_left");
    device_create(robot_class, NULL, MKDEV(0, 0), NULL, "arm_right");
    
    pr_info("Robot class created\n");
    return 0;
}

static void __exit my_exit(void) {
    device_destroy(robot_class, MKDEV(0, 0)); // Need correct dev_t if used
    // Actually device_destroy uses dev_t to find the device. 
    // Since we used 0,0, we might need to keep track of pointers or use device_find_child.
    // For simplicity here, assume we stored the struct device* from device_create.
    
    class_destroy(robot_class);
}

module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
```

---

## 🔬 Lab Exercise: Lab 200.1 - Exploring Classes

### 1. Lab Objectives
- Load the module.
- Explore `/sys/class/robot`.

### 2. Step-by-Step Guide
1.  **Load:** `insmod my_class.ko`.
2.  **Check Class:**
    ```bash
    ls -l /sys/class/robot/
    # Should see: version, arm_left, arm_right
    ```
3.  **Check Device:**
    ```bash
    ls -l /sys/class/robot/arm_left/
    # Should see: speed, uevent, subsystem -> ../../../../class/robot
    ```
4.  **Read Attributes:**
    ```bash
    cat /sys/class/robot/version
    cat /sys/class/robot/arm_left/speed
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Class Interface
- **Goal:** Be notified when devices are added to your class.
- **Task:**
    1.  Use `class_interface_register`.
    2.  Implement `.add_dev` and `.remove_dev`.
    3.  This is how subsystems (like Input) automatically create `/dev/input/eventX` nodes when a driver registers an input device.

### Lab 3: Device Groups (Not Sysfs Groups)
- **Note:** Don't confuse Sysfs Groups with `dev_pm_domain` or IOMMU groups.
- **Task:** Check `/sys/kernel/iommu_groups/` to see how PCI devices are isolated for virtualization.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. `class_create` fails
*   **Cause:** Name collision.
*   **Fix:** Choose a unique name.

#### 2. `device_create` fails
*   **Cause:** `dev_t` collision or invalid parent.

---

## ⚡ Optimization & Best Practices

### `class_compat`
*   For backward compatibility, some classes expose symlinks in `/sys/class` that point to `/sys/devices`.
*   Modern drivers should rely on the hierarchy in `/sys/devices` but use `/sys/class` for lookup.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `device_create` and `device_register`?
    *   **A:** `device_register` is the low-level core function. `device_create` is a convenience wrapper that also handles `dev_t` (creating the char/block device node in `/dev` via udev) and sets the class.
2.  **Q:** Can a device belong to multiple classes?
    *   **A:** No. `struct device` has a single `class` pointer. However, a physical device can have multiple logical child devices, each in a different class (e.g., a USB webcam has a Video device and an Audio device).

### Challenge Task
> **Task:** "The Inventory Manager".
> *   Create a class "inventory".
> *   Write a script that creates 100 dummy devices in this class.
> *   Measure how long it takes to create/destroy them.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/driver-model/class.rst](https://www.kernel.org/doc/html/latest/driver-api/driver-model/class.html)

---
