# Day 199: Uevents and Udev
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
1.  **Explain** the Uevent mechanism (Netlink multicast).
2.  **Monitor** Uevents using `udevadm monitor`.
3.  **Send** custom Uevents from kernel space (`kobject_uevent`).
4.  **Write** Udev rules to create symlinks and run scripts.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   `udev` (systemd-udevd).
*   **Prior Knowledge:**
    *   Day 197 (Kobjects).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Notification Chain
1.  **Kernel:** A device is added (or changed).
2.  **Kobject:** `kobject_uevent()` is called.
3.  **Netlink:** The kernel formats a message (ACTION=add, DEVPATH=/devices/...) and broadcasts it on the `NETLINK_KOBJECT_UEVENT` socket.
4.  **Userspace (Udevd):** Listens on the socket.
5.  **Rules:** Udevd matches the message against `/etc/udev/rules.d/*.rules`.
6.  **Action:** Creates `/dev` nodes, symlinks, or runs scripts.

### 🔹 Part 2: Environment Variables
Uevents carry payload in the form of KEY=VALUE pairs:
*   `ACTION`: add, remove, change, online, offline.
*   `DEVPATH`: Path in sysfs.
*   `SUBSYSTEM`: block, net, usb, etc.
*   `MODALIAS`: String for module loading.

---

## 💻 Implementation: Sending Custom Uevents

> **Instruction:** Create a module that sends a "CHANGE" event with custom data when a sysfs file is written.

### 👨‍💻 Code Implementation

```c
#include <linux/kobject.h>

static struct kobject *my_kobj;

static ssize_t trigger_store(struct kobject *kobj, struct kobj_attribute *attr,
                             const char *buf, size_t count) {
    char *envp[] = {
        "MY_VAR=Hello",
        "STATUS=Critical",
        NULL
    };
    
    // Send "CHANGE" event
    kobject_uevent_env(kobj, KOBJ_CHANGE, envp);
    
    return count;
}

static struct kobj_attribute trigger_attr = __ATTR(trigger, 0200, NULL, trigger_store);

// ... init/exit code to create kobject and file ...
```

---

## 🔬 Lab Exercise: Lab 199.1 - Monitoring Uevents

### 1. Lab Objectives
- Use `udevadm` to see what the kernel is saying.

### 2. Step-by-Step Guide
1.  **Start Monitor:**
    ```bash
    udevadm monitor --kernel --property
    ```
2.  **Trigger Event:**
    *   Plug in a USB stick.
    *   Or write to the module from above: `echo 1 > /sys/kernel/my_kobject/trigger`.
3.  **Observe Output:**
    ```
    KERNEL[1234.56] change   /kernel/my_kobject (module)
    ACTION=change
    DEVPATH=/kernel/my_kobject
    SUBSYSTEM=module
    MY_VAR=Hello
    STATUS=Critical
    ```

---

## 🔬 Lab Exercise: Lab 199.2 - Writing Udev Rules

### 1. Lab Objectives
- Create a rule that runs a script when our custom event fires.

### 2. Step-by-Step Guide
1.  **Create Script:** `/usr/local/bin/my_handler.sh`
    ```bash
    #!/bin/bash
    echo "Received event from $DEVPATH with Status $STATUS" >> /tmp/udev_log.txt
    ```
    *   `chmod +x` it.
2.  **Create Rule:** `/etc/udev/rules.d/99-my-test.rules`
    ```
    SUBSYSTEM=="module", ENV{STATUS}=="Critical", RUN+="/usr/local/bin/my_handler.sh"
    ```
3.  **Reload:**
    ```bash
    udevadm control --reload
    ```
4.  **Trigger:**
    ```bash
    echo 1 > /sys/kernel/my_kobject/trigger
    ```
5.  **Verify:** Check `/tmp/udev_log.txt`.

---

## 🧪 Additional / Advanced Labs

### Lab 3: Persistent Symlinks
- **Goal:** Create a stable name for a USB device.
- **Task:**
    1.  Plug in a USB-Serial adapter.
    2.  Find attributes: `udevadm info -a -n /dev/ttyUSB0`.
    3.  Create rule:
        ```
        SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", SYMLINK+="my_serial_port"
        ```
    4.  Reload and replug.
    5.  `ls -l /dev/my_serial_port`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Script not running
*   **Cause:** Udev kills long-running scripts. Or `PATH` is different.
*   **Fix:** Use full paths. Don't sleep in the script (spawn a background job if needed). Check `journalctl -u systemd-udevd`.

#### 2. Rule not matching
*   **Cause:** Wrong syntax (== vs =).
*   **Fix:** Use `udevadm test /sys/kernel/my_kobject` to simulate and see which rules apply.

---

## ⚡ Optimization & Best Practices

### `TAG+="systemd"`
*   If you want your device to be visible as a systemd device unit (so services can depend on it), add `TAG+="systemd"`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `RUN` and `PROGRAM` in udev rules?
    *   **A:** `PROGRAM` runs a command to *generate* a value for matching (e.g., check serial number). `RUN` executes a command *after* the node is created (side effect).
2.  **Q:** Why does `kobject_uevent` use Netlink?
    *   **A:** It's a reliable, multicast IPC mechanism. Unlike `call_usermodehelper`, it doesn't fork a process for every event (which would be slow and memory intensive).

### Challenge Task
> **Task:** "The Auto-Mounter".
> *   Write a udev rule that detects when a specific USB drive (by Serial Number) is plugged in.
> *   Run a script to mount it to `/mnt/secure`.
> *   (Note: Modern desktops use Udisks, but doing it manually is a great exercise).

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/usb/URB.rst](https://www.kernel.org/doc/html/latest/driver-api/usb/URB.html) (Wait, wrong link. Should be kobject docs).
- [Udev Man Page](https://www.freedesktop.org/software/systemd/man/udev.html)

---
