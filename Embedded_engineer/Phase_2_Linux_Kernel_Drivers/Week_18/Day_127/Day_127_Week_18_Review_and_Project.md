# Day 127: Week 18 Review & Project - The Minimalist Linux System
## Phase 2: Linux Kernel & Device Drivers | Week 18: Linux Kernel Fundamentals

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
1.  **Synthesize** all Week 18 concepts (Kernel Build, Boot Process, RootFS, DTS, Modules).
2.  **Architect** a complete, minimal Embedded Linux system from source.
3.  **Develop** a "System Info" Kernel Module that exposes internal kernel data to userspace.
4.  **Debug** boot failures and integration issues in a full-stack environment.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   All tools from Days 121-126.
*   **Prior Knowledge:**
    *   Week 18 Content.

---

## 🔄 Week 18 Review

### 1. The Linux Kernel (Days 121-122)
*   **Monolithic:** Drivers inside the kernel (mostly).
*   **Kbuild:** `make menuconfig` -> `.config` -> `vmlinux` -> `Image`.
*   **Cross-Compilation:** Building on x86 for ARM64.

### 2. The Boot Process (Day 123)
*   **Bootloader** loads Kernel + DTB + Initramfs.
*   **Kernel** initializes hardware, mounts RootFS.
*   **Init** (PID 1) starts userspace.

### 3. Filesystems (Day 124)
*   **RootFS:** The directory hierarchy (`/bin`, `/etc`, `/lib`).
*   **Initramfs:** Temporary RAM disk for early boot.
*   **Ext4:** Persistent storage.

### 4. Device Tree (Day 125)
*   **DTS:** Hardware description language.
*   **DTB:** Binary blob passed to kernel.
*   **Compatible String:** Binds hardware nodes to drivers.

### 5. Kernel Modules (Day 126)
*   **LKM:** Code loaded at runtime (`.ko`).
*   **Macros:** `module_init`, `module_exit`.
*   **License:** GPL vs Proprietary.

---

## 🛠️ Project: The "SysInfo" Embedded Node

### 📋 Project Requirements
Create a bootable Linux system for QEMU (ARM64) that:
1.  Boots a custom-compiled **Linux 6.6 Kernel**.
2.  Mounts a persistent **Ext4 RootFS** populated with **BusyBox**.
3.  Loads a custom **Device Tree Overlay** to enable a specific feature (simulated).
4.  Auto-loads a custom **Kernel Module (`sysinfo.ko`)** on boot.
5.  The module creates a file `/proc/lab_sysinfo` that reports:
    *   Current Jiffies (Uptime).
    *   Number of running processes.

---

## 💻 Implementation: Step-by-Step Guide

### 🔹 Phase 1: The Kernel & DTB
*Ref: Day 122, 125*

1.  **Clean Build:**
    ```bash
    cd ~/linux-6.6.1
    make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- mrproper
    make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- defconfig
    ```
2.  **Customize:** Ensure `CONFIG_PROC_FS=y` and `CONFIG_MODULES=y`.
3.  **Build:**
    ```bash
    make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- -j$(nproc) Image dtbs modules
    ```
4.  **Artifacts:** `arch/arm64/boot/Image`, `arch/arm64/boot/dts/arm/virt.dtb` (we will use the QEMU virt DTB).

### 🔹 Phase 2: The Root Filesystem
*Ref: Day 124*

1.  **Structure:** Create `~/project_rootfs` with standard FHS.
2.  **BusyBox:** Compile dynamic BusyBox and install.
3.  **Libs:** Copy `libc.so`, `ld-linux.so`, etc.
4.  **Init Script (`/etc/init.d/rcS`):**
    ```bash
    #!/bin/sh
    mount -t proc proc /proc
    mount -t sysfs sysfs /sys
    mount -t devtmpfs none /dev
    
    echo "Loading SysInfo Module..."
    insmod /lib/modules/sysinfo.ko
    
    echo "System Ready."
    ```
5.  **Pack:** Create `rootfs.ext4` (64MB).

### 🔹 Phase 3: The Kernel Module
*Ref: Day 126*

**Source: `sysinfo.c`**
```c
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/jiffies.h>
#include <linux/sched/signal.h> // For for_each_process

MODULE_LICENSE("GPL");

static int sysinfo_show(struct seq_file *m, void *v) {
    struct task_struct *task;
    int process_count = 0;

    for_each_process(task) {
        process_count++;
    }

    seq_printf(m, "--- System Info ---\n");
    seq_printf(m, "Jiffies: %lu\n", jiffies);
    seq_printf(m, "Processes: %d\n", process_count);
    return 0;
}

static int sysinfo_open(struct inode *inode, struct file *file) {
    return single_open(file, sysinfo_show, NULL);
}

static const struct proc_ops sysinfo_ops = {
    .proc_open = sysinfo_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

static int __init sysinfo_init(void) {
    proc_create("lab_sysinfo", 0, NULL, &sysinfo_ops);
    printk(KERN_INFO "SysInfo: Module loaded.\n");
    return 0;
}

static void __exit sysinfo_exit(void) {
    remove_proc_entry("lab_sysinfo", NULL);
    printk(KERN_INFO "SysInfo: Module unloaded.\n");
}

module_init(sysinfo_init);
module_exit(sysinfo_exit);
```

**Build & Install:**
1.  Compile `sysinfo.ko`.
2.  Copy to `$ROOTFS/lib/modules/sysinfo.ko`.

### 🔹 Phase 4: Integration & Boot

**Command:**
```bash
qemu-system-aarch64 \
    -M virt \
    -cpu cortex-a53 \
    -nographic \
    -smp 2 \
    -m 512M \
    -kernel arch/arm64/boot/Image \
    -drive format=raw,file=rootfs.ext4,if=virtio \
    -append "console=ttyAMA0 root=/dev/vda rw"
```

**Verification:**
1.  Login as root.
2.  Check load: `dmesg | grep SysInfo`.
3.  Check proc: `cat /proc/lab_sysinfo`.
    ```text
    --- System Info ---
    Jiffies: 4294937500
    Processes: 5
    ```
4.  Start a background process (`sleep 1000 &`) and check again. Process count should increase.

---

## 🐞 Troubleshooting the Project

### Issue 1: "insmod: can't insert 'sysinfo.ko': invalid module format"
*   **Fix:** You compiled the module against a different kernel version than the one you are booting. Rebuild the module pointing `KDIR` to the exact kernel source tree used for `Image`.

### Issue 2: "Unknown symbol"
*   **Fix:** Did you use `for_each_process`? It requires `GPL` license. Ensure `MODULE_LICENSE("GPL")` is present.

---

## 📈 Grading Rubric

| Criteria | Excellent (A) | Good (B) | Needs Improvement (C) |
| :--- | :--- | :--- | :--- |
| **Kernel Boot** | Boots to shell in < 5s. No errors. | Boots but with warnings. | Panic or hang. |
| **RootFS** | Clean structure, persistent storage works. | Messy structure, read-only issues. | Missing files, cannot mount. |
| **Module** | Correctly reports dynamic data via /proc. | Static data or crashes on read. | Fails to load. |
| **Automation** | `rcS` auto-loads module. | Manual `insmod` required. | No init script. |

---

## 🔮 Looking Ahead: Week 19
Now that we have a running kernel, we will dive into **Character Device Drivers**.
*   Major/Minor numbers.
*   `file_operations` (open, read, write, ioctl).
*   Interacting with GPIO from Kernel Space.

---
