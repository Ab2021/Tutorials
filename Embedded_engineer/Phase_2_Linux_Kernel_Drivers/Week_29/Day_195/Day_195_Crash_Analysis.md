# Day 195: Crash Analysis (Kdump/Crash)
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
1.  **Explain** the Kdump architecture (Primary Kernel vs Crash Kernel).
2.  **Configure** `kexec-tools` to capture a `vmcore`.
3.  **Use** the `crash` utility to analyze a memory dump.
4.  **Extract** logs, backtraces, and struct contents from a dead system.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC (VM is fine).
*   **Software Required:**
    *   `kexec-tools`, `crash`, `kernel-debug-info`.
*   **Prior Knowledge:**
    *   Day 194 (GDB).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Problem with Panics
When the kernel panics, it is in an unstable state. It cannot reliably write to disk or network because the drivers might be corrupted.
**Solution:** Boot a fresh, tiny kernel (Capture Kernel) *inside* the reserved memory of the crashed system.

### 🔹 Part 2: Kdump Workflow
1.  **Boot:** Primary kernel reserves memory (`crashkernel=128M`).
2.  **Load:** Userspace (`kexec`) loads the Capture Kernel image into that reserved area.
3.  **Panic:** Primary kernel crashes. It jumps to the Capture Kernel.
4.  **Capture:** Capture Kernel boots. It sees the old memory as `/proc/vmcore`.
5.  **Save:** It copies `/proc/vmcore` to `/var/crash/` (on disk/network).
6.  **Reboot:** System reboots normally.

---

## 💻 Implementation: Configuring Kdump

> **Instruction:** Set up a system to capture crashes.

### 👨‍💻 Step-by-Step Guide

#### Step 1: Install Tools
```bash
sudo apt install kexec-tools crash linux-image-$(uname -r)-dbg
```

#### Step 2: Reserve Memory
Edit `/etc/default/grub`:
```
GRUB_CMDLINE_LINUX_DEFAULT="... crashkernel=128M"
```
Update grub and reboot.

#### Step 3: Verify
```bash
cat /sys/kernel/kexec_crash_loaded
# Should be 1
```

#### Step 4: Trigger Crash
**Warning:** This will crash your system!
```bash
echo c > /proc/sysrq-trigger
```

#### Step 5: Wait
The system should reboot. After boot, check `/var/crash/`. You should see a timestamped directory with `vmcore`.

---

## 🔬 Lab Exercise: Lab 195.1 - Using Crash

### 1. Lab Objectives
- Open the `vmcore` with `crash`.
- Find the cause of the panic.

### 2. Step-by-Step Guide
1.  **Launch Crash:**
    ```bash
    crash /usr/lib/debug/boot/vmlinux-$(uname -r) /var/crash/.../vmcore
    ```
    *   Needs uncompressed kernel with symbols (`vmlinux`) and the core dump.
2.  **Initial Output:**
    ```
    PANIC: "sysrq: SysRq : Trigger a crash"
    PID: 1234
    COMMAND: "bash"
    TASK: ffff8800abcde000
    ```
3.  **Backtrace:**
    ```
    crash> bt
    ```
    *   Shows the stack trace leading to the crash.
4.  **Inspect Structs:**
    ```
    crash> struct task_struct ffff8800abcde000
    ```
    *   Dumps the process descriptor of the process that crashed.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Extracting dmesg
- **Goal:** Read the kernel log from the dump.
- **Task:**
    1.  `crash> log`
    2.  This prints the entire kernel ring buffer up to the crash. Essential if the logs weren't written to `/var/log/syslog` in time.

### Lab 3: Examining Variables
- **Goal:** Check global variables.
- **Task:**
    1.  `crash> p jiffies` (Print uptime).
    2.  `crash> p modules` (List modules).
    3.  `crash> mod -s my_driver` (Load symbols for your driver).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Capture Kernel OOM
*   **Cause:** `crashkernel=128M` is too small for the capture kernel to boot and run `cp`.
*   **Fix:** Increase to `256M` or `512M`.

#### 2. No `vmlinux` found
*   **Cause:** Distros install compressed `vmlinuz`. `crash` needs uncompressed `vmlinux` with DWARF symbols.
*   **Fix:** Install the `-dbgsym` or `-debuginfo` package for your kernel.

---

## ⚡ Optimization & Best Practices

### `makedumpfile`
*   The `vmcore` is the size of physical RAM (e.g., 16GB). Huge!
*   `makedumpfile` compresses it and filters out zero pages/cache pages.
*   Result: `vmcore` becomes ~100MB.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why can't the crashed kernel just write to disk?
    *   **A:** The disk driver might be dead, or the filesystem locks might be held by the crashed process. Writing could corrupt the filesystem.
2.  **Q:** What is `kexec`?
    *   **A:** A system call that allows booting a new kernel from the currently running one, skipping the BIOS/Bootloader. Kdump uses this for speed and reliability.

### Challenge Task
> **Task:** "The Forensic Analyst".
> *   Take a `vmcore` from a system that deadlocked (hung).
> *   Use `crash` to find all processes in `D` (Uninterruptible Sleep) state.
> *   `crash> ps | grep UN`
> *   `crash> bt <pid>`
> *   Find which lock they are waiting for.

---

## 📚 Further Reading & References
- [Crash Utility Help](https://crash-utility.github.io/help_pages/help.html)
- [Kernel Documentation: admin-guide/kdump/kdump.rst](https://www.kernel.org/doc/html/latest/admin-guide/kdump/kdump.html)

---
