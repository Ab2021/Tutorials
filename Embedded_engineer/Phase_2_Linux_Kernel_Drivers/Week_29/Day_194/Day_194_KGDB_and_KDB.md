# Day 194: KGDB and KDB (Kernel Debugger)
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
1.  **Configure** the kernel for KGDB/KDB support.
2.  **Connect** GDB from a host machine to a target kernel via Serial/Agent-Proxy.
3.  **Use** KDB commands (`md`, `rd`, `bt`) to inspect state without GDB.
4.  **Set** Breakpoints and Step through kernel code.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Two machines (Host and Target) connected via Serial Cable (or QEMU).
*   **Software Required:**
    *   `gdb-multiarch` (on Host).
    *   Kernel with `CONFIG_KGDB`, `CONFIG_KGDB_KDB`.
*   **Prior Knowledge:**
    *   Basic GDB usage.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: KGDB vs KDB
*   **KGDB (Kernel GDB):** The backend. Allows a remote GDB instance to control the kernel. Supports source-level debugging, variables, macros.
*   **KDB (Kernel Debugger):** A simple shell built into the kernel. Runs on the target's console. Good for quick inspection (`lsmod`, `dmesg`, memory dump) when you don't have a second PC.

### 🔹 Part 2: How it works
*   **Polling Mode:** When KGDB is active, the kernel stops all other CPUs. It switches the Serial Driver to "Polling Mode" (bypassing interrupts) to communicate with GDB.
*   **Exceptions:** Breakpoints (`int3`) trigger the KGDB exception handler.

---

## 💻 Implementation: Setting up KGDB

> **Instruction:** We will configure QEMU to allow GDB connection.

### 👨‍💻 Step-by-Step Guide

#### Step 1: Kernel Config
Ensure these are set:
```
CONFIG_KGDB=y
CONFIG_KGDB_SERIAL_CONSOLE=y
CONFIG_KGDB_KDB=y
CONFIG_DEBUG_INFO=y
```

#### Step 2: Boot Arguments
Add to kernel command line:
```
kgdboc=ttyS0,115200 kgdbwait
```
*   `kgdboc`: KGDB over Console (Serial Port 0).
*   `kgdbwait`: Stop kernel at boot and wait for GDB.

#### Step 3: Launch QEMU
```bash
qemu-system-x86_64 -kernel bzImage -append "console=ttyS0 kgdboc=ttyS0,115200 kgdbwait" -serial tcp::1234,server,nowait
```
*   Exposes serial port on TCP 1234.

#### Step 4: Connect GDB
On Host:
```bash
gdb-multiarch ./vmlinux
(gdb) target remote localhost:1234
```
**Result:** GDB connects! The kernel is paused.
```
Remote debugging using localhost:1234
kgdb_breakpoint () at kernel/debug/debug_core.c:1073
(gdb) continue
```

---

## 🔬 Lab Exercise: Lab 194.1 - Debugging a Module

### 1. Lab Objectives
- Load a module.
- Break into KGDB.
- Set a breakpoint in the module.
- Trigger it.

### 2. Step-by-Step Guide
1.  **Boot:** Start QEMU/Target.
2.  **Load Module:** `insmod my_driver.ko`.
3.  **Get Address:**
    *   Need the load address for GDB to know where symbols are.
    *   Target: `cat /sys/module/my_driver/sections/.text`.
    *   (Let's say it is `0xffffffffa0000000`).
4.  **Load Symbols (Host GDB):**
    *   Break into GDB (Send `SysRq-g` or `echo g > /proc/sysrq-trigger`).
    *   `(gdb) add-symbol-file my_driver.ko 0xffffffffa0000000`.
5.  **Set Breakpoint:**
    *   `(gdb) break my_write`.
    *   `(gdb) continue`.
6.  **Trigger:**
    *   Target: `echo "test" > /dev/my_device`.
7.  **Hit:**
    *   GDB stops at `my_write`.
    *   `(gdb) print buf`.
    *   `(gdb) step`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Using KDB
- **Goal:** Use the built-in shell.
- **Task:**
    1.  Boot with `kgdboc` but *without* `kgdbwait`.
    2.  Trigger debugger: `echo g > /proc/sysrq-trigger`.
    3.  You drop into `kdb> ` prompt on the serial console.
    4.  Commands:
        *   `bt`: Backtrace.
        *   `md <addr>`: Memory Dump.
        *   `rd`: Register Dump.
        *   `go`: Continue.

### Lab 3: Debugging a Crash
- **Goal:** Catch a panic.
- **Task:**
    1.  Create a NULL pointer dereference module.
    2.  Set `sysctl kernel.panic = 0` (Don't reboot on panic).
    3.  Trigger crash.
    4.  Kernel should drop into KDB (if configured) allowing you to inspect variables at the moment of death.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Target remote failed"
*   **Cause:** Serial port mismatch or baud rate wrong.
*   **Fix:** Check `kgdboc` parameter.

#### 2. Optimization optimized out variables
*   **Cause:** `-O2` optimization.
*   **Fix:** Use `volatile` or compile kernel with `CONFIG_DEBUG_INFO_BTF` / `CONFIG_GDB_SCRIPTS`. Or just inspect registers (`$rdi`, `$rsi`).

---

## ⚡ Optimization & Best Practices

### `scripts/gdb/vmlinux-gdb.py`
*   The kernel source provides Python scripts for GDB.
*   Enable `CONFIG_GDB_SCRIPTS`.
*   In GDB: `source /path/to/linux/vmlinux-gdb.py`.
*   New commands: `lx-dmesg`, `lx-lsmod`, `lx-ps`. Very powerful!

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can I debug the kernel using KGDB over Ethernet?
    *   **A:** Yes, using `kgdboe` (KGDB over Ethernet). However, it is less reliable than serial because the network stack is complex and might be the thing you are debugging.
2.  **Q:** What is `SysRq-g`?
    *   **A:** The Magic System Request key combination to force the kernel to enter the debugger immediately.

### Challenge Task
> **Task:** "The Detective".
> *   Boot a kernel with a hidden bug (e.g., a variable that gets corrupted).
> *   Use a **Watchpoint** in GDB (`watch my_global_var`).
> *   Run the system.
> *   GDB should stop exactly when the variable is modified, revealing the culprit code.

---

## 📚 Further Reading & References
- [Kernel Documentation: dev-tools/kgdb.rst](https://www.kernel.org/doc/html/latest/dev-tools/kgdb.html)

---
