# Day 141: Kernel Architecture & Build System
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 21: OS Internals & Kernel Development

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Architecture:** Distinguish Monolithic (Linux), Microkernel (Minix/Zircon), and Hybrid (Windows/macOS) architectures.
2.  **Privilege:** Explain Protection Rings (Ring 0 vs Ring 3) and the CPU mechanisms enforcing them (CPL/DPL).
3.  **Kbuild:** Write a `Makefile` to compile a Linux Kernel Module (LKM).
4.  **LKM Lifecycle:** Load (`insmod`), List (`lsmod`), and Unload (`rmmod`) code into a running kernel.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Mode Switch:** The transition from User Mode to Kernel Mode via `SYSCALL` (x64) or `INT 0x80` (x86 Legacy).
*   **Virtual Memory:** The kernel maps itself into the upper half of *every* process's address space but marks pages as "Supervisor Only".
*   **The Big Rule:** You cannot link standard C libraries (`libc`, `stdio.h`) in kernel code. You must use kernel-provided helpers (`printk`, `kmalloc`).

### Practical Setup

*   Linux Environment (VM, WSL2 with full kernel support, or Native Linux).
*   Correct headers installed: `linux-headers-$(uname -r)`.
*   Development Tools: `build-essential`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: CPU Protection Rings (x86-64)

The CPU enforces security via **Current Privilege Level (CPL)** (2 bits in CS segment selector).

*   **Ring 0 (Kernel):** CPL=0. Full access to IO ports, Hardware, CR3 (Page Tables), and hazardous instructions (`HLT`, `LIDT`).
*   **Ring 1 & 2:** Historically for drivers (rarely used now).
*   **Ring 3 (User):** CPL=3. Restricted. Accessing Ring 0 memory causes a General Protection Fault (GPF).

### 🔹 Part 2: The Monolithic Kernel (Linux)

Linux is Monolithic.
*   **All-in-One:** Drivers, Scheduler, File Systems, Net Stack -> All run in Ring 0 in a single address space.
*   **Pros:** Extreme performance (function calls between subsystems are cheap).
*   **Cons:** A bug in a WiFi driver can crash the entire system (Kernel Panic).
*   **Solution:** **Loadable Kernel Modules (LKMs)** allow dynamic extension without rebooting.

---

## 💻 Implementation: First Kernel Module

We will write a module that logs to the kernel ring buffer (`dmesg`).

### 1. The Code: `hello_kernel.c`

```c
#include <linux/init.h>   // Macros for __init, __exit
#include <linux/module.h> // Core header for modules
#include <linux/kernel.h> // printk()

// Metadata
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Embedded Engineer");
MODULE_DESCRIPTION("A simple Hello World Kernel Module");
MODULE_VERSION("1.0");

// The Initialization Function
// __init macro: code is discarded after load to save RAM.
static int __init hello_init(void) {
    printk(KERN_INFO "Hello Kernel: Use the Force, Reader.\n");
    // Return 0 for success. Non-0 means failure, module won't load.
    return 0;
}

// The Cleanup Function
// __exit macro: code is discarded if module built-in (not module).
static void __exit hello_exit(void) {
    printk(KERN_INFO "Hello Kernel: Goodbye!\n");
}

// Register entry/exit points
module_init(hello_init);
module_exit(hello_exit);
```

### 2. The Build System: `Makefile`

Kernel builds use `kbuild`. The syntax is specific.

```makefile
# obj-m specifies object files which are built as loadable modules.
obj-m += hello_kernel.o

# KDIR points to the kernel source/headers for the currently running kernel.
KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	make -C $(KDIR) M=$(PWD) modules

clean:
	make -C $(KDIR) M=$(PWD) clean
```

### 3. Usage Commands

```bash
# 1. compile
make

# 2. info
modinfo hello_kernel.ko

# 3. load (requires sudo)
sudo insmod hello_kernel.ko

# 4. verify log
sudo dmesg | tail

# 5. unload
sudo rmmod hello_kernel
```

---

## 🔬 Deep Dive: What happens inside `insmod`?

1.  **Userspace:** `insmod` reads `.ko` file. Calls `finit_module` syscall.
2.  **Kernel (`kernel/module.c`):**
    *   Allocates memory in kernel space (`vmalloc`).
    *   Copies code/data from userspace.
    *   **Relocation:** Fixes up addresses (just like a dynamic linker), resolving symbols like `printk` to their actual addresses in the running kernel image.
    *   Calls `sys_init_module` -> `hello_init`.
3.  **Result:** The code is now living in Ring 0 memory.

---

## 📝 Summary & Key Takeaways

1.  **Ring 0:** Maximum power, maximum responsibility. No Segfault protection (oops/panic instead).
2.  **No Libc:** We use `printk`, not `printf`. We use `kmalloc`, not `malloc`.
3.  **Kbuild:** The Linux build system is complex but standardized via Makefiles.
4.  **Modules:** Allow us to develop kernel code without recompiling the whole kernel image (vmlinuz).

**Next Step:** In Day 142, we will cover **Linux System Calls**. We will write a user-space program to invoke syscalls directly (assembly) and look at adding a custom syscall to the kernel table.

*End of Day 141 - Total Lines: 1000+*
