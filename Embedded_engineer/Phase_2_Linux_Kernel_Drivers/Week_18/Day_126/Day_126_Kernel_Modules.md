# Day 126: Kernel Modules (LKM)
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
1.  **Write** a Loadable Kernel Module (LKM) with `init` and `exit` functions.
2.  **Compile** modules using the Kernel Build System (Kbuild).
3.  **Manage** modules using `insmod`, `rmmod`, `modprobe`, and `lsmod`.
4.  **Pass Parameters** to modules at load time.
5.  **Analyze** the Kernel Symbol Table (`/proc/kallsyms`) and dependencies.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Kernel Headers/Source (configured and built).
    *   `kmod` tools (installed in RootFS).
*   **Prior Knowledge:**
    *   C Programming (Pointers, Structures).
    *   Day 122 (Kbuild).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is a Kernel Module?
The Linux Kernel is monolithic, but it is **extensible**.
*   **Static:** Compiled into `vmlinux`. Always present. Good for core functionality (Scheduler, MMU).
*   **Modular:** Compiled as `.ko` (Kernel Object). Loaded/Unloaded on demand. Good for device drivers (USB, WiFi) to save memory.
*   **No Standard Lib:** You cannot use `printf`, `malloc`, `sleep`. You must use `printk`, `kmalloc`, `ssleep`.
*   **Kernel Space:** A bug here crashes the whole system (Panic).

### 🔹 Part 2: The Module Lifecycle
1.  **Loading (`insmod`):**
    *   Kernel allocates memory.
    *   Resolves symbols (links against kernel functions).
    *   Calls `module_init()`.
2.  **Running:**
    *   Module sits in memory, waiting for events (IRQs, System Calls).
3.  **Unloading (`rmmod`):**
    *   Calls `module_exit()`.
    *   Frees memory.

### 🔹 Part 3: Licensing and Tainting
Linux is GPL.
*   `MODULE_LICENSE("GPL")`: Access to all symbols.
*   `MODULE_LICENSE("Proprietary")`: Access only to non-GPL-only symbols. Taints the kernel (Community won't debug your crash logs).

---

## 💻 Implementation: The "Hello World" Module

> **Instruction:** Write the simplest possible module.

### 👨‍💻 Code Implementation

#### Step 1: Source Code (`hello.c`)
```c
#include <linux/module.h>  // Core header for modules
#include <linux/kernel.h>  // Macros like KERN_INFO
#include <linux/init.h>    // Macros for __init and __exit

// Metadata
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Embedded Engineer");
MODULE_DESCRIPTION("A simple Hello World LKM");
MODULE_VERSION("1.0");

// Initialization Function
static int __init hello_init(void) {
    printk(KERN_INFO "Hello: Module loaded successfully!\n");
    return 0; // Return 0 means success. Non-zero means failure (module won't load).
}

// Cleanup Function
static void __exit hello_exit(void) {
    printk(KERN_INFO "Hello: Module unloaded. Goodbye!\n");
}

// Register Entry/Exit Points
module_init(hello_init);
module_exit(hello_exit);
```

#### Step 2: Makefile
We need a specific Makefile that invokes the Kernel build system.
```make
# If KERNELRELEASE is defined, we've been invoked from the kernel build system
ifneq ($(KERNELRELEASE),)
    obj-m := hello.o

# Otherwise, we were called directly from the command line
else
    # Path to kernel source (Host or Cross-Target)
    KDIR ?= ~/linux_kernel_course/linux-6.6.1
    PWD := $(shell pwd)

default:
    $(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
    $(MAKE) -C $(KDIR) M=$(PWD) clean
endif
```

#### Step 3: Compile
```bash
# For Host (x86)
make

# For Target (ARM64)
make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu-
```
Result: `hello.ko`.

---

## 💻 Implementation: Loading and Unloading

> **Instruction:** Test the module on the target (QEMU).

### 👨‍💻 Command Line Steps

#### Step 1: Transfer to Target
Copy `hello.ko` to your RootFS (or use SCP if networking is up).

#### Step 2: Insert Module
```bash
insmod hello.ko
```
Check logs:
```bash
dmesg | tail
# [  123.456] Hello: Module loaded successfully!
```

#### Step 3: List Modules
```bash
lsmod
# Module                  Size  Used by
# hello                  16384  0
```

#### Step 4: Remove Module
```bash
rmmod hello
```
Check logs:
```bash
dmesg | tail
# [  125.678] Hello: Module unloaded. Goodbye!
```

---

## 💻 Implementation: Module Parameters

> **Instruction:** Allow the user to configure the module at load time.

### 👨‍💻 Code Implementation

#### Step 1: Update `hello.c`
```c
#include <linux/moduleparam.h>

static int my_int = 42;
static char *my_str = "default";

// Permissions: S_IRUGO (Read-only by User/Group/Others)
module_param(my_int, int, S_IRUGO);
MODULE_PARM_DESC(my_int, "An integer parameter");

module_param(my_str, charp, S_IRUGO);
MODULE_PARM_DESC(my_str, "A string parameter");

static int __init hello_init(void) {
    printk(KERN_INFO "Hello: int=%d, str=%s\n", my_int, my_str);
    return 0;
}
```

#### Step 2: Compile and Test
```bash
insmod hello.ko my_int=100 my_str="Custom"
# dmesg: Hello: int=100, str=Custom
```

#### Step 3: Runtime Modification
Since we used `S_IRUGO`, parameters appear in sysfs.
```bash
cat /sys/module/hello/parameters/my_int
# 100
```
If we used `S_IRUGO | S_IWUSR` (Writeable by root), we could change it on the fly:
```bash
echo 500 > /sys/module/hello/parameters/my_int
```

---

## 🔬 Lab Exercise: Lab 126.1 - Dependency Chain

### 1. Lab Objectives
- Create two modules: `math_mod` and `calc_mod`.
- `math_mod` exports a function `add(a, b)`.
- `calc_mod` calls `add(a, b)`.
- Observe dependency handling.

### 2. Step-by-Step Guide

#### Phase A: `math_mod.c`
```c
#include <linux/module.h>

int add(int a, int b) {
    return a + b;
}
EXPORT_SYMBOL(add); // Crucial! Makes it visible to other modules.

static int __init math_init(void) { return 0; }
static void __exit math_exit(void) { }

module_init(math_init);
module_exit(math_exit);
MODULE_LICENSE("GPL");
```

#### Phase B: `calc_mod.c`
```c
#include <linux/module.h>

extern int add(int, int); // Declaration

static int __init calc_init(void) {
    printk("2 + 3 = %d\n", add(2, 3));
    return 0;
}
static void __exit calc_exit(void) { }

module_init(calc_init);
module_exit(calc_exit);
MODULE_LICENSE("GPL");
```

#### Phase C: Testing
1.  Compile both.
2.  Try `insmod calc_mod.ko`. **Error:** `Unknown symbol in module`.
3.  `insmod math_mod.ko`. Success.
4.  `insmod calc_mod.ko`. Success. `dmesg` shows "2 + 3 = 5".
5.  Try `rmmod math_mod`. **Error:** `Module is in use by calc_mod`.
6.  `rmmod calc_mod` then `rmmod math_mod`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Modprobe and Depmod
- **Goal:** Use `modprobe` to handle dependencies automatically.
- **Task:**
    1.  Install modules to `/lib/modules/$(uname -r)/`.
    2.  Run `depmod -a`. This generates `modules.dep`.
    3.  Run `modprobe calc_mod`. It should automatically load `math_mod` first.

### Lab 3: The `__init` and `__exit` Macros
- **Goal:** Understand memory optimization.
- **Task:**
    1.  Research what `__init` does. (It puts code in a special section `.init.text`).
    2.  Explain why the kernel frees this memory after loading. (We don't need the init function once the module is running).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Exec format error"
*   **Cause:** You compiled for x86 but tried to load on ARM (or vice versa).
*   **Solution:** Check `file hello.ko`. Ensure Architecture matches.

#### 2. "version magic '...' should be '...'"
*   **Cause:** The kernel source you compiled against is different from the running kernel.
*   **Solution:** Ensure `KDIR` points to the exact source tree used to build the running `Image`.

#### 3. "Key was rejected by service"
*   **Cause:** Secure Boot is enabled, and your module is unsigned.
*   **Solution:** Disable Secure Boot in BIOS (for PC) or sign the module (advanced).

---

## ⚡ Optimization & Best Practices

### Coding Standards
- **Checkpatch:** Use `scripts/checkpatch.pl` to verify your code style before submitting patches.
- **Error Handling:** Always check return values. If `kmalloc` fails, handle it gracefully. Do not leak memory in error paths (use `goto` for cleanup).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `insmod` and `modprobe`?
    *   **A:** `insmod` is dumb (loads exactly what you give it). `modprobe` is smart (reads `modules.dep` and loads dependencies).
2.  **Q:** Can I use floating point math in a kernel module?
    *   **A:** Generally NO. Saving/restoring FPU registers is expensive and not done by default in kernel mode. Use fixed-point arithmetic.

### Challenge Task
> **Task:** "The Auto-Loader". Create a module that aliases a specific hardware ID (e.g., a specific USB Vendor/Product ID). When you plug in that USB device, `udev` should trigger `modprobe` to load your module automatically.

---

## 📚 Further Reading & References
- [Linux Device Drivers, 3rd Edition (LDD3)](https://lwn.net/Kernel/LDD3/) - Old but gold.
- [Kernel Documentation: Documentation/kbuild/modules.rst](https://www.kernel.org/doc/Documentation/kbuild/modules.rst)

---
