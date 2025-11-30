# Day 123: Kernel Boot Process
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
1.  **Trace** the Linux boot sequence from Bootloader handoff to User Space `init`.
2.  **Analyze** Kernel Boot Logs (`dmesg`) to identify hardware initialization order.
3.  **Modify** Kernel Command Line Parameters (`bootargs`) to change boot behavior.
4.  **Create** a minimal `initramfs` (Initial RAM Filesystem) from scratch.
5.  **Explain** the role of PID 1 (`init` or `systemd`) in bringing up the system.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU for simulation).
*   **Software Required:**
    *   `qemu-system-aarch64` (for testing kernels without hardware).
    *   `cpio`, `gzip`.
*   **Prior Knowledge:**
    *   Day 122 (Kernel Build).
    *   Basic File System concepts.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Handoff (Bootloader -> Kernel)
The Bootloader (U-Boot, GRUB) does the heavy lifting of initializing DDR RAM and loading the Kernel Image from disk/flash into RAM.
*   **Registers:** Bootloader sets up CPU registers (e.g., `r0=0`, `r1=machine_id`, `r2=dtb_addr` on ARM32).
*   **Jump:** It jumps to the Kernel Entry Point (`stext` or `_start`).

### 🔹 Part 2: Inside the Kernel (Early Boot)
1.  **Decompression:** The kernel (zImage) decompresses itself into RAM.
2.  **Assembly Start (`head.S`):**
    *   Checks CPU ID.
    *   Validates Device Tree (DTB).
    *   Sets up initial Page Tables (MMU).
    *   Enables MMU (Virtual Memory on!).
    *   Jumps to C code (`start_kernel`).
3.  **C Start (`init/main.c`):**
    *   `start_kernel()`: The "main" function of the OS.
    *   Initializes Interrupts, Memory, Scheduler, Timers, Console.
    *   Spawns `init` process (PID 1).

### 🔹 Part 3: Mounting the Root FS
The kernel needs a filesystem to run user programs (`/bin/sh`, `/sbin/init`).
*   **Problem:** The drivers for the disk (SATA/NVMe/SD) might be modules stored *on* the disk! Chicken and Egg.
*   **Solution:** **Initramfs** (Initial RAM Filesystem).
    *   A small CPIO archive linked into the kernel or loaded by bootloader.
    *   Contains essential drivers (ko) and a script (`/init`).
    *   The script loads drivers, mounts the *real* root FS, and `switch_root` to it.

```mermaid
graph TD
    Bootloader[U-Boot/GRUB] -->|Load| Kernel[zImage/Image]
    Bootloader -->|Load| DTB[Device Tree]
    Bootloader -->|Load| Initramfs[Initramfs.cpio.gz]
    Kernel -->|Decompress| Self[vmlinux]
    Self -->|start_kernel| InitHardware[Init IRQ/Mem/Sched]
    InitHardware -->|Mount| RootFS_Ram[Initramfs (TmpFS)]
    RootFS_Ram -->|Execute| InitScript[/init]
    InitScript -->|Load Modules| Drivers
    InitScript -->|Mount| RealRoot[/dev/sda1]
    InitScript -->|Switch Root| Systemd[/sbin/init]
```

---

## 💻 Implementation: Analyzing Boot Logs

> **Instruction:** Use QEMU to boot your kernel and inspect the logs.

### 👨‍💻 Command Line Steps

#### Step 1: Run QEMU
Assuming you built the ARM64 kernel in Day 122.
```bash
qemu-system-aarch64 \
    -M virt \
    -cpu cortex-a53 \
    -nographic \
    -smp 1 \
    -m 512M \
    -kernel arch/arm64/boot/Image \
    -append "console=ttyAMA0"
```
*   `-M virt`: Generic Virtual Machine.
*   `-nographic`: Output to terminal.
*   `-kernel`: Your built image.
*   `-append`: Kernel Command Line (set console to UART0).

#### Step 2: The Panic
It will crash!
```text
Kernel panic - not syncing: VFS: Unable to mount root fs on unknown-block(0,0)
```
**Why?** We provided a kernel, but no filesystem. The kernel finished booting, looked for `init`, couldn't find it, and panicked. This is **Success** (The kernel works!).

---

## 💻 Implementation: Creating a Minimal Initramfs

> **Instruction:** Build a tiny filesystem with just a "Hello World" init.

### 👨‍💻 Code Implementation

#### Step 1: Create Structure
```bash
mkdir -p ~/initramfs_work
cd ~/initramfs_work
mkdir -p bin dev etc lib proc sys tmp usr
```

#### Step 2: Create `init` (The C Program)
We will write a static C program to act as `init`.
```c
// init.c
#include <stdio.h>
#include <unistd.h>

int main(void) {
    printf("Hello from Userspace!\n");
    printf("I am the Init Process (PID %d)\n", getpid());
    while(1) {
        sleep(1);
    }
    return 0;
}
```

#### Step 3: Compile Static
Must be static because we have no libraries (`libc.so`) in our initramfs yet.
```bash
aarch64-linux-gnu-gcc -static -o init init.c
```

#### Step 4: Package (CPIO)
```bash
find . -print0 | cpio --null -ov --format=newc | gzip -9 > ../initramfs.cpio.gz
```

---

## 💻 Implementation: Booting with Initramfs

> **Instruction:** Boot QEMU again, this time with the filesystem.

### 👨‍💻 Command Line Steps

#### Step 1: Run QEMU
```bash
qemu-system-aarch64 \
    -M virt \
    -cpu cortex-a53 \
    -nographic \
    -smp 1 \
    -m 512M \
    -kernel ~/linux_kernel_course/linux-6.6.1/arch/arm64/boot/Image \
    -initrd ../initramfs.cpio.gz \
    -append "console=ttyAMA0 rdinit=/init"
```
*   `-initrd`: Pass the archive.
*   `rdinit=/init`: Tell kernel to execute `/init` from RAM disk.

#### Step 2: Success
```text
[    0.000000] Booting Linux on physical CPU 0x0000000000 [0x410fd034]
...
[    0.543210] Run /init as init process
Hello from Userspace!
I am the Init Process (PID 1)
```
**Victory!** You have booted a full Linux system from scratch.

---

## 🔬 Lab Exercise: Lab 123.1 - BusyBox Initramfs

### 1. Lab Objectives
- Replace the dummy C program with **BusyBox**.
- Get a working Shell (`/bin/sh`).

### 2. Step-by-Step Guide

#### Phase A: Download & Config BusyBox
1.  Download source from busybox.net.
2.  `make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- defconfig`
3.  `make menuconfig` -> **Settings** -> **Build static binary (no shared libs)** -> Enable.

#### Phase B: Build & Install
1.  `make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- -j8`
2.  `make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- install CONFIG_PREFIX=~/initramfs_busybox`

#### Phase C: Package & Boot
1.  `cd ~/initramfs_busybox`
2.  Create `init` script (Shell script this time):
    ```bash
    #!/bin/sh
    mount -t proc none /proc
    mount -t sysfs none /sys
    echo "Welcome to BusyBox Linux!"
    exec /bin/sh
    ```
3.  `chmod +x init`
4.  Package with `cpio`.
5.  Boot QEMU.
6.  **Observation:** You drop into a shell! `ls`, `cat`, `top` work.

### 3. Verification
Run `cat /proc/cpuinfo` inside the VM. It should show AArch64 Processor.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Kernel Command Line
- **Goal:** Change behavior without recompiling.
- **Task:**
    1.  Add `loglevel=8` to `-append`. (More verbose).
    2.  Add `init=/bin/sh` (Bypass init script, drop to shell directly).
    3.  Add `mem=256M` (Restrict RAM).

### Lab 3: Init Systems
- **Goal:** Understand SysVinit vs Systemd.
- **Task:**
    1.  Research `inittab`.
    2.  Create `etc/inittab` in BusyBox:
        ```text
        ::sysinit:/etc/init.d/rcS
        ::askfirst:/bin/sh
        ```
    3.  See how BusyBox parses this to manage startup services.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Kernel panic - not syncing: No working init found"
*   **Cause:** The kernel couldn't execute `/init`.
*   **Reasons:**
    *   File missing.
    *   File not executable (`chmod +x`).
    *   Architecture mismatch (Compiled x86 instead of ARM).
    *   Missing libraries (if not static).

#### 2. "console=ttyAMA0" vs "ttyS0"
*   **Cause:** Different boards use different UART drivers.
*   **Solution:** For QEMU `virt` machine, it is `ttyAMA0`. For x86 PC, it is `ttyS0`. Check `/proc/tty/drivers` if possible.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Initramfs Size:** Keep it small. It sits in RAM. Don't put 100MB files there. Use `xz` compression instead of `gzip` for better ratio (kernel must support XZ decompression).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is PID 1?
    *   **A:** The first process started by the kernel. If PID 1 dies, the kernel panics (System Crash).
2.  **Q:** Why do we mount `/proc` and `/sys`?
    *   **A:** These are virtual filesystems. They are the interface between Kernel and User Space. `ps` needs `/proc` to list processes.

### Challenge Task
> **Task:** "The Rescue Disk". Create an initramfs that contains `gdbserver` and network tools. Boot it, configure networking (`ifconfig`), and connect from your host to debug a "crashed" application.

---

## 📚 Further Reading & References
- [Bootlin: Embedded Linux Course Slides](https://bootlin.com/doc/training/embedded-linux/)
- [BusyBox Command Help](https://busybox.net/downloads/BusyBox.html)

---
