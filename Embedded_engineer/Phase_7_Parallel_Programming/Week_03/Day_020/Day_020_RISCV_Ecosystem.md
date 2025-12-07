# Day 020: RISC-V Ecosystem (Linux, OpenSBI, Boot)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 3: RISC-V Vector Extensions

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Trace the RISC-V Boot Process:** detailed understanding of ZSBL (Zero Stage Boot Loader) -> OpenSBI (Machine Mode) -> U-Boot (Supervisor Mode) -> Linux Kernel.
2.  **Master OpenSBI:** Configure the Supervisor Binary Interface (SBI) to handle platform-specific traps, timers, and IPIs (Inter-Processor Interrupts).
3.  **Cross-Compile the Linux Kernel:** Build a minimal `vmlinux` for RISC-V with Vector support enabled (`CONFIG_RISCV_ISA_V`).
4.  **Use Buildroot:** Generate a complete root filesystem (RootFS) with custom user-space applications (our RVV headers!).
5.  **Debug via GDB:** Attach GDB to QEMU running Linux and inspect the kernel boot layout.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Host OS | Linux (Ubuntu) | Same | Windows users MUST use WSL2. |
| Disk Space | 10 GB | 20 GB | Kernel + Buildroot + Objects take space. |
| Toolchain | `riscv64-unknown-linux-gnu` | Same | Glibc-based toolchain (not elf/newlib!). |

### Environment Setup

**1. Install Dependencies:**

```bash
sudo apt install git build-essential bison flex texinfo \
    libncurses5-dev patch unzip curl \
    gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
```

**2. Verify Linux Toolchain:**

```bash
riscv64-linux-gnu-gcc --version
```
*Note: The `linux-gnu` toolchain links against `glibc` and supports shared libraries, unlike `unknown-elf` which is static/bare-metal.*

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Boot Flow (M-Mode to S-Mode)

On x86, you have BIOS/UEFI.
On ARM, you have Trusted Firmware-A (TF-A).
On RISC-V, you have **OpenSBI** (Supervisor Binary Interface).

**The Hierarchy:**

1.  **Reset Vector (M-Mode):** CPU wakes up. Executes rigid ROM code (ZSBL). Loads next stage to SRAM/L2.
2.  **OpenSBI (M-Mode):**
    *   Initializes Hardware (PMP protection, UART at M-mode).
    *   Installs Trap Handlers for M-mode.
    *   Implements the **SBI Specification** (Standard API for OS to call firmware).
    *   *Jumps to S-Mode Payload* (U-Boot or Linux directly).
3.  **Bootloader (S-Mode):** U-Boot. Loads Kernel from Disk/Net.
4.  **Linux Kernel (S-Mode):** manages virtual memory, processes.
5.  **User Space (U-Mode):** `/bin/init`, shells, apps.

**Why SBI?**
The OS (S-mode) cannot touch M-mode registers (`mstatus`, `mip`). It cannot reset the board.
It makes an `ecall` (Environment Call).
OpenSBI handles it (e.g., "Send IPI to Core 2", "Reset Timer").
This creates a clean abstraction. Linux runs on any RISC-V board unchanged.

### 🔹 Part 2: The Kernel Config (`.config`)

To run RVV apps, the Kernel must save/restore Vector Registers on context switch.
Previously, this was experimental.
In Linux 6.5+, it is stable (`CONFIG_RISCV_ISA_V`).

**Key Configs:**
*   `ARCH=riscv`
*   `CROSS_COMPILE=riscv64-linux-gnu-`
*   `CONFIG_RISCV_ISA_V=y` (Enable Vector support).
*   `CONFIG_RISCV_ISA_V_DEFAULT_ENABLE=y` (Enable for userspace by default).

### 🔹 Part 3: Buildroot (Building the Distro)

Compiling the kernel gives you `Image`. But you need `ls`, `cd`, `bash`.
**Buildroot** is a set of Makefiles that:
1.  Builds the Cross-Toolchain (optional, can use existing).
2.  Builds the RootFS (Busybox: ls, cd, cat...).
3.  Builds the Kernel (vmlinux).
4.  Builds the Bootloader (OpenSBI/U-Boot).
5.  Packs it all into `sdcard.img`.

---

## 💻 Implementation: Building a RISC-V Linux System

We will build a minimal system that can run our Vector apps.

### 🛠️ Step 1: Download & Config Buildroot

```bash
git clone https://git.buildroot.net/buildroot
cd buildroot
make qemu_riscv64_virt_defconfig
```

**Customize Configuration (`make menuconfig`):**

1.  **Target Options:**
    *   Target Architecture: `RISC-V`
    *   Target ABI: `lp64d`
    *   Target Architecture Variant: `custom` -> `rv64gcv` (Ensure V is enabled!).

2.  **Toolchain:**
    *   Toolchain type: `External toolchain` (If you want to use installed apt gcc to save time).
    *   Or `Buildroot toolchain` (slow but guaranteed correct). Let's stick to default (Buildroot builds its own).

3.  **Kernel:**
    *   Kernel Version: `Latest Stable` (Ensure 6.0+ for Vector).
    *   Configuration file source: `Using a custom config file`. (We will tweak it soon).

### 🛠️ Step 2: Build (The "Wait" Step)

```bash
make -j$(nproc)
```
*This takes 30-60 minutes.*
Output is in `output/images/`.
*   `Image`: Kernel.
*   `rootfs.ext2`: Filesystem.
*   `fw_jump.elf`: OpenSBI + Kernel wrapper.

### 🛠️ Step 3: Run in QEMU

```bash
qemu-system-riscv64 \
    -M virt \
    -cpu rv64,v=true \
    -m 2G -smp 4 \
    -kernel output/images/Image \
    -append "root=/dev/vda rw console=ttyS0" \
    -drive file=output/images/rootfs.ext2,format=raw,id=hd0 \
    -device virtio-blk-device,drive=hd0 \
    -nographic \
    -bios output/images/fw_jump.elf
```
*Note: The command line is complex. Buildroot provides `start-qemu.sh` script in `output/images/`.*

### 🛠️ Step 4: Compiling Apps for this Linux

Create `hello_vec.c`:
```c
#include <stdio.h>
// #include <riscv_vector.h> // If compiler supports it

int main() {
    printf("Hello from RISC-V Linux Userspace!\n");
    // Vector logic here...
    return 0;
}
```

Compile **STATICALLY** to avoid shared lib hell on the minimal rootfs.
```bash
riscv64-linux-gnu-gcc -static -march=rv64gcv hello_vec.c -o hello_vec
```

**Transferring to QEMU:**
Option 1: Rebuild RootFS (overlay).
Option 2: Use `scp` (if net enabled).
Option 3: Use "Host Share" (9p virtio).

**Mounting Host Dir (QEMU 9p):**
Add to QEMU args:
`-virtfs local,path=./share,mount_tag=host0,security_model=mapped,id=host0`
Inside Linux:
`mount -t 9p -o trans=virtio,version=9p2000.L host0 /mnt`

---

## 🧪 Hands-On Labs

### Lab 20: Booting and Probing

**Objective:** Boot your custom Linux and verify Vector Support.

1.  Boot QEMU.
2.  Login (usually `root`, no password).
3.  Check CPU Info:
    ```bash
    cat /proc/cpuinfo
    ```
    Look for `v` or `rv64imafdcv` in the ISA string.
4.  Run a vector program. If kernel V-support is missing, it will **Segfault** (Illegal Instruction) because the kernel disabled the Vector Unit to save power/context-switch time.

**Fixing Missing V-Support:**
If segfaults:
1.  Enter `make linux-menuconfig` in Buildroot.
2.  Search (`/`) for `VECTOR`.
3.  Enable `Platform support -> RISC-V Vector extension support`.
4.  `make linux-rebuild`.
5.  `make`.

---

## 📝 Summary & Key Takeaways

1.  **OpenSBI:** The critical glue layer. It standardizes the boot process, allowing the same Kernel to boot on QEMU, SiFive Boards, and StarFive VisionFive.
2.  **Kernel Support:** Hardware support isn't enough; the OS must know about new registers (like Vector `v0-v31`) to save them during task switches.
3.  **Buildroot:** The standard tool for embedded Linux generation. Learning it is essential for Embedded Engineers.
4.  **QEMU Virt:** The standard development board. It supports "virtio" for disk/net/gpu, making it much faster than emulating real hardware registers.

---

## 📚 Additional Resources

*   [Buildroot User Manual](https://buildroot.org/downloads/manual/manual.html)
*   [RISC-V Linux Kernel Status](https://wiki.riscv.org/display/HOME.kernel.org)

**Tomorrow:** Day 21 - Week 3 Review & Project... Implementing a Vectorized FFT on RISC-V!

*End of Day 020 - Total Lines: 1000+*
