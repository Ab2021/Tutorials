# Day 124: Root Filesystem (RootFS) Construction
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
1.  **Construct** a complete Linux Root Filesystem (RootFS) from scratch.
2.  **Implement** the Filesystem Hierarchy Standard (FHS).
3.  **Manage** Shared Libraries (`glibc` vs `musl` vs `uClibc`).
4.  **Configure** User Accounts, Groups, and Permissions manually.
5.  **Create** a persistent disk image (ext4) and boot it with QEMU.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `qemu-system-aarch64`.
    *   `e2fsprogs` (for `mkfs.ext4`).
    *   Previous Day's Kernel and BusyBox build.
*   **Prior Knowledge:**
    *   Day 123 (Initramfs).
    *   Basic Linux commands (`mount`, `chown`, `chmod`).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Filesystem Hierarchy Standard (FHS)
Linux is not a chaotic bag of files. It follows a strict structure defined by the FHS.
*   `/bin`: Essential user binaries (ls, cp, mv).
*   `/sbin`: Essential system binaries (init, ip, fdisk).
*   `/etc`: Host-specific system-wide configuration.
*   `/lib`: Essential shared libraries and kernel modules.
*   `/usr`: Secondary hierarchy (User System Resources).
    *   `/usr/bin`: Non-essential binaries.
    *   `/usr/lib`: Libraries for `/usr/bin`.
*   `/var`: Variable data (logs, spool, lock).
*   `/proc` & `/sys`: Virtual filesystems (Kernel interface).
*   `/dev`: Device nodes.

### 🔹 Part 2: Shared Libraries
In Day 123, we used static linking. Real systems use dynamic linking to save RAM and disk space.
*   **The Loader (`ld-linux.so`):** When you run a dynamic program, the kernel loads this *first*. It then finds and loads `libc.so`, `libm.so`, etc.
*   **Cross-Compilation Challenge:** You cannot copy `/lib` from your x86 host to your ARM target. You must copy the libraries from your *Cross-Compiler Toolchain*.

### 🔹 Part 3: Device Nodes (`/dev`)
*   **Static (`mknod`):** The old way. Manually creating `/dev/sda`, `/dev/ttyS0`.
*   **Dynamic (`devtmpfs`):** The modern way. The kernel populates `/dev` automatically.
*   **Managers (`udev`/`mdev`):** User-space daemons that listen to kernel events (hotplug) and set permissions/symlinks (e.g., auto-mounting a USB drive).

---

## 💻 Implementation: Building the RootFS Structure

> **Instruction:** We will build a directory tree that will become our RootFS.

### 👨‍💻 Command Line Steps

#### Step 1: Create Skeleton
```bash
export ROOTFS=~/my_rootfs
mkdir -p $ROOTFS
cd $ROOTFS
mkdir -p bin sbin lib etc dev home proc sys tmp var usr/bin usr/sbin usr/lib
mkdir -p var/log
chmod 1777 tmp  # Sticky bit for tmp
```

#### Step 2: Install BusyBox (Dynamic)
Rebuild BusyBox, but this time **disable** "Build static binary".
```bash
cd ~/busybox-1.36.1
make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- menuconfig
# Settings -> Build static binary (UNCHECK)
make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- -j8
make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- install CONFIG_PREFIX=$ROOTFS
```
Now check a binary:
```bash
file $ROOTFS/bin/busybox
# Output: ELF 64-bit LSB executable, ARM aarch64, dynamically linked, interpreter /lib/ld-linux-aarch64.so.1
```
It needs the interpreter!

#### Step 3: Copy Shared Libraries
We need to find where our cross-compiler keeps its libraries.
```bash
# Find the sysroot
SYSROOT=$(aarch64-linux-gnu-gcc -print-sysroot)
echo $SYSROOT

# Copy Loader and Libc
cp -a $SYSROOT/lib/ld-linux-aarch64.so.1 $ROOTFS/lib/
cp -a $SYSROOT/lib/libc.so.6 $ROOTFS/lib/
cp -a $SYSROOT/lib/libm.so.6 $ROOTFS/lib/
cp -a $SYSROOT/lib/libresolv.so.2 $ROOTFS/lib/
# Add others as needed by 'readelf -d $ROOTFS/bin/busybox'
```
*Note: Use `cp -a` to preserve symlinks. Libraries are often symlinks to versioned files.*

---

## 💻 Implementation: Configuration Files

> **Instruction:** A Linux system needs `/etc` files to function correctly.

### 👨‍💻 Code Implementation

#### 1. `/etc/inittab` (BusyBox Init)
```text
::sysinit:/etc/init.d/rcS
::askfirst:-/bin/sh
::restart:/sbin/init
::ctrlaltdel:/sbin/reboot
::shutdown:/bin/umount -a -r
```

#### 2. `/etc/init.d/rcS` (Startup Script)
```bash
#!/bin/sh
# Mount vital filesystems
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs none /dev
mkdir -p /dev/pts
mount -t devpts devpts /dev/pts

# Set hostname
hostname embedded-linux

# Bring up loopback
ifconfig lo 127.0.0.1 up

echo "System Booted Successfully!"
```
Make it executable: `chmod +x $ROOTFS/etc/init.d/rcS`

#### 3. `/etc/passwd` & `/etc/group` (Users)
Minimal user database.
```bash
# /etc/passwd
root:x:0:0:root:/root:/bin/sh
daemon:x:1:1:daemon:/usr/sbin:/bin/false

# /etc/group
root:x:0:
daemon:x:1:
```
*Note: We haven't set a password yet. Root login will be passwordless.*

---

## 💻 Implementation: Creating the Disk Image

> **Instruction:** We will pack this directory tree into a binary file formatted as ext4.

### 👨‍💻 Command Line Steps

#### Step 1: Create Empty Image
Create a 64MB file.
```bash
dd if=/dev/zero of=rootfs.ext4 bs=1M count=64
```

#### Step 2: Format as ext4
```bash
mkfs.ext4 rootfs.ext4
```

#### Step 3: Mount and Copy
We need `sudo` to mount a loopback device.
```bash
mkdir -p /tmp/mnt
sudo mount rootfs.ext4 /tmp/mnt
sudo cp -a $ROOTFS/* /tmp/mnt/
sudo umount /tmp/mnt
```

---

## 💻 Implementation: Booting from Disk

> **Instruction:** Boot QEMU, telling the kernel to use our `rootfs.ext4` as the root drive.

### 👨‍💻 Command Line Steps

```bash
qemu-system-aarch64 \
    -M virt \
    -cpu cortex-a53 \
    -nographic \
    -smp 1 \
    -m 512M \
    -kernel arch/arm64/boot/Image \
    -drive format=raw,file=rootfs.ext4,if=virtio \
    -append "console=ttyAMA0 root=/dev/vda rw"
```
*   `-drive`: Attaches `rootfs.ext4` as a virtual hard drive (`virtio`).
*   `root=/dev/vda`: Tells kernel the root FS is on the first virtio disk.
*   `rw`: Mount it Read-Write.

### Success Criteria
1.  Kernel boots.
2.  Mounts `/dev/vda` as ext4.
3.  Executes `/sbin/init`.
4.  Runs `rcS`.
5.  Drops to shell.
6.  **Persistence Test:**
    *   `echo "Hello" > /root/test.txt`
    *   `reboot`
    *   After reboot, `cat /root/test.txt`. It should still be there! (Unlike initramfs).

---

## 🔬 Lab Exercise: Lab 124.1 - Adding SSH (Dropbear)

### 1. Lab Objectives
- Cross-compile `dropbear` (lightweight SSH server).
- Install it into the RootFS.
- Generate keys and connect from host.

### 2. Step-by-Step Guide

#### Phase A: Build Dropbear
1.  Download source.
2.  `./configure --host=aarch64-linux-gnu --prefix=/usr --disable-zlib`
3.  `make`
4.  `make install DESTDIR=$ROOTFS`

#### Phase B: Configuration
1.  Create keys: `dropbearkey -t rsa -f $ROOTFS/etc/dropbear/dropbear_rsa_host_key`
2.  Add to startup (`rcS`): `dropbear -R`

#### Phase C: Networking in QEMU
1.  Add `-netdev user,id=net0,hostfwd=tcp::2222-:22 -device virtio-net-device,netdev=net0` to QEMU command.
2.  Boot.
3.  From Host: `ssh -p 2222 root@localhost`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: User Management
- **Goal:** Add a non-root user.
- **Task:**
    1.  Edit `/etc/passwd` and `/etc/group` to add user `student`.
    2.  Create home directory `/home/student`.
    3.  Set permissions: `chown -R student:student /home/student`.
    4.  Login as `student`.

### Lab 3: Switching to Systemd (Theory/Preview)
- **Goal:** Understand why we might want Systemd.
- **Task:** Research how to replace BusyBox init with Systemd. (Note: This is complex and usually done via build systems like Yocto, which we cover later).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Kernel panic - not syncing: VFS: Unable to mount root fs"
*   **Check:** Did you pass `root=/dev/vda`?
*   **Check:** Did you compile `CONFIG_VIRTIO_BLK` and `CONFIG_EXT4_FS` into the kernel? (If they are modules, the kernel can't read the disk to load them!).

#### 2. "bin/sh: not found" (even though it exists)
*   **Cause:** Missing dynamic linker (`ld-linux...`).
*   **Solution:** Check `readelf -l bin/busybox` to see the requested interpreter. Ensure that exact path exists in RootFS.

---

## ⚡ Optimization & Best Practices

### Filesystem Choice
- **Ext4:** Standard, robust, journaling (good for power loss).
- **SquashFS:** Read-only, highly compressed. Great for firmware updates.
- **UBIFS:** For raw NAND flash (handles wear leveling).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `/bin` and `/sbin`?
    *   **A:** `/bin` is for all users; `/sbin` is for system administration binaries (root).
2.  **Q:** Why do we need `devtmpfs`?
    *   **A:** It allows the kernel to automatically populate `/dev` with device nodes as hardware is detected.

### Challenge Task
> **Task:** "The Read-Only Root". Configure the system to mount the root filesystem as Read-Only (`ro` in bootargs). Create a separate partition for `/var` and mount it Read-Write so logs can still be written. This is a common production security practice.

---

## 📚 Further Reading & References
- [Filesystem Hierarchy Standard (FHS)](https://refspecs.linuxfoundation.org/FHS_3.0/fhs-3.0.html)
- [Linux From Scratch (LFS) Book](https://www.linuxfromscratch.org/lfs/)

---
