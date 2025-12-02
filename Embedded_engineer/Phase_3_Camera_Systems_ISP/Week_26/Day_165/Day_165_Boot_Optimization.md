# Day 165: Boot Time Optimization
## Phase 3: Camera Systems & ISP | Week 26: Performance Optimization & Power

---

## 🎯 Learning Objectives
1.  **Analyze** the Linux Boot Process: Bootloader (U-Boot) -> Kernel -> Init (Systemd) -> Application.
2.  **Measure** Boot Time using `systemd-analyze` and `bootchart`.
3.  **Optimize** U-Boot: Remove delay, silent boot, Falcon Mode.
4.  **Optimize** Kernel: Remove unused drivers, use LZ4 compression.
5.  **Optimize** Userspace: Parallelize services, Early Camera (EVS).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Jetson/Pi.
*   **Software:** `systemd-analyze`, Serial Console (UART).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Boot Chain
1.  **ROM Code:** Hardcoded in SoC. Loads Bootloader.
2.  **Bootloader (U-Boot):** Initializes RAM, loads Kernel.
3.  **Kernel:** Initializes Drivers, mounts RootFS.
4.  **Init (Systemd):** Starts services (Network, GUI, Camera App).

### 🔹 Part 2: Measurement
*   **Total Time:** From Power Button to "First Frame".
*   **Breakdown:**
    *   U-Boot: 2-5s.
    *   Kernel: 2-10s.
    *   Userspace: 5-20s.

### 🔹 Part 3: Strategies
*   **Remove:** If you don't need USB/HDMI/WiFi, disable them.
*   **Defer:** Start the Camera App *before* the Network.
*   **Compress:** LZ4 decompresses faster than GZIP.

---

## 💻 Implementation Examples

### Example 1: Analyzing Boot Time

```bash
# 1. Overall Stats
systemd-analyze
# Startup finished in 3.4s (kernel) + 5.2s (userspace) = 8.6s

# 2. Blame (Find slow services)
systemd-analyze blame
# 2.1s nv-l4t-bootloader-config.service
# 1.5s NetworkManager.service
# ...

# 3. Critical Chain (Tree view)
systemd-analyze critical-chain
```

### Example 2: Optimizing U-Boot (`extlinux.conf`)

Reduce the wait time.

```text
TIMEOUT 1  # Was 30 (3 seconds)
DEFAULT primary

LABEL primary
      MENU LABEL primary kernel
      LINUX /boot/Image
      INITRD /boot/initrd
      APPEND ${cbootargs} quiet root=/dev/mmcblk0p1 rw rootwait
```

### Example 3: Creating a Systemd Service for Fast Camera

Start early, before network.

```ini
[Unit]
Description=Fast Camera Service
DefaultDependencies=no  # Don't wait for basic.target
After=local-fs.target   # Wait only for disk mount

[Service]
ExecStart=/usr/bin/python3 /opt/camera/main.py
Restart=always
User=root

[Install]
WantedBy=sysinit.target # Start very early
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Kernel Config Diet

**Objective:** Shrink the kernel.

**Steps:**
1.  Clone kernel source.
2.  `make menuconfig`.
3.  **Disable:** Sound, Joystick, Printer, IPv6 (if not needed), Debugging (Ftrace, KGDB).
4.  **Change:** Compression from GZIP to LZ4.
5.  **Compile & Install.**
6.  **Measure:** Kernel boot time should drop by ~30%.

### Lab 2: U-Boot "Falcon Mode" (Advanced)

**Objective:** Skip U-Boot full initialization.

**Steps:**
1.  SPL (Secondary Program Loader) loads the Kernel directly, bypassing U-Boot proper.
2.  Requires saving `args` to a specific memory location.
3.  **Result:** Saves ~2 seconds.

### Lab 3: Early Frame Capture

**Objective:** Show image before GUI.

**Steps:**
1.  Modify your app to write directly to the Framebuffer (`/dev/fb0`) or DRM KMS.
2.  Don't wait for X11 or Wayland.
3.  **Result:** "Splash Screen" effect with live video.

---

## 🐛 Debugging Boot Issues

### Debug 1: "Kernel Panic - VFS: Unable to mount root"

**Symptom:** Boot loop.

**Cause:**
*   You removed the driver for the SD Card (MMC) or Filesystem (EXT4) from the kernel.
*   **Fix:** Make sure storage drivers are built-in (`y`), not modules (`m`).

### Debug 2: Service Fails to Start

**Symptom:** Camera app crashes on boot, but runs fine manually.

**Cause:**
*   Dependency missing (e.g., `/dev/video0` not ready yet).
*   **Fix:** Add `After=dev-video0.device` in systemd unit. Or add a retry loop in python script.

---

## ⚡ Performance Optimization

### Optimization 1: Static IP

*   DHCP takes time (Negotiation).
*   Use a Static IP to save 2-5 seconds on network startup.

### Optimization 2: Initramfs

*   If your RootFS is on a slow HDD/USB, use a small Initramfs to load drivers and mount it.
*   If RootFS is on fast eMMC, you might not need Initramfs at all (Direct Boot). Removing it saves load time.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "init" (PID 1)?** (The first process started by the kernel. It starts everything else).
2.  **Difference between `systemd` and `SysVinit`?** (Systemd starts services in parallel. SysVinit is sequential).
3.  **What is "XIP" (Execute In Place)?** (Running code directly from Flash without copying to RAM. Saves copy time but slower execution).

### Practical Challenges

1.  **Bootchart:** Generate a bootchart SVG. Identify the "Longest Bar". Optimize it.
2.  **2-Second Boot:** Try to reach the "2-Second Boot" milestone required for Automotive Rear View Cameras (FMVSS 111).

---

## 📚 Further Reading & Resources

### Documentation
*   **"Booting Linux in 1 Second" (Presentation by Jan Altenberg).**
*   **Systemd Optimization Guide.**

---

## 🎓 Summary

Today we covered:
- ✅ **Boot Chain:** The sequence of events.
- ✅ **Measurement:** `systemd-analyze`.
- ✅ **U-Boot:** Reducing timeout.
- ✅ **Kernel:** Removing bloat.
- ✅ **Userspace:** Parallel execution.

**Next:** Day 166 - Memory Optimization (CMA/ION).

---

**Day 165 Complete** | Phase 3: Camera Systems & ISP | Week 26: Performance Optimization & Power


