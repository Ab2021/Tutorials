# Day 72: Fast Boot & Latency Optimization
## Phase 3: Camera Systems & ISP | Week 13: Optimization & Final Project

---

## 🎯 Learning Objectives
1.  **Define** Key Performance Indicators (KPIs): Boot Time (< 2s) and Latency (< 100ms).
2.  **Analyze** the Linux Boot Process: Bootloader -> Kernel -> Init -> App.
3.  **Optimize** Kernel Boot: Reducing size, removing drivers, "Quiet" mode.
4.  **Optimize** Userspace: Parallelizing services, Early Camera Service.
5.  **Reduce** Glass-to-Glass Latency: Zero-Copy pipelines, DMA-BUF, High-Priority Threads.
6.  **Measure** Latency using LED/Photodiode or High-Speed Camera.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** High-speed camera (e.g., iPhone Slo-Mo) or LED Latency Tester.
*   **Software:** `bootchart`, `systemd-analyze`, `perf`.
*   **Knowledge:** Linux Init Systems (Systemd/SysVinit), Real-Time Scheduling.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Boot Timeline
1.  **Power On Reset (POR):** Hardware stabilization.
2.  **BootROM:** Loads SPL/U-Boot.
3.  **Bootloader (U-Boot):** Initializes DDR, loads Kernel. (Optimization: Falcon Mode).
4.  **Kernel:** Probes drivers. (Optimization: Deferred Probe, Built-in drivers).
5.  **Init (Systemd/Android Init):** Starts services. (Optimization: Dependency graph).
6.  **Application:** Shows first frame.

### 🔹 Part 2: Latency Sources
*   **Exposure:** 33ms (at 30fps).
*   **Readout:** Rolling shutter time (e.g., 10-30ms).
*   **ISP Processing:** Demosaic, NR (1-2 frames).
*   **Transport:** MIPI/DMA (negligible).
*   **Display:** VSYNC wait (16-33ms).
*   **Total:** Often > 100ms without optimization.

### 🔹 Part 3: Zero-Copy Pipeline
*   **Bad:** Sensor -> RAM (memcpy) -> ISP -> RAM (memcpy) -> Display.
*   **Good:** Sensor -> RAM (DMA-BUF) -> ISP (DMA-BUF) -> Display (DMA-BUF).
*   The CPU never touches the pixel data. Only passes pointers (File Descriptors).

---

## 💻 Implementation Examples

### Example 1: Measuring Boot Time (Systemd)

```bash
# 1. Overall Time
systemd-analyze time
# Output: Kernel: 1.5s, Userspace: 2.0s

# 2. Blame (Who is slow?)
systemd-analyze blame
# Output:
# 1.2s networking.service
# 0.8s camera.service

# 3. Critical Chain (Dependencies)
systemd-analyze critical-chain
```

### Example 2: Optimizing U-Boot (bootdelay)

In `u-boot.env` or `config`:

```bash
# Don't wait for user input
bootdelay=0

# Silent console (Speeds up boot by not printing text)
silent=1
```

### Example 3: High Priority Thread (C++)

Ensuring the Camera Thread is not preempted by background tasks.

```cpp
#include <pthread.h>
#include <sched.h>

void set_realtime_priority() {
    struct sched_param param;
    param.sched_priority = 90; // High priority (1-99)
    
    // SCHED_FIFO: First-In, First-Out Real-Time policy
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
        perror("Failed to set RT priority");
    }
}
```

### Example 4: Kernel Command Line Optimization

In `bootargs`:

```text
quiet loglevel=0 lpj=... rootfstype=ext4 no_console_suspend
```
*   `quiet`: Suppress printk.
*   `lpj`: Preset "Loops Per Jiffy" (skips calibration).

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Bootchart Analysis

**Objective:** Visualize the boot.

**Steps:**
1.  Install `systemd-bootchart`.
2.  Add `init=/usr/lib/systemd/systemd-bootchart` to kernel cmdline.
3.  Reboot.
4.  Check `/run/log/bootchart.svg`.
5.  **Task:** Identify the longest "bar" and disable that service if not needed (e.g., `ModemManager` on a WiFi-only device).

### Lab 2: Glass-to-Glass Latency Measurement

**Objective:** Measure reality.

**Steps:**
1.  Setup: Camera pointing at a Stopwatch (running on PC screen).
2.  Display: Camera feed shown on a Monitor next to the PC screen.
3.  Capture: Take a photo of both screens with a smartphone.
4.  **Calc:** $T_{monitor} - T_{pc} = Latency$.
5.  **Goal:** < 100ms.

### Lab 3: Reducing Frame Buffering

**Objective:** Trade smoothness for speed.

**Steps:**
1.  In the Camera App, check the "Queue Size".
2.  Default might be 3-4 buffers (to absorb jitter).
3.  Reduce to 2 buffers (Double Buffering).
4.  **Result:** Latency drops by 33ms per buffer removed.
5.  **Risk:** Tearing or dropped frames if processing spikes.

---

## 🐛 Debugging Performance Issues

### Debug 1: Jitter / Stutter

**Symptom:** Average FPS is 30, but video looks jerky.

**Cause:**
*   Garbage Collection (Java).
*   CPU Frequency Scaling (Governor switching to low freq).
*   **Fix:** Set CPU Governor to `performance`.
    ```bash
    echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
    ```

### Debug 2: Slow Kernel Boot

**Symptom:** Kernel takes 5s to load.

**Cause:**
*   Probing unused drivers (USB, Audio, Ethernet) sequentially.
*   **Fix:** Compile a custom kernel (`make menuconfig`). Remove everything not needed for the camera. Use `CONFIG_MODULES=n` for a monolithic (faster) kernel if possible.

---

## ⚡ Performance Optimization

### Optimization 1: Falcon Mode (U-Boot)

*   U-Boot usually loads the Kernel.
*   **Falcon Mode:** SPL (Secondary Program Loader) loads the Kernel *directly*, skipping the full U-Boot stage.
*   Saves ~500ms.

### Optimization 2: Early Camera (Userspace)

*   Don't wait for the full Android/Desktop UI.
*   Start a simple `camera_service` binary from `init.rc` immediately after filesystem mount.
*   Draw directly to the Framebuffer (`/dev/fb0`) or DRM Plane.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "Glass-to-Glass" latency?** (Photon hitting sensor -> Photon leaving display).
2.  **Why does `printk` slow down boot?** (Serial console is slow, e.g., 115200 baud. Blocking writes delay the CPU).
3.  **Difference between `SCHED_FIFO` and `SCHED_RR`?** (FIFO runs until it yields/blocks; RR has time slices).
4.  **What is "Deferred Probe"?** (Driver waits for dependencies, e.g., Regulator, to be ready).

### Practical Challenges

1.  **Implement "Quiet Boot":** Modify bootloader and kernel args to show *nothing* on the screen until the Camera App appears. (Professional look).
2.  **Create a Latency Histogram:** Modify the app to log the time difference between "Frame Capture" timestamp and "Frame Render" timestamp for 1000 frames. Plot the distribution.

---

## 📚 Further Reading & Resources

### Documentation
*   **Bootlin: Embedded Linux Boot Time Optimization Course.**
*   **Real-Time Linux Wiki.**

---

## 🎓 Summary

Today we covered:
- ✅ **Boot:** Timeline analysis.
- ✅ **Latency:** Sources and fixes.
- ✅ **Tools:** Bootchart, Systemd-analyze.
- ✅ **Kernel:** Stripping it down.
- ✅ **Scheduling:** Real-time priority.

**Next:** Day 73 - Thermal Management & Reliability.

---

**Day 72 Complete** | Phase 3: Camera Systems & ISP | Week 13: Optimization & Final Project
