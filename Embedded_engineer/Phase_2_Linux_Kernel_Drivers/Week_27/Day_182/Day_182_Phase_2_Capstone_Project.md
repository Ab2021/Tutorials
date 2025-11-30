# Day 182: Phase 2 Capstone Project - The Universal IoT Gateway
## Phase 2: Linux Kernel & Device Drivers | Week 27: Phase 2 Conclusion

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
1.  **Implement** a complex, multi-subsystem Linux Kernel Driver.
2.  **Integrate** Char, Block, Net, USB, I2C, SPI, V4L2, and ALSA concepts.
3.  **Debug** system-wide issues (concurrency, memory leaks, race conditions).
4.  **Document** the driver architecture and usage.
5.  **Transition** from Kernel development to Embedded Android (Phase 3).

---

## 🛠️ Project Specification: The Universal IoT Gateway

### 1. Overview
The goal is to build a monolithic (or multi-module) driver that manages a hypothetical IoT Gateway board. The board has sensors, a camera, a speaker, and a network uplink.

### 2. Architecture
*   **Module Name:** `iot_gateway.ko`
*   **Device Tree Node:** `/soc/iot-gateway`

### 3. Subsystem Requirements

#### A. The Sensor Hub (I2C/Char)
*   **Hardware:** Virtual I2C Sensor at 0x40.
*   **Function:** Reads Temperature (Reg 0x00) and Humidity (Reg 0x01).
*   **Interface:**
    *   `/dev/iot_sensors` (Char Device).
    *   `ioctl(fd, IOT_GET_TEMP, &val)`
    *   `ioctl(fd, IOT_GET_HUM, &val)`
    *   Sysfs: `/sys/class/iot/sensors/temp` (Read-only).

#### B. The Camera (V4L2)
*   **Hardware:** Virtual Sensor.
*   **Function:** Generates a video stream.
    *   **Normal Mode:** Color Bars.
    *   **Alert Mode:** Flashing Red/Black (Triggered when Temp > 50C).
*   **Interface:** `/dev/video0`.

#### C. The Audio Alarm (ALSA)
*   **Hardware:** Virtual PCM.
*   **Function:** Plays audio.
    *   **Normal Mode:** Silent.
    *   **Alert Mode:** Generates a Siren tone.
*   **Interface:** ALSA Card "IoTAlarm".

#### D. The Network Uplink (Netdev)
*   **Hardware:** Virtual Ethernet.
*   **Function:**
    *   Interface `iot0`.
    *   Accepts UDP packets on Port 9999.
    *   Packet Format: `CMD:VALUE` (e.g., `SET_THRESH:50`).
    *   Updates the kernel's internal threshold variable.

### 4. The Glue Logic (Workqueues/Timers)
*   **Poller:** A kernel timer runs every 1 second.
    *   Reads I2C Sensor.
    *   Checks if Temp > Threshold.
    *   If Yes -> Sets "Alert Mode" flag.
    *   If No -> Clears "Alert Mode" flag.
*   **Alert Handler:**
    *   When Alert Mode toggles, notify V4L2 and ALSA components to switch their generation logic.

---

## 💻 Implementation Guide

### 🔹 Step 1: The Platform Driver
Start by creating the skeleton `platform_driver`.
Parse the Device Tree to get I2C addresses and configuration.

### 🔹 Step 2: The Sensor Sub-Module
Implement the I2C client and the Char Device.
Test with `cat /sys/class/iot/sensors/temp`.

### 🔹 Step 3: The Logic
Implement the Timer and the Threshold logic.
Add `pr_info("ALERT! Temp: %d\n", temp)` to verify.

### 🔹 Step 4: The Multimedia
Add the V4L2 and ALSA drivers.
Link them to the global "Alert Mode" flag.
*   V4L2 `fill_buffer`: `if (alert) fill_red(); else fill_bars();`
*   ALSA `timer_callback`: `if (alert) generate_siren(); else generate_silence();`

### 🔹 Step 5: The Network
Add the Netdev.
Implement `ndo_start_xmit` to parse outgoing packets (or just RX hook to parse incoming).
Actually, for "Control", a Netdev might be overkill, but it's a requirement to demonstrate networking skills. A better approach might be a simple UDP socket listener in kernel (rare) or just using the Netdev to receive raw frames. Let's stick to **Netdev RX Handler**:
*   Intercept packets sent to `iot0`.
*   Parse payload.

---

## 📈 Grading Rubric

| Criteria | Points | Description |
| :--- | :--- | :--- |
| **Compilation** | 10 | Compiles without warnings. |
| **Loading** | 10 | Loads/Unloads cleanly. |
| **Sensors** | 20 | I2C + Char Device work. |
| **Multimedia** | 30 | Camera/Audio react to alert. |
| **Network** | 20 | Can set threshold via network. |
| **Integration** | 10 | All parts work together. |

---

## 🏁 Conclusion of Phase 2

You have mastered the Linux Kernel.
You understand how the OS talks to hardware.
You are ready for **Phase 3: Embedded Android**, where we will build the userspace layers that sit on top of these drivers.

**See you in Phase 3!**

---
