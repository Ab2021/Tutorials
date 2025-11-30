# Day 162: Week 23 Review & Project - The Virtual Camera Pipeline
## Phase 2: Linux Kernel & Device Drivers | Week 23: V4L2 Subdevices

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
1.  **Synthesize** Week 23 concepts (Subdevices, Media Controller, Async Probe).
2.  **Architect** a multi-driver camera system (Sensor -> ISP -> Bridge).
3.  **Implement** format propagation logic across the entire pipeline.
4.  **Demonstrate** runtime reconfiguration using `media-ctl`.
5.  **Debug** pipeline validation errors.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `media-ctl`, `v4l2-ctl`.
*   **Prior Knowledge:**
    *   Week 23 Content.

---

## 🔄 Week 23 Review

### 1. Subdevices (Day 156)
*   **Struct:** `v4l2_subdev`.
*   **Ops:** `video`, `pad`, `core`.
*   **Registration:** `v4l2_device_register_subdev`.

### 2. Sensor Drivers (Day 157)
*   **Role:** Generate data.
*   **Controls:** Exposure, Gain.
*   **Format:** `MEDIA_BUS_FMT`.

### 3. Media Controller (Day 158)
*   **Graph:** Entities, Pads, Links.
*   **Validation:** `link_validate`.

### 4. Async Framework (Day 159)
*   **Problem:** Probe order.
*   **Solution:** `v4l2_async_notifier`.

### 5. Userspace API (Day 160)
*   **Node:** `/dev/v4l-subdevX`.
*   **IOCTLs:** `SUBDEV_S_FMT`, `SUBDEV_S_SELECTION`.

### 6. ISP (Day 161)
*   **Role:** Process data (Bayer -> YUV).
*   **Logic:** Format propagation.

---

## 🛠️ Project: The "VirtPipeline"

### 📋 Project Requirements
Create a suite of 3 drivers:
1.  **`virt_sensor.ko`**: Generates Raw Bayer (Pattern).
2.  **`virt_isp.ko`**: Converts Bayer to YUYV (Simulation).
3.  **`virt_bridge.ko`**: DMA Engine, registers `/dev/video0`.

**Features:**
*   **Topology:** Sensor (Pad 0) -> (Pad 0) ISP (Pad 1) -> (Pad 0) Bridge.
*   **Configurable:** Change Sensor resolution, ISP follows.
*   **Async:** Bridge waits for Sensor and ISP.

---

## 💻 Implementation: Step-by-Step Guide

### 🔹 Phase 1: The Sensor (`virt_sensor.c`)
*   **Pads:** 1 Source.
*   **Formats:** `SRGGB10`.
*   **Async:** `v4l2_async_register_subdev`.

### 🔹 Phase 2: The ISP (`virt_isp.c`)
*   **Pads:** 1 Sink, 1 Source.
*   **Logic:**
    *   `set_fmt(pad=0)`: Updates Sink format. Resets Source to YUYV with same dims.
    *   `set_fmt(pad=1)`: Only allows YUYV.
*   **Async:** `v4l2_async_register_subdev`.

### 🔹 Phase 3: The Bridge (`virt_bridge.c`)
*   **Pads:** 1 Sink.
*   **Notifier:** Waits for "Sensor" and "ISP".
*   **Link Creation:**
    *   In `notify_complete`, create links:
    *   `media_create_pad_link(sensor, 0, isp, 0, ...)`
    *   `media_create_pad_link(isp, 1, bridge, 0, ...)`

### 🔹 Phase 4: Testing Script

**`test_pipeline.sh`**
```bash
#!/bin/bash

# 1. Load Modules
insmod virt_bridge.ko
insmod virt_isp.ko
insmod virt_sensor.ko

# 2. Verify Topology
media-ctl -p

# 3. Configure Pipeline (1280x720)
# Sensor Out
media-ctl -V '"virt_sensor":0 [fmt:SRGGB10_1X10/1280x720]'
# ISP In (Should auto-propagate to Out)
media-ctl -V '"virt_isp":0 [fmt:SRGGB10_1X10/1280x720]'

# 4. Verify ISP Out
media-ctl -p -e "virt_isp"
# Expected: pad1: Source [fmt:YUYV8_2X8/1280x720]

# 5. Capture
v4l2-ctl -d /dev/video0 --set-fmt-video=width=1280,height=720,pixelformat=YUYV --stream-mmap --stream-count=10
```

---

## 📈 Grading Rubric

| Criteria | Excellent (A) | Good (B) | Needs Improvement (C) |
| :--- | :--- | :--- | :--- |
| **Architecture** | Clean separation of 3 modules. | Logic mixed in one file. | Monolithic driver. |
| **Async** | Correctly handles load order. | Crashes if sensor loaded late. | No async support. |
| **Propagation** | Changing Sensor fmt updates ISP. | ISP fmt fixed. | Formats don't match. |

---

## 🔮 Looking Ahead: Phase 3
Congratulations! You have completed **Phase 2: Linux Kernel & Device Drivers**.
Next, we move to **Phase 3: Embedded Android (AOSP)**.
*   Building AOSP.
*   HAL (Hardware Abstraction Layer).
*   Binder IPC.
*   Native Services.

---
