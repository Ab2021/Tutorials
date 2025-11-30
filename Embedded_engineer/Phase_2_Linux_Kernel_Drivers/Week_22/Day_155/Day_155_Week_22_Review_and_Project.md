# Day 155: Week 22 Review & Project - The Virtual Webcam
## Phase 2: Linux Kernel & Device Drivers | Week 22: V4L2 Subsystem Basics

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
1.  **Synthesize** Week 22 concepts (V4L2, IOCTLs, Controls, Formats, Buffers).
2.  **Architect** a complete Video Capture Driver from scratch.
3.  **Implement** a dynamic test pattern generator (Bars, Static, Moving).
4.  **Integrate** `v4l2_ctrl` to switch patterns at runtime.
5.  **Verify** the driver using standard tools (`v4l2-ctl`, `ffplay`, `qv4l2`).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `qv4l2` (GUI testing tool).
*   **Prior Knowledge:**
    *   Week 22 Content.

---

## 🔄 Week 22 Review

### 1. Architecture (Day 149)
*   **V4L2 Device:** The parent.
*   **Video Device:** The node (`/dev/videoX`).
*   **Media Controller:** The graph.

### 2. IOCTLs (Day 150)
*   **Dispatcher:** `video_ioctl2`.
*   **Ops:** `querycap`, `enum_input`.

### 3. Controls (Day 151)
*   **Handler:** `v4l2_ctrl_handler`.
*   **Types:** Integer, Boolean, Menu.

### 4. Formats (Day 152)
*   **Negotiation:** Enum, Try, Set, Get.
*   **Struct:** `v4l2_format`.

### 5. Buffers (Day 153-154)
*   **Framework:** `videobuf2` (vb2).
*   **Memory:** `vmalloc` (Virtual), `dma_contig` (Physical).
*   **Loop:** Queue -> Fill -> Done.

---

## 🛠️ Project: The "VirtCam" Driver

### 📋 Project Requirements
Create a driver `virtcam` that:
1.  **Registers** `/dev/videoX`.
2.  **Supports** 640x480 YUYV.
3.  **Generates** 3 Patterns (Selectable via Control):
    *   Color Bars.
    *   Moving Ball.
    *   Noise (Random static).
4.  **Controls:**
    *   Brightness (Modifies pixel intensity).
    *   Pattern Select (Menu).
5.  **Frame Rate:** 30 FPS.

---

## 💻 Implementation: Step-by-Step Guide

### 🔹 Phase 1: Header & Structs

**`virtcam.h`**
```c
#ifndef VIRTCAM_H
#define VIRTCAM_H

#include <linux/module.h>
#include <linux/platform_device.h>
#include <media/v4l2-device.h>
#include <media/v4l2-dev.h>
#include <media/v4l2-ctrls.h>
#include <media/videobuf2-v4l2.h>
#include <media/videobuf2-vmalloc.h>

#define PATTERN_BARS   0
#define PATTERN_BALL   1
#define PATTERN_NOISE  2

struct virtcam_dev {
    struct v4l2_device v4l2_dev;
    struct video_device vdev;
    struct v4l2_ctrl_handler ctrl_hdl;
    struct vb2_queue vb_vidq;
    
    struct mutex lock;      // IOCTL serialization
    spinlock_t slock;       // Buffer list protection
    struct list_head buf_list;
    
    struct hrtimer timer;
    int sequence;
    
    // State
    struct v4l2_format fmt;
    int pattern_mode;
    int brightness;
};

#endif
```

### 🔹 Phase 2: Pattern Generator

**`virtcam_gen.c`**
```c
#include "virtcam.h"
#include <linux/random.h>

void virtcam_fill_buffer(struct virtcam_dev *dev, struct vb2_v4l2_buffer *vbuf) {
    struct vb2_buffer *vb = &vbuf->vb2_buf;
    u8 *ptr = vb2_plane_vaddr(vb, 0);
    int width = dev->fmt.fmt.pix.width;
    int height = dev->fmt.fmt.pix.height;
    int i, j;
    u8 y, u, v;
    
    // Apply Brightness Offset (0-255, center 128)
    int b_offset = dev->brightness - 128;

    if (dev->pattern_mode == PATTERN_NOISE) {
        get_random_bytes(ptr, width * height * 2);
        return;
    }

    for (j = 0; j < height; j++) {
        for (i = 0; i < width; i += 2) {
            if (dev->pattern_mode == PATTERN_BARS) {
                // ... (Bar logic from Day 154) ...
                y = (i < width/2) ? 200 : 50;
                u = 128; v = 128;
            } else {
                // Ball Logic
                int dx = i - (dev->sequence % width);
                int dy = j - (height/2);
                if (dx*dx + dy*dy < 2500) { // Radius 50
                    y = 235; u = 128; v = 128; // White Ball
                } else {
                    y = 16; u = 128; v = 128; // Black BG
                }
            }
            
            // Apply Brightness
            int y_val = y + b_offset;
            if (y_val < 0) y_val = 0;
            if (y_val > 255) y_val = 255;
            
            ptr[0] = y_val;
            ptr[1] = u;
            ptr[2] = y_val;
            ptr[3] = v;
            ptr += 4;
        }
    }
}
```

### 🔹 Phase 3: Controls & IOCTLs

**`virtcam_core.c`**
```c
#include "virtcam.h"

// Control Callback
static int virtcam_s_ctrl(struct v4l2_ctrl *ctrl) {
    struct virtcam_dev *dev = container_of(ctrl->handler, struct virtcam_dev, ctrl_hdl);
    
    switch (ctrl->id) {
    case V4L2_CID_BRIGHTNESS:
        dev->brightness = ctrl->val;
        break;
    case V4L2_CID_TEST_PATTERN:
        dev->pattern_mode = ctrl->val;
        break;
    default:
        return -EINVAL;
    }
    return 0;
}

static const struct v4l2_ctrl_ops virtcam_ctrl_ops = {
    .s_ctrl = virtcam_s_ctrl,
};

static const char * const pattern_menu[] = {
    "Color Bars",
    "Bouncing Ball",
    "Static Noise",
    NULL
};

// Probe
static int virtcam_probe(struct platform_device *pdev) {
    struct virtcam_dev *dev;
    int ret;
    
    dev = devm_kzalloc(&pdev->dev, sizeof(*dev), GFP_KERNEL);
    // ... Init Locks, Lists, V4L2 Device ...
    
    // Controls
    v4l2_ctrl_handler_init(&dev->ctrl_hdl, 2);
    v4l2_ctrl_new_std(&dev->ctrl_hdl, &virtcam_ctrl_ops, 
                      V4L2_CID_BRIGHTNESS, 0, 255, 1, 128);
    v4l2_ctrl_new_std_menu(&dev->ctrl_hdl, &virtcam_ctrl_ops,
                           V4L2_CID_TEST_PATTERN, 2, 0, 0, pattern_menu);
    
    dev->v4l2_dev.ctrl_handler = &dev->ctrl_hdl;
    
    // ... Init Queue (vb2), Video Device ...
    
    return 0;
}
```

---

## 💻 Implementation: Testing

> **Instruction:** Compile and Load.

### 👨‍💻 Command Line Steps

1.  **Load:** `insmod virtcam.ko`
2.  **Verify Controls:**
    ```bash
    v4l2-ctl -d /dev/video0 --list-ctrls
    # brightness (int) : min=0 max=255 ...
    # test_pattern (menu) : min=0 max=2 ...
    ```
3.  **Set Pattern:**
    ```bash
    v4l2-ctl -d /dev/video0 --set-ctrl test_pattern=1
    ```
4.  **View:**
    ```bash
    ffplay -f v4l2 /dev/video0
    ```
    *   You should see a ball moving.
5.  **Adjust Brightness:**
    *   While `ffplay` is running, open another terminal.
    *   `v4l2-ctl -d /dev/video0 --set-ctrl brightness=200`
    *   The ball should get brighter instantly.

---

## 📈 Grading Rubric

| Criteria | Excellent (A) | Good (B) | Needs Improvement (C) |
| :--- | :--- | :--- | :--- |
| **Stability** | No crashes on load/unload or rapid streaming toggle. | Occasional warning in dmesg. | Kernel Panic. |
| **Functionality** | All patterns work. Brightness works. | Patterns static. | Black screen. |
| **Compliance** | Passes `v4l2-compliance` tool. | Fails minor checks. | Fails major checks. |

---

## 🔮 Looking Ahead: Week 23
Next week, we dive into **V4L2 Subdevices**.
*   We will separate the "Sensor" from the "Bridge".
*   We will write a driver for a real I2C Camera Sensor.
*   We will use Media Controller to link them.

---
