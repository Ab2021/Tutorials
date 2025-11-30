# Day 160: V4L2 Subdev Userspace API
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
1.  **Enable** the userspace API for a subdevice (`V4L2_SUBDEV_FL_HAS_DEVNODE`).
2.  **Use** `media-ctl` to configure subdevice formats (Resolution, Code).
3.  **Implement** `get_selection` and `set_selection` for cropping/scaling.
4.  **Understand** the difference between `V4L2_SUBDEV_FORMAT_TRY` and `ACTIVE`.
5.  **Debug** subdevice configuration using `v4l2-ctl --list-subdev-mbus-codes`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `media-ctl`, `v4l2-ctl`.
*   **Prior Knowledge:**
    *   Day 157 (Sensor Drivers).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why a separate API?
The main video node (`/dev/video0`) controls the DMA engine (Final Output).
But what if I want to:
*   Crop the sensor from 5MP to 1080p?
*   Tell the ISP to convert Bayer to YUV?
*   Tell the Scaler to downscale?

We need to talk to the intermediate blocks directly. That's what `/dev/v4l-subdevX` is for.

### 🔹 Part 2: The Try vs Active Context
*   **Active:** Changes the hardware registers immediately.
*   **Try:** Stores the configuration in a temporary "File Handle" state.
    *   Allows userspace to "Test" a full pipeline config (Sensor -> ISP -> DMA) without touching hardware until everything is validated.

---

## 💻 Implementation: Enabling the Node

> **Instruction:** Modify the sensor driver to expose a device node.

### 👨‍💻 Code Implementation

```c
// In Probe
sensor->sd.flags |= V4L2_SUBDEV_FL_HAS_DEVNODE;

// When registered, the core will automatically create /dev/v4l-subdevX
```

---

## 💻 Implementation: Selection API (Cropping)

> **Instruction:** Implement cropping support.

### 👨‍💻 Code Implementation

#### Step 1: Get Selection
```c
static int my_get_selection(struct v4l2_subdev *sd, struct v4l2_subdev_state *state,
                            struct v4l2_subdev_selection *sel) {
    struct my_sensor *sensor = to_my_sensor(sd);
    
    if (sel->target != V4L2_SEL_TGT_CROP) return -EINVAL;
    
    sel->r = sensor->crop_rect; // Return current crop
    return 0;
}
```

#### Step 2: Set Selection
```c
static int my_set_selection(struct v4l2_subdev *sd, struct v4l2_subdev_state *state,
                            struct v4l2_subdev_selection *sel) {
    struct my_sensor *sensor = to_my_sensor(sd);
    
    if (sel->target != V4L2_SEL_TGT_CROP) return -EINVAL;
    
    // 1. Clamp rectangle to hardware limits
    v4l2_rect_map_inside(&sel->r, &sensor->native_rect);
    
    // 2. Update State
    if (sel->which == V4L2_SUBDEV_FORMAT_ACTIVE) {
        sensor->crop_rect = sel->r;
        // Write to Hardware Registers (Window X/Y/W/H)
        regmap_write(sensor->map, REG_X_START, sel->r.left);
        // ...
    } else {
        // Update Try State (in 'state')
        struct v4l2_rect *try_crop = v4l2_subdev_get_try_crop(sd, state, sel->pad);
        *try_crop = sel->r;
    }
    
    return 0;
}

static const struct v4l2_subdev_pad_ops my_pad_ops = {
    // ...
    .get_selection = my_get_selection,
    .set_selection = my_set_selection,
};
```

---

## 🔬 Lab Exercise: Lab 160.1 - Configuring via media-ctl

### 1. Lab Objectives
- Change the sensor output resolution using `media-ctl`.

### 2. Step-by-Step Guide
1.  **Check Current:**
    ```bash
    media-ctl -p
    # pad0: Source [fmt:SRGGB10_1X10/1920x1080 field:none]
    ```
2.  **Change Format:**
    ```bash
    media-ctl -V '"my_sensor 0-0010":0 [fmt:SRGGB10_1X10/1280x720]'
    ```
    *   This calls `set_fmt` on the subdevice.
3.  **Verify:**
    ```bash
    media-ctl -p
    # pad0: Source [fmt:SRGGB10_1X10/1280x720 field:none]
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Cropping via v4l2-ctl
- **Goal:** Crop the center 640x480.
- **Task:**
    ```bash
    v4l2-ctl -d /dev/v4l-subdev0 --set-selection=target=crop,top=300,left=640,width=640,height=480
    ```
    *   Verify with `--get-selection`.

### Lab 3: Try Format
- **Goal:** Use the Try flag.
- **Task:**
    *   `media-ctl` uses ACTIVE by default.
    *   Write a small C program that opens the subdev and uses `VIDIOC_SUBDEV_S_FMT` with `which = V4L2_SUBDEV_FORMAT_TRY`.
    *   Verify that the hardware registers (via debugfs or active get) did NOT change.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Inappropriate ioctl for device"
*   **Cause:** Trying to use `VIDIOC_S_FMT` (Video Node API) on a Subdevice Node.
*   **Fix:** Subdevices use `VIDIOC_SUBDEV_S_FMT`. Tools like `v4l2-ctl` handle this automatically if you use `-d /dev/v4l-subdevX`.

#### 2. Crop rectangle ignored
*   **Cause:** Driver `set_selection` logic is missing or doesn't update hardware.
*   **Cause:** `v4l2_rect_map_inside` modified the request because it was out of bounds.

---

## ⚡ Optimization & Best Practices

### `v4l2_subdev_get_try_format`
*   Helper to access the `try` state in the `state` container.
*   Always use this instead of manually managing try buffers.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `V4L2_SEL_TGT_COMPOSE`?
    *   **A:** It defines the window *inside* the destination buffer where the image is written. Used for scaling (Source Crop -> Sink Compose).
2.  **Q:** Can I use `v4l2-ctl --set-fmt-video` on a subdevice?
    *   **A:** No. `--set-fmt-video` is for `/dev/videoX`. For subdevices, use `--set-subdev-fmt` or `media-ctl`.

### Challenge Task
> **Task:** "The Digital Zoom".
> *   Implement a control that modifies the Crop Rectangle.
> *   When the user slides "Zoom" to 2x, set the crop to the center 50% of the sensor.

---

## 📚 Further Reading & References
- [Kernel Documentation: userspace-api/media/v4l/dev-subdev.rst](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/dev-subdev.html)

---
