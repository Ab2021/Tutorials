# Day 157: Camera Sensor Drivers
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
1.  **Develop** a V4L2 Subdevice driver for a raw camera sensor (e.g., OV7670 or IMX219).
2.  **Implement** `s_stream` to start/stop the sensor output.
3.  **Implement** `get_fmt` and `set_fmt` pad operations.
4.  **Expose** sensor controls (Exposure, Gain, Test Pattern) via `v4l2_ctrl_handler`.
5.  **Debug** I2C register sequences for sensor initialization.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 156 (Subdevices).
    *   Day 143 (I2C Drivers).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Sensor Architecture
A typical raw sensor (Bayer) has:
*   **I2C Interface:** For configuration (Resolution, FPS, Gain).
*   **Parallel/CSI Interface:** For video data output.
*   **Registers:** Thousands of them. Usually provided as "Magic Sequences" by the vendor.

### 🔹 Part 2: Pad Operations
Unlike video nodes (`/dev/video0`), subdevices negotiate formats on **Pads**.
*   `get_fmt(pad, fmt)`: What format is this pad outputting?
*   `set_fmt(pad, fmt)`: Change the format (e.g., crop, binning).
*   `enum_mbus_code`: List supported Media Bus formats (e.g., `MEDIA_BUS_FMT_SBGGR10_1X10`).

---

## 💻 Implementation: The Sensor Driver

> **Instruction:** We will build a driver for a hypothetical "MySensor".

### 👨‍💻 Code Implementation

#### Step 1: Pad Operations
```c
static int my_get_fmt(struct v4l2_subdev *sd, struct v4l2_subdev_state *state,
                      struct v4l2_subdev_format *format) {
    struct my_sensor *sensor = to_my_sensor(sd);
    
    if (format->pad != 0) return -EINVAL;
    
    // Return current format
    format->format.code = MEDIA_BUS_FMT_SRGGB10_1X10;
    format->format.width = 1920;
    format->format.height = 1080;
    format->format.field = V4L2_FIELD_NONE;
    
    return 0;
}

static int my_set_fmt(struct v4l2_subdev *sd, struct v4l2_subdev_state *state,
                      struct v4l2_subdev_format *format) {
    // Validate and update internal state
    // (Simplified: We only support 1080p)
    format->format.code = MEDIA_BUS_FMT_SRGGB10_1X10;
    format->format.width = 1920;
    format->format.height = 1080;
    return 0;
}

static const struct v4l2_subdev_pad_ops my_pad_ops = {
    .get_fmt = my_get_fmt,
    .set_fmt = my_set_fmt,
};
```

#### Step 2: Video Operations (Streaming)
```c
static int my_s_stream(struct v4l2_subdev *sd, int enable) {
    struct my_sensor *sensor = to_my_sensor(sd);
    struct i2c_client *client = sensor->client;
    
    if (enable) {
        // Write Regs to Start Streaming
        regmap_write(sensor->map, 0x0100, 0x01); // MODE_STREAMING
    } else {
        // Write Regs to Stop
        regmap_write(sensor->map, 0x0100, 0x00); // MODE_STANDBY
    }
    return 0;
}

static const struct v4l2_subdev_video_ops my_video_ops = {
    .s_stream = my_s_stream,
};
```

#### Step 3: Controls (Exposure/Gain)
```c
static int my_s_ctrl(struct v4l2_ctrl *ctrl) {
    struct my_sensor *sensor = container_of(ctrl->handler, struct my_sensor, ctrl_hdl);
    
    switch (ctrl->id) {
    case V4L2_CID_EXPOSURE:
        // Write Exposure Regs (High/Low bytes)
        regmap_write(sensor->map, 0x0202, ctrl->val >> 8);
        regmap_write(sensor->map, 0x0203, ctrl->val & 0xFF);
        break;
    case V4L2_CID_GAIN:
        regmap_write(sensor->map, 0x0205, ctrl->val);
        break;
    }
    return 0;
}
```

#### Step 4: Ops Structure
```c
static const struct v4l2_subdev_ops my_subdev_ops = {
    .video = &my_video_ops,
    .pad = &my_pad_ops,
};
```

---

## 🔬 Lab Exercise: Lab 157.1 - Subdev IOCTLs

### 1. Lab Objectives
- Load the driver.
- Use `v4l2-ctl` to talk to the subdevice directly.

### 2. Step-by-Step Guide
1.  **Find Subdev Node:**
    ```bash
    ls /dev/v4l-subdev*
    ```
2.  **Get Format:**
    ```bash
    v4l2-ctl -d /dev/v4l-subdev0 --get-subdev-fmt
    # Output: 1920x1080, Code: SRGGB10
    ```
3.  **Set Control:**
    ```bash
    v4l2-ctl -d /dev/v4l-subdev0 --set-ctrl exposure=1000
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Test Pattern Generator
- **Goal:** Enable the sensor's internal test pattern.
- **Task:**
    1.  Add `V4L2_CID_TEST_PATTERN` menu.
    2.  In `s_ctrl`, write to the sensor's Test Pattern register (usually 0x0600 or similar).
    3.  Verify visually (if you have a full pipeline).

### Lab 3: Frame Rate Control
- **Goal:** Implement `frame_interval`.
- **Task:**
    1.  Implement `.g_frame_interval` and `.s_frame_interval`.
    2.  Calculate VTS (Vertical Total Size) register based on requested FPS.
    3.  `FPS = PixelClock / (HTS * VTS)`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Remote I/O Error" (I2C NACK)
*   **Cause:** Wrong I2C address.
*   **Cause:** Sensor held in reset (Check GPIOs).
*   **Cause:** Missing MCLK (Master Clock). Sensors need a clock to respond to I2C!

#### 2. Black Image
*   **Cause:** Exposure too low.
*   **Cause:** `s_stream(1)` never called.
*   **Cause:** MIPI CSI lanes configuration mismatch (2 lanes vs 4 lanes).

---

## ⚡ Optimization & Best Practices

### Register Tables
*   Sensors have massive init sequences.
*   Don't write `regmap_write` 100 times.
*   Use a table:
    ```c
    struct reg_val { u16 reg; u8 val; };
    static const struct reg_val init_seq[] = { ... };
    
    for (i=0; i < ARRAY_SIZE(init_seq); i++)
        regmap_write(map, init_seq[i].reg, init_seq[i].val);
    ```

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `MEDIA_BUS_FMT`?
    *   **A:** It describes the format on the physical wire (e.g., 10-bit Raw Bayer). This is different from `V4L2_PIX_FMT` which describes the format in memory (e.g., 16-bit container).
2.  **Q:** Why `s_stream`?
    *   **A:** The bridge driver calls this when the user starts capturing. It tells the sensor to start outputting data on the CSI bus.

### Challenge Task
> **Task:** "The Binning Mode".
> *   If `set_fmt` requests 960x540 (1/2 of 1080p), enable 2x2 Binning in the sensor registers.
> *   Update `get_fmt` to return the new resolution.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/media/camera-sensor.rst](https://www.kernel.org/doc/html/latest/driver-api/media/camera-sensor.html)

---
