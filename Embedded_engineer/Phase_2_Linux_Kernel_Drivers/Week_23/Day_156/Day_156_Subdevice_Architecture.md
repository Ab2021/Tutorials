# Day 156: V4L2 Subdevice Architecture
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
1.  **Explain** the role of `v4l2_subdev` in complex video pipelines.
2.  **Differentiate** between a Video Device (DMA) and a Subdevice (Processing/Source).
3.  **Implement** the `v4l2_subdev_ops` (core, video, pad).
4.  **Register** a subdevice with the V4L2 core (`v4l2_device_register_subdev`).
5.  **Visualize** Pads and Links using `media-ctl --print-dot`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `media-ctl` (`v4l-utils`).
*   **Prior Knowledge:**
    *   Day 149 (V4L2 Architecture).
    *   Day 142 (I2C).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Monolith vs The Pipeline
*   **Monolith (Week 22):** One driver handles everything (Sensor control + DMA). Good for USB webcams.
*   **Pipeline (Week 23):**
    *   **Sensor (I2C):** Generates pixels.
    *   **CSI Receiver (SoC):** Receives pixels.
    *   **ISP (SoC):** Processes pixels.
    *   **DMA Engine (SoC):** Writes to RAM.
    *   **Problem:** We need separate drivers for each, but they must work together.
    *   **Solution:** **Subdevices**.

### 🔹 Part 2: Pads and Links
*   **Pad:** A port on a chip.
    *   Sensor has 1 Source Pad (Output).
    *   CSI Receiver has 1 Sink Pad (Input) and 1 Source Pad (Output).
*   **Link:** A connection between two pads.
    *   Sensor:Pad0 -> CSI:Pad0.

---

## 💻 Implementation: The Subdevice Skeleton

> **Instruction:** We will create a dummy sensor driver that registers as a subdevice.

### 👨‍💻 Code Implementation

#### Step 1: Structure
```c
#include <linux/module.h>
#include <linux/i2c.h>
#include <media/v4l2-subdev.h>

struct my_sensor {
    struct i2c_client *client;
    struct v4l2_subdev sd;
    struct media_pad pad;
};
```

#### Step 2: Subdev Operations
Subdevices don't use `file_operations`. They use `subdev_ops`.
```c
static int my_log_status(struct v4l2_subdev *sd) {
    struct my_sensor *sensor = container_of(sd, struct my_sensor, sd);
    dev_info(&sensor->client->dev, "My Sensor Status: OK\n");
    return 0;
}

static const struct v4l2_subdev_core_ops my_core_ops = {
    .log_status = my_log_status,
};

static const struct v4l2_subdev_ops my_subdev_ops = {
    .core = &my_core_ops,
};
```

#### Step 3: Probe (I2C Driver)
```c
static int my_probe(struct i2c_client *client, const struct i2c_device_id *id) {
    struct my_sensor *sensor;
    
    sensor = devm_kzalloc(&client->dev, sizeof(*sensor), GFP_KERNEL);
    if (!sensor) return -ENOMEM;
    
    sensor->client = client;
    
    // 1. Initialize Subdev
    v4l2_i2c_subdev_init(&sensor->sd, client, &my_subdev_ops);
    
    // 2. Initialize Media Pad
    sensor->pad.flags = MEDIA_PAD_FL_SOURCE; // Output
    sensor->sd.entity.function = MEDIA_ENT_F_CAM_SENSOR;
    
    // 3. Register Entity
    int ret = media_entity_pads_init(&sensor->sd.entity, 1, &sensor->pad);
    if (ret) return ret;
    
    dev_info(&client->dev, "Subdevice Registered\n");
    return 0;
}

static int my_remove(struct i2c_client *client) {
    struct v4l2_subdev *sd = i2c_get_clientdata(client);
    media_entity_cleanup(&sd->entity);
    return 0;
}
```

---

## 💻 Implementation: The Bridge Driver (Host)

> **Instruction:** A subdevice is useless without a "Master" (Bridge) to register it.

### 👨‍💻 Code Implementation

#### Step 1: Registering the Subdev
In your `virtcam` or platform driver (from Week 22):
```c
// In Probe
struct v4l2_subdev *sensor_sd;
struct i2c_adapter *adapter = i2c_get_adapter(0);

// Load the module and get the subdev
sensor_sd = v4l2_i2c_new_subdev_board(&dev->v4l2_dev, adapter, &board_info, NULL);

if (sensor_sd) {
    printk("Sensor found: %s\n", sensor_sd->name);
}
```

---

## 🔬 Lab Exercise: Lab 156.1 - Media Topology

### 1. Lab Objectives
- Load the subdevice driver.
- Use `media-ctl` to see the entity.

### 2. Step-by-Step Guide
1.  Load `my_sensor.ko`.
2.  Load the bridge driver (if using `vivid` or custom).
3.  Run:
    ```bash
    media-ctl -p
    ```
4.  **Expected Output:**
    ```text
    Entity 1: my_sensor 0-0010 (1 pad, 0 links)
        type V4L2 subdev subtype Sensor flags 0
        pad0: Source
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Internal Ops
- **Goal:** Call a subdev function from the bridge.
- **Task:**
    1.  In bridge driver: `v4l2_subdev_call(sensor_sd, core, log_status);`.
    2.  Check dmesg. You should see "My Sensor Status: OK".

### Lab 3: Async Registration
- **Goal:** Handle probe order issues.
- **Task:**
    1.  Use `v4l2_async_register_subdev` in the sensor driver.
    2.  Use `v4l2_async_notifier` in the bridge driver.
    3.  This allows the sensor to load *after* the bridge.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Symbol not found"
*   **Cause:** `v4l2_i2c_subdev_init` requires `CONFIG_VIDEO_V4L2_I2C`.

#### 2. Subdev not showing in media-ctl
*   **Cause:** Forgot `media_entity_pads_init`.
*   **Cause:** Forgot to register the `v4l2_device` with a `media_device`.

---

## ⚡ Optimization & Best Practices

### `v4l2_i2c_subdev_init`
*   Helper that sets up `sd->name`, `sd->owner`, `sd->dev`, and `i2c_set_clientdata`.
*   Always use this for I2C subdevices.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can a subdevice have IOCTLs?
    *   **A:** Yes! If you create a device node (`V4L2_SUBDEV_FL_HAS_DEVNODE`), userspace can talk directly to the subdevice via `/dev/v4l-subdevX`.
2.  **Q:** What is a "Pad"?
    *   **A:** A logical input or output port on a media entity. It defines the flow of data.

### Challenge Task
> **Task:** "The Passthrough".
> *   Create a dummy "Filter" subdevice.
> *   1 Sink Pad, 1 Source Pad.
> *   Register it.
> *   Verify it appears in `media-ctl`.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/media/v4l2-subdev.rst](https://www.kernel.org/doc/html/latest/driver-api/media/v4l2-subdev.html)

---
