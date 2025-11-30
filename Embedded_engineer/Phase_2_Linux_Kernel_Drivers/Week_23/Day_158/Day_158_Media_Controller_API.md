# Day 158: The Media Controller API
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
1.  **Explain** the Media Controller graph (Entities, Pads, Links, Interfaces).
2.  **Register** a `media_device` and create links between entities.
3.  **Implement** `link_validate` to ensure formats match across the pipeline.
4.  **Use** `media-ctl` to configure the pipeline at runtime (Enable/Disable links).
5.  **Debug** "Broken Pipe" errors caused by format mismatches.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `media-ctl`, `dot`, `graphviz`.
*   **Prior Knowledge:**
    *   Day 156 (Subdevices).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Graph Model
*   **Entity:** A hardware block (Sensor, ISP, Scaler, DMA).
*   **Pad:** Connection point (Source/Sink).
*   **Link:** Connection between two pads.
    *   **Immutable:** Hardwired on the board (Sensor -> CSI).
    *   **Dynamic:** Configurable (CSI -> Memory OR CSI -> ISP).
*   **Interface:** The userspace handle (`/dev/video0`) linked to an Entity (DMA).

### 🔹 Part 2: Pipeline Validation
Before streaming starts, the Media Controller checks the graph.
*   Are all enabled links connected?
*   Do the formats match? (e.g., Sensor outputs 1080p, but ISP expects 720p -> Error).
*   This logic is handled by `v4l2_subdev_link_validate`.

---

## 💻 Implementation: Registering the Media Device

> **Instruction:** Add Media Controller support to our Bridge Driver.

### 👨‍💻 Code Implementation

#### Step 1: Structure Update
```c
#include <media/media-device.h>
#include <media/media-entity.h>

struct my_bridge_dev {
    struct v4l2_device v4l2_dev;
    struct media_device mdev; // The Media Controller
    // ...
};
```

#### Step 2: Initialization (in Probe)
```c
// 1. Init Media Device
strscpy(dev->mdev.model, "My Camera System", sizeof(dev->mdev.model));
dev->mdev.dev = &pdev->dev;
media_device_init(&dev->mdev);

// 2. Link V4L2 to Media
dev->v4l2_dev.mdev = &dev->mdev;

// 3. Register V4L2 Device
v4l2_device_register(&pdev->dev, &dev->v4l2_dev);

// ... Register Subdevs (Sensor) ...
// ... Register Video Node ...

// 4. Register Media Device (Must be last!)
media_device_register(&dev->mdev);
```

#### Step 3: Creating Links
Usually done by the bridge driver that knows the board layout.
```c
struct media_entity *sensor = &sensor_sd->entity;
struct media_entity *bridge = &bridge_sd->entity;

// Link Sensor:Pad0 (Source) -> Bridge:Pad0 (Sink)
// Flags: IMMUTABLE (Can't be changed), ENABLED (Active)
media_create_pad_link(sensor, 0, bridge, 0, 
                      MEDIA_LNK_FL_IMMUTABLE | MEDIA_LNK_FL_ENABLED);
```

---

## 💻 Implementation: Link Validation

> **Instruction:** Ensure formats match.

### 👨‍💻 Code Implementation

The V4L2 core provides a default helper: `v4l2_subdev_link_validate`.
It calls `get_fmt` on both ends of the link and compares them.

If you need custom logic (e.g., 10-bit packed matches 10-bit unpacked):
```c
static int my_link_validate(struct media_link *link) {
    struct v4l2_subdev *source_sd = media_entity_to_v4l2_subdev(link->source->entity);
    struct v4l2_subdev *sink_sd = media_entity_to_v4l2_subdev(link->sink->entity);
    
    // Get formats
    // Compare
    // Return 0 if OK, -EPIPE if mismatch
    return v4l2_subdev_link_validate_default(source_sd, link, &source_fmt, &sink_fmt);
}
```

---

## 🔬 Lab Exercise: Lab 158.1 - Visualizing the Graph

### 1. Lab Objectives
- Load the full stack (Sensor + Bridge).
- Generate a graph image.

### 2. Step-by-Step Guide
1.  **Generate Dot File:**
    ```bash
    media-ctl --print-dot > topology.dot
    ```
2.  **Convert to Image:**
    ```bash
    dot -Tpng topology.dot -o topology.png
    ```
3.  **View:** Open `topology.png`. You should see boxes (Entities) connected by arrows (Links).

---

## 🧪 Additional / Advanced Labs

### Lab 2: Dynamic Links
- **Goal:** Switch between two sensors.
- **Task:**
    1.  Create 2 Sensor Subdevs.
    2.  Link both to the Bridge (Multiplexer).
    3.  Links should NOT be Immutable.
    4.  Use `media-ctl -l '"Sensor A":0->"Bridge":0[1]'` to enable A.
    5.  Use `media-ctl -l '"Sensor A":0->"Bridge":0[0]'` to disable A.

### Lab 3: Interface Links
- **Goal:** Link the DMA entity to the `/dev/video0` node.
- **Task:**
    1.  `media_create_intf_link`.
    2.  This shows userspace which device node controls which part of the graph.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Link creation failed"
*   **Cause:** Pad index out of bounds.
*   **Cause:** Entities not registered yet.

#### 2. "Stream start failed: -EPIPE"
*   **Cause:** Link validation failed.
*   **Fix:** Check `dmesg`. It usually says "fmt mismatch: 1920x1080 vs 640x480".
*   **Fix:** Use `media-ctl -V` to propagate formats through the pipeline.

---

## ⚡ Optimization & Best Practices

### `media_pipeline_start`
*   When streaming starts, the driver calls `media_pipeline_start(&entity, &pipe)`.
*   This locks the graph (prevents link changes during streaming) and runs validation.
*   Don't forget `media_pipeline_stop`!

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why are some links "Immutable"?
    *   **A:** Because the hardware connection is physical (traces on PCB). You can't unplug the sensor from the CSI port via software.
2.  **Q:** What is `media-ctl -V`?
    *   **A:** It sets the format on a specific pad. `media-ctl -V '"Sensor":0 [fmt:SRGGB10/1920x1080]'`. This is how you configure the pipeline before streaming.

### Challenge Task
> **Task:** "The Resizer".
> *   Add a "Scaler" entity between Sensor and DMA.
> *   Input Pad: 1920x1080.
> *   Output Pad: 640x480.
> *   Validate that the link fails if you try to connect 1920x1080 directly to a DMA expecting 640x480 without the scaler.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/media/mc-core.rst](https://www.kernel.org/doc/html/latest/driver-api/media/mc-core.html)

---
