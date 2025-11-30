# Day 159: V4L2 Async Framework
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
1.  **Explain** the "Probe Order Problem" in embedded systems.
2.  **Implement** `v4l2_async_register_subdev` in sensor drivers.
3.  **Implement** `v4l2_async_nf_init` and `v4l2_async_nf_add_fwnode_remote` in bridge drivers.
4.  **Handle** the `bound`, `complete`, and `unbind` callbacks.
5.  **Parse** Device Tree endpoints (`fwnode`) to find connected subdevices.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 156 (Subdevices).
    *   Day 136 (Device Tree).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Probe Order Problem
In a camera system, the "Bridge" (CSI Receiver) and the "Sensor" (I2C Device) are separate drivers.
*   **Scenario A:** Bridge probes first. It looks for the sensor. Sensor isn't loaded yet. Bridge fails.
*   **Scenario B:** Sensor probes first. It registers. Bridge loads later and finds it.
*   **Solution:** **Deferred Probing** or **V4L2 Async**.
    *   V4L2 Async allows the Bridge to say "I am waiting for a sensor at this I2C address/DT node. Call me when it arrives."

### 🔹 Part 2: The Async Notifier
*   **Notifier:** The Bridge creates a list of subdevices it expects.
*   **Async Subdev:** The Sensor registers itself as "Available".
*   **Matching:** The V4L2 core matches them (usually via Device Tree `endpoint` nodes).

---

## 💻 Implementation: Sensor Driver (Async)

> **Instruction:** Modify the sensor driver to register asynchronously.

### 👨‍💻 Code Implementation

```c
// In Probe
// Instead of v4l2_device_register_subdev (which requires the v4l2_dev pointer)
// We use:
ret = v4l2_async_register_subdev(&sensor->sd);
if (ret) return ret;
```

That's it! The sensor just announces "I'm here".

---

## 💻 Implementation: Bridge Driver (Async Notifier)

> **Instruction:** The Bridge must parse DT and wait for the sensor.

### 👨‍💻 Code Implementation

#### Step 1: Structure Update
```c
struct my_bridge {
    struct v4l2_device v4l2_dev;
    struct v4l2_async_notifier notifier;
    struct v4l2_subdev *sensor_sd;
};
```

#### Step 2: Callbacks
```c
static int my_notify_bound(struct v4l2_async_notifier *notifier,
                           struct v4l2_subdev *sd,
                           struct v4l2_async_subdev *asd) {
    struct my_bridge *bridge = container_of(notifier, struct my_bridge, notifier);
    
    dev_info(bridge->v4l2_dev.dev, "Bound subdev: %s\n", sd->name);
    bridge->sensor_sd = sd;
    
    // Create Link (Sensor -> Bridge)
    // media_create_pad_link(...)
    
    return 0;
}

static int my_notify_complete(struct v4l2_async_notifier *notifier) {
    struct my_bridge *bridge = container_of(notifier, struct my_bridge, notifier);
    
    dev_info(bridge->v4l2_dev.dev, "All subdevs bound. Registering Video Node.\n");
    
    // Register /dev/video0 here!
    // video_register_device(...)
    
    return 0;
}

static const struct v4l2_async_notifier_operations my_notify_ops = {
    .bound = my_notify_bound,
    .complete = my_notify_complete,
};
```

#### Step 3: Parsing DT (in Probe)
```c
struct fwnode_handle *ep;
struct v4l2_async_subdev *asd;

// 1. Init Notifier
v4l2_async_nf_init(&bridge->notifier);
bridge->notifier.ops = &my_notify_ops;

// 2. Find Remote Endpoint (The Sensor)
// Assuming DT: port { endpoint { remote-endpoint = <&sensor_ep>; }; };
ep = fwnode_graph_get_endpoint_by_id(dev_fwnode(dev), 0, 0, 0);
if (!ep) return -ENODEV;

struct fwnode_handle *remote = fwnode_graph_get_remote_endpoint(ep);
fwnode_handle_put(ep);

// 3. Add to Notifier List
asd = v4l2_async_nf_add_fwnode_remote(&bridge->notifier, remote,
                                      struct v4l2_async_subdev);
fwnode_handle_put(remote);
if (IS_ERR(asd)) return PTR_ERR(asd);

// 4. Register Notifier
ret = v4l2_async_nf_register(&bridge->v4l2_dev, &bridge->notifier);
if (ret) {
    v4l2_async_nf_cleanup(&bridge->notifier);
    return ret;
}
```

---

## 🔬 Lab Exercise: Lab 159.1 - Testing Async Probe

### 1. Lab Objectives
- Load Bridge first (should wait).
- Load Sensor second (should trigger bind).

### 2. Step-by-Step Guide
1.  **Load Bridge:**
    ```bash
    insmod my_bridge.ko
    ```
    *   Check `dmesg`. It should NOT say "Video Device Registered". It's waiting.
2.  **Load Sensor:**
    ```bash
    insmod my_sensor.ko
    ```
3.  **Check `dmesg`:**
    *   "Bound subdev: my_sensor"
    *   "All subdevs bound. Registering Video Node."
    *   "/dev/video0 registered"

---

## 🧪 Additional / Advanced Labs

### Lab 2: Multiple Sensors
- **Goal:** Wait for 2 sensors (Front/Rear).
- **Task:**
    1.  Parse 2 endpoints.
    2.  Add both to notifier.
    3.  `complete` is only called when BOTH are bound.

### Lab 3: Unbind
- **Goal:** Handle sensor removal.
- **Task:**
    1.  `rmmod my_sensor`.
    2.  Implement `.unbind` callback.
    3.  Unregister video node to prevent crashes.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Notifier registration failed"
*   **Cause:** `v4l2_device` not registered first.
*   **Cause:** `fwnode` lookup failed (DT mismatch).

#### 2. Never Binds
*   **Cause:** Sensor driver didn't call `v4l2_async_register_subdev`.
*   **Cause:** DT `remote-endpoint` phandles don't match.

---

## ⚡ Optimization & Best Practices

### `v4l2_async_register_subdev_sensor`
*   Special helper for sensors.
*   Automatically parses the sensor's DT node to find its own properties (orientation, rotation).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not just use `EPROBE_DEFER`?
    *   **A:** `EPROBE_DEFER` works for the *entire* driver. V4L2 Async allows the Bridge to load partially, register the Media Device, and then add subdevices as they appear. It's more granular.
2.  **Q:** What is `fwnode`?
    *   **A:** Firmware Node. An abstraction over Device Tree (OF) and ACPI.

### Challenge Task
> **Task:** "The Hotplug Sensor".
> *   Simulate a sensor that is powered on/off by a GPIO.
> *   When GPIO is toggled, load/unload the sensor module.
> *   Verify the Bridge handles the disappearance gracefully.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/media/v4l2-async.rst](https://www.kernel.org/doc/html/latest/driver-api/media/v4l2-async.html)

---
