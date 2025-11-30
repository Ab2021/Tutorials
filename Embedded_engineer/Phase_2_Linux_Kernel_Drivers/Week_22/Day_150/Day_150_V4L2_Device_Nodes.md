# Day 150: V4L2 Device Nodes & Capabilities
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
1.  **Implement** the `v4l2_file_operations` (open, release, ioctl).
2.  **Define** Device Capabilities using `VIDIOC_QUERYCAP`.
3.  **Handle** Input/Output enumeration (`VIDIOC_ENUMINPUT`).
4.  **Connect** the `video_device` to the IOCTL handler using `video_ioctl2`.
5.  **Debug** IOCTL calls using `v4l2-ctl --log-status`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 149 (V4L2 Registration).
    *   Day 129 (IOCTLs).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The IOCTL Dispatcher
In a standard Char Driver, you write a giant `switch(cmd)` statement in `ioctl`.
In V4L2, the kernel provides a helper: `video_ioctl2`.
*   **Mechanism:** You populate a `struct v4l2_ioctl_ops` with function pointers (`.vidioc_querycap`, `.vidioc_g_fmt`).
*   **Dispatcher:** `video_ioctl2` looks up the command, locks the mutex, calls your function, and handles copy_to/from_user.

### 🔹 Part 2: Capabilities
The first thing userspace does is call `VIDIOC_QUERYCAP`.
*   **Driver Name:** "my_driver".
*   **Card Name:** "My Camera".
*   **Bus Info:** "platform:my_driver".
*   **Capabilities:**
    *   `V4L2_CAP_VIDEO_CAPTURE`: It's a camera.
    *   `V4L2_CAP_STREAMING`: It supports DMA streaming (mmap).
    *   `V4L2_CAP_READWRITE`: It supports `read()` (slow, legacy).

---

## 💻 Implementation: File Operations

> **Instruction:** Extend the driver from Day 149.

### 👨‍💻 Code Implementation

#### Step 1: File Operations
```c
static int my_open(struct file *file) {
    struct my_video_dev *dev = video_drvdata(file);
    // Power up hardware?
    return v4l2_fh_open(file); // Helper for file handle management
}

static int my_release(struct file *file) {
    // Power down hardware?
    return v4l2_fh_release(file);
}

static const struct v4l2_file_operations my_fops = {
    .owner = THIS_MODULE,
    .open = my_open,
    .release = my_release,
    .unlocked_ioctl = video_ioctl2, // THE MAGIC DISPATCHER
    .mmap = vb2_fop_mmap,           // Later
    .poll = vb2_fop_poll,           // Later
};
```

---

## 💻 Implementation: IOCTL Operations

> **Instruction:** Implement `querycap` and `enum_input`.

### 👨‍💻 Code Implementation

#### Step 1: Query Capabilities
```c
static int my_querycap(struct file *file, void *priv, struct v4l2_capability *cap) {
    strscpy(cap->driver, "my_v4l2_driver", sizeof(cap->driver));
    strscpy(cap->card, "Virtual Camera", sizeof(cap->card));
    strscpy(cap->bus_info, "platform:virtual", sizeof(cap->bus_info));
    
    cap->device_caps = V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING;
    cap->capabilities = cap->device_caps | V4L2_CAP_DEVICE_CAPS;
    
    return 0;
}
```

#### Step 2: Enum Input (Selecting Source)
Even if you only have one sensor, you must support this.
```c
static int my_enum_input(struct file *file, void *priv, struct v4l2_input *inp) {
    if (inp->index > 0) return -EINVAL; // Only 1 input
    
    inp->type = V4L2_INPUT_TYPE_CAMERA;
    strscpy(inp->name, "Camera Sensor", sizeof(inp->name));
    return 0;
}

static int my_g_input(struct file *file, void *priv, unsigned int *i) {
    *i = 0;
    return 0;
}

static int my_s_input(struct file *file, void *priv, unsigned int i) {
    if (i > 0) return -EINVAL;
    return 0;
}
```

#### Step 3: Registering Ops
```c
static const struct v4l2_ioctl_ops my_ioctl_ops = {
    .vidioc_querycap = my_querycap,
    .vidioc_enum_input = my_enum_input,
    .vidioc_g_input = my_g_input,
    .vidioc_s_input = my_s_input,
};

// In Probe:
dev->vdev.fops = &my_fops;
dev->vdev.ioctl_ops = &my_ioctl_ops;
```

---

## 🔬 Lab Exercise: Lab 150.1 - Verification

### 1. Lab Objectives
- Compile and load.
- Use `v4l2-ctl` to query the new capabilities.

### 2. Step-by-Step Guide
1.  Load module.
2.  Run:
    ```bash
    v4l2-ctl -d /dev/video0 --all
    ```
3.  **Expected Output:**
    *   Driver name: my_v4l2_driver
    *   Card type: Virtual Camera
    *   Inputs: Camera Sensor (Index 0)

---

## 🧪 Additional / Advanced Labs

### Lab 2: Priority Handling
- **Goal:** Prevent one app from changing settings while another records.
- **Task:**
    1.  Use `v4l2_prio_init`, `v4l2_prio_open`, `v4l2_prio_check`.
    2.  Add `.vidioc_g_priority` and `.vidioc_s_priority`.
    3.  Try to change format from a second terminal while first is "recording" (simulated).

### Lab 3: Debugging IOCTLs
- **Goal:** Trace the calls.
- **Task:**
    1.  `echo 1 > /sys/class/video4linux/video0/dev_debug`.
    2.  Run `v4l2-ctl -d /dev/video0 --all`.
    3.  Check `dmesg`. You will see every IOCTL call logged by the kernel core.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Inappropriate ioctl for device"
*   **Cause:** `unlocked_ioctl` not set in `fops`.
*   **Cause:** `video_register_device` called with `VFL_TYPE_SUBDEV` instead of `VFL_TYPE_VIDEO`.

#### 2. Missing Capabilities
*   **Cause:** Forgot to set `cap->device_caps`.
*   **Cause:** Forgot `V4L2_CAP_DEVICE_CAPS` flag.

---

## ⚡ Optimization & Best Practices

### `video_device_release`
*   **Critical:** Never use `devm_kzalloc` for the `video_device` structure if you use `video_device_release_empty`.
*   **Why?** If the device is unplugged while an app has it open, `remove()` is called, but the structure must persist until the app closes the file.
*   **Fix:** Embed `video_device` in your struct, and use a custom release function that frees the whole container.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `unlocked_ioctl`?
    *   **A:** It means the BKL (Big Kernel Lock) is not held. In modern Linux, all IOCTLs are unlocked. The driver must handle its own locking (which `video_ioctl2` does via `vdev.lock`).
2.  **Q:** Why `v4l2_fh_open`?
    *   **A:** It initializes a "File Handle" (`v4l2_fh`) which tracks per-open-file state (like priority, events subscribed).

### Challenge Task
> **Task:** "The Multi-Input Switch".
> *   Modify `enum_input` to report 2 inputs: "Front Camera" (0) and "Rear Camera" (1).
> *   Implement `s_input` to store the selected index in `dev->input_index`.
> *   Print a log message when input switches.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/media/v4l2-dev.rst](https://www.kernel.org/doc/html/latest/driver-api/media/v4l2-dev.html)

---
