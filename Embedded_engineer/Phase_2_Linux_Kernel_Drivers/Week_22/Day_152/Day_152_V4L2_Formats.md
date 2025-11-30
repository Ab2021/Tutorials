# Day 152: V4L2 Formats & Negotiation
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
1.  **Understand** Pixel Formats (RGB565, YUYV, NV12, MJPEG) and FOURCC codes.
2.  **Implement** Format IOCTLs: `VIDIOC_ENUM_FMT`, `VIDIOC_G_FMT`, `VIDIOC_S_FMT`, `VIDIOC_TRY_FMT`.
3.  **Negotiate** resolution and frame rates with userspace.
4.  **Calculate** buffer sizes (`sizeimage`) and line strides (`bytesperline`).
5.  **Debug** format mismatches using `v4l2-ctl --get-fmt-video`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 150 (IOCTLs).
    *   Color Spaces (RGB vs YUV).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Pixel Formats & FOURCC
Video data is heavy. We need standard ways to describe the memory layout.
*   **FOURCC:** Four Character Code. A 32-bit integer formed by 4 ASCII chars.
    *   `V4L2_PIX_FMT_RGB565` ('R','G','B','P')
    *   `V4L2_PIX_FMT_YUYV` ('Y','U','Y','V')
    *   `V4L2_PIX_FMT_MJPEG` ('M','J','P','G')

### 🔹 Part 2: The Negotiation Dance
1.  **Enum:** App asks "What do you support?" (Driver lists YUYV, MJPEG).
2.  **Try:** App asks "Can you do 1920x1080 YUYV?" (Driver says "No, max is 640x480").
3.  **Set:** App says "Okay, set 640x480 YUYV". (Driver configures hardware).
4.  **Get:** App confirms "What is currently set?".

---

## 💻 Implementation: Format Structures

> **Instruction:** Extend the driver to store the current format.

### 👨‍💻 Code Implementation

#### Step 1: Structure Update
```c
struct my_video_dev {
    // ...
    struct v4l2_format fmt;
};
```

#### Step 2: Default Format (in Probe)
```c
dev->fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
dev->fmt.fmt.pix.width = 640;
dev->fmt.fmt.pix.height = 480;
dev->fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
dev->fmt.fmt.pix.field = V4L2_FIELD_NONE;
dev->fmt.fmt.pix.bytesperline = 640 * 2; // YUYV is 2 bytes/pixel
dev->fmt.fmt.pix.sizeimage = 640 * 480 * 2;
dev->fmt.fmt.pix.colorspace = V4L2_COLORSPACE_SRGB;
```

---

## 💻 Implementation: Format IOCTLs

> **Instruction:** Implement the 4 key IOCTLs.

### 👨‍💻 Code Implementation

#### Step 1: Enum Format
```c
static int my_enum_fmt_vid_cap(struct file *file, void *priv, struct v4l2_fmtdesc *f) {
    if (f->index > 0) return -EINVAL; // Only 1 format supported for now

    f->pixelformat = V4L2_PIX_FMT_YUYV;
    return 0;
}
```

#### Step 2: Get Format
```c
static int my_g_fmt_vid_cap(struct file *file, void *priv, struct v4l2_format *f) {
    struct my_video_dev *dev = video_drvdata(file);
    
    *f = dev->fmt; // Return current state
    return 0;
}
```

#### Step 3: Try Format (Validation)
This function checks if the requested format is valid, but DOES NOT change the hardware state.
```c
static int my_try_fmt_vid_cap(struct file *file, void *priv, struct v4l2_format *f) {
    // 1. Force Pixel Format
    if (f->fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV)
        f->fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    
    // 2. Clamp Resolution (Min 16x16, Max 1920x1080)
    v4l_bound_align_image(&f->fmt.pix.width, 16, 1920, 0,
                          &f->fmt.pix.height, 16, 1080, 0, 0);
                          
    // 3. Recalculate Bytes/Line and Size
    f->fmt.pix.bytesperline = f->fmt.pix.width * 2;
    f->fmt.pix.sizeimage = f->fmt.pix.bytesperline * f->fmt.pix.height;
    f->fmt.pix.field = V4L2_FIELD_NONE;
    
    return 0;
}
```

#### Step 4: Set Format
```c
static int my_s_fmt_vid_cap(struct file *file, void *priv, struct v4l2_format *f) {
    struct my_video_dev *dev = video_drvdata(file);
    
    // 1. Check if busy (Streaming?)
    if (vb2_is_busy(&dev->vb_vidq)) return -EBUSY;
    
    // 2. Run Try logic first
    my_try_fmt_vid_cap(file, priv, f);
    
    // 3. Store it
    dev->fmt = *f;
    
    return 0;
}
```

#### Step 5: Register Ops
```c
static const struct v4l2_ioctl_ops my_ioctl_ops = {
    // ...
    .vidioc_enum_fmt_vid_cap = my_enum_fmt_vid_cap,
    .vidioc_g_fmt_vid_cap = my_g_fmt_vid_cap,
    .vidioc_try_fmt_vid_cap = my_try_fmt_vid_cap,
    .vidioc_s_fmt_vid_cap = my_s_fmt_vid_cap,
};
```

---

## 🔬 Lab Exercise: Lab 152.1 - Format Testing

### 1. Lab Objectives
- Compile and load.
- Check default format.
- Try to set an invalid format.

### 2. Step-by-Step Guide
1.  **Check Default:**
    ```bash
    v4l2-ctl -d /dev/video0 --get-fmt-video
    # Output: Width/Height: 640/480, Pixel Format: 'YUYV'
    ```
2.  **Set Resolution:**
    ```bash
    v4l2-ctl -d /dev/video0 --set-fmt-video=width=320,height=240
    ```
3.  **Try Invalid (RGB):**
    ```bash
    v4l2-ctl -d /dev/video0 --set-fmt-video=pixelformat=RGB3
    # It should fail or silently revert to YUYV (depending on implementation).
    # Our try_fmt forces YUYV, so it will revert.
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Multiple Formats
- **Goal:** Support RGB565 and YUYV.
- **Task:**
    1.  Update `enum_fmt` to return RGB565 for index 1.
    2.  Update `try_fmt` to accept RGB565.
    3.  Update `bytesperline` calculation (RGB565 is also 2 bytes, but RGB24 is 3).

### Lab 3: Frame Sizes (`VIDIOC_ENUM_FRAMESIZES`)
- **Goal:** Tell userspace which resolutions are supported.
- **Task:**
    1.  Implement `.vidioc_enum_framesizes`.
    2.  Return discrete list: 640x480, 1280x720.
    3.  Verify with `v4l2-ctl --list-formats-ext`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Invalid Argument" on Set Format
*   **Cause:** `try_fmt` failed or returned a format the hardware can't handle.
*   **Cause:** `v4l_bound_align_image` not used, leading to weird alignment (e.g., width=641).

#### 2. Buffer Size Mismatch
*   **Cause:** `sizeimage` calculated incorrectly.
*   **Fix:** Always ensure `sizeimage >= bytesperline * height`.

---

## ⚡ Optimization & Best Practices

### `v4l2_fill_pix_format`
*   Helper function to calculate stride and size.
*   `v4l2_fill_pix_format(&f->fmt.pix, &f->fmt.pix);`

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `TRY_FMT` and `S_FMT`?
    *   **A:** `TRY_FMT` is a "Dry Run". It asks "If I requested X, what would you give me?". `S_FMT` actually commits the configuration.
2.  **Q:** Why do we need `bytesperline`?
    *   **A:** Padding. A 640-pixel wide image might be stored in a buffer with a stride of 1024 bytes for hardware alignment reasons.

### Challenge Task
> **Task:** "The Planar Format".
> *   Add support for `V4L2_PIX_FMT_YUV420` (NV12 or I420).
> *   Calculate `sizeimage` correctly (Width * Height * 1.5).

---

## 📚 Further Reading & References
- [Kernel Documentation: userspace-api/media/v4l/pixfmt.rst](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/pixfmt.html)

---
