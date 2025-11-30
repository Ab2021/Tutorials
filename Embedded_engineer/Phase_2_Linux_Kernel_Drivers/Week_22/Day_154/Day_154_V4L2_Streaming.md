# Day 154: V4L2 Streaming & Capture Loop
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
1.  **Implement** the Streaming Loop (Start -> Capture -> Done -> Next).
2.  **Generate** synthetic video data (Color Bars) in software.
3.  **Use** Kernel Timers (`hrtimer`) to simulate a specific frame rate (e.g., 30 FPS).
4.  **Fill** `vb2` buffers with pixel data (`vb2_plane_vaddr`).
5.  **Capture** video using `ffmpeg` or `v4l2-ctl`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `ffmpeg` (optional).
*   **Prior Knowledge:**
    *   Day 153 (Videobuf2).
    *   Kernel Timers.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Capture Loop
1.  **Start Streaming:** Enable the "Sensor".
2.  **Interrupt/Timer:** Indicates a new frame is ready.
3.  **Acquire Buffer:** Get the next available buffer from `buf_list`.
4.  **Fill Buffer:** Copy data (DMA or memcpy).
5.  **Return Buffer:** Call `vb2_buffer_done`.
6.  **Repeat.**

### 🔹 Part 2: Virtual Frame Generation
Since we don't have a real sensor, we will use a Kernel Timer (`hrtimer`) to wake up every 33ms (30 FPS) and fill the buffer with a test pattern.

---

## 💻 Implementation: The Frame Generator

> **Instruction:** Add a timer and a fill function.

### 👨‍💻 Code Implementation

#### Step 1: Structure Update
```c
#include <linux/hrtimer.h>
#include <linux/ktime.h>

struct my_video_dev {
    // ...
    struct hrtimer timer;
    int sequence;
};
```

#### Step 2: Pattern Generator (YUYV)
```c
static void fill_buffer(struct my_video_dev *dev, struct vb2_v4l2_buffer *vbuf) {
    struct vb2_buffer *vb = &vbuf->vb2_buf;
    u8 *ptr = vb2_plane_vaddr(vb, 0);
    int width = dev->fmt.fmt.pix.width;
    int height = dev->fmt.fmt.pix.height;
    int i, j;
    
    // Simple Color Bar (Vertical)
    for (j = 0; j < height; j++) {
        for (i = 0; i < width; i += 2) {
            // YUYV: Y0 U0 Y1 V0
            // Generate a moving bar based on sequence
            int bar = (i / (width / 8) + dev->sequence) % 8;
            
            // White, Yellow, Cyan, Green, Magenta, Red, Blue, Black
            // (Simplified YUV values)
            ptr[0] = (bar & 4) ? 235 : 16; // Y0
            ptr[1] = (bar & 2) ? 16 : 128; // U
            ptr[2] = (bar & 4) ? 235 : 16; // Y1
            ptr[3] = (bar & 1) ? 16 : 128; // V
            
            ptr += 4;
        }
    }
    
    vb2_set_plane_payload(vb, 0, width * height * 2);
}
```

#### Step 3: Timer Callback
```c
static enum hrtimer_restart my_timer_callback(struct hrtimer *timer) {
    struct my_video_dev *dev = container_of(timer, struct my_video_dev, timer);
    struct vb2_v4l2_buffer *vbuf;
    unsigned long flags;
    
    spin_lock_irqsave(&dev->slock, flags);
    
    if (list_empty(&dev->buf_list)) {
        // No buffers available! Drop frame.
        spin_unlock_irqrestore(&dev->slock, flags);
        goto restart;
    }
    
    // Get next buffer
    vbuf = list_first_entry(&dev->buf_list, struct vb2_v4l2_buffer, list);
    list_del(&vbuf->list);
    
    spin_unlock_irqrestore(&dev->slock, flags);
    
    // Fill it
    fill_buffer(dev, vbuf);
    
    // Timestamp & Sequence
    vbuf->vb2_buf.timestamp = ktime_get_ns();
    vbuf->sequence = dev->sequence++;
    
    // Mark done
    vb2_buffer_done(&vbuf->vb2_buf, VB2_BUF_STATE_DONE);
    
restart:
    hrtimer_forward_now(timer, ktime_set(0, 33333333)); // 30 FPS
    return HRTIMER_RESTART;
}
```

#### Step 4: Start/Stop Streaming Update
```c
static int my_start_streaming(struct vb2_queue *q, unsigned int count) {
    struct my_video_dev *dev = vb2_get_drv_priv(q);
    
    dev->sequence = 0;
    hrtimer_init(&dev->timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
    dev->timer.function = my_timer_callback;
    hrtimer_start(&dev->timer, ktime_set(0, 33333333), HRTIMER_MODE_REL);
    
    return 0;
}

static void my_stop_streaming(struct vb2_queue *q) {
    struct my_video_dev *dev = vb2_get_drv_priv(q);
    
    hrtimer_cancel(&dev->timer);
    
    // ... return buffers (Day 153 code) ...
}
```

---

## 🔬 Lab Exercise: Lab 154.1 - Capturing Video

### 1. Lab Objectives
- Compile and load.
- Capture 100 frames to a file.
- Play it back.

### 2. Step-by-Step Guide
1.  **Capture:**
    ```bash
    v4l2-ctl -d /dev/video0 --set-fmt-video=width=640,height=480,pixelformat=YUYV --stream-mmap --stream-count=100 --stream-to=video.raw
    ```
2.  **Play (using ffplay):**
    ```bash
    ffplay -f rawvideo -pixel_format yuyv422 -video_size 640x480 video.raw
    ```
3.  **Observation:** You should see moving color bars.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Changing Frame Rate
- **Goal:** Support `VIDIOC_S_PARM`.
- **Task:**
    1.  Implement `.vidioc_s_parm`.
    2.  Read `parm->parm.capture.timeperframe`.
    3.  Update timer interval.

### Lab 3: Blocking vs Non-Blocking
- **Goal:** Understand `poll`.
- **Task:**
    1.  Write a C program using `select()`.
    2.  Verify `select` blocks until a frame is ready.
    3.  This works because `vb2_buffer_done` calls `wake_up`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Buffer underrun" / Dropped Frames
*   **Cause:** Userspace is too slow to DQBUF and QBUF back.
*   **Fix:** Increase number of buffers (REQBUFS=8).

#### 2. Colors look wrong
*   **Cause:** YUYV byte order.
*   **Fix:** Check if your generator writes Y0 U0 Y1 V0 or U0 Y0 V0 Y1.

---

## ⚡ Optimization & Best Practices

### `vb2_plane_vaddr`
*   This returns the Kernel Virtual Address of the buffer.
*   Only works for `vb2_vmalloc` or `vb2_dma_contig` (if mapped).
*   **Performance:** Writing pixel-by-pixel in C is slow. Real drivers use DMA.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What happens if I call `vb2_buffer_done` twice on the same buffer?
    *   **A:** Kernel Panic or corruption. Ownership passes to userspace immediately.
2.  **Q:** Why `hrtimer` instead of `timer_list`?
    *   **A:** `hrtimer` (High Resolution Timer) is more precise for video timing (33.33ms). Standard timers depend on `HZ` (jiffies).

### Challenge Task
> **Task:** "The Bouncing Ball".
> *   Modify `fill_buffer` to draw a white square moving across a black background.
> *   Update X/Y position in each frame.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/media/v4l2-subdev.rst](https://www.kernel.org/doc/html/latest/driver-api/media/v4l2-subdev.html)

---
