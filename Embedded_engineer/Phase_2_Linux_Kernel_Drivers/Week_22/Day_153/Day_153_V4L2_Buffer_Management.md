# Day 153: V4L2 Buffer Management (Videobuf2)
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
1.  **Explain** the role of `videobuf2` (vb2) in managing DMA buffers.
2.  **Initialize** a `vb2_queue` with appropriate ops and mem_ops (`vb2_vmalloc` vs `vb2_dma_contig`).
3.  **Implement** the core vb2 callbacks: `queue_setup`, `buf_prepare`, `buf_queue`.
4.  **Handle** MMAP IOCTLs (`REQBUFS`, `QUERYBUF`, `QBUF`, `DQBUF`) via vb2 helpers.
5.  **Debug** buffer state transitions (Queued -> Active -> Done).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 152 (Formats).
    *   DMA Concepts.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Buffer Lifecycle
Video streaming requires a ring of buffers.
1.  **REQBUFS:** Userspace asks for N buffers. Driver allocates them.
2.  **QUERYBUF:** Userspace asks for the address/offset of each buffer.
3.  **MMAP:** Userspace maps them into its address space.
4.  **QBUF:** Userspace gives a buffer to the driver ("Fill this!").
5.  **Capture:** Hardware fills the buffer via DMA.
6.  **DQBUF:** Userspace takes the filled buffer back ("Thanks!").

### 🔹 Part 2: Videobuf2 (vb2)
Writing this logic manually is a nightmare (Race conditions, locking, memory management).
*   **vb2:** The standard kernel framework that handles ALL of the above.
*   **Driver Role:** Just implement a few callbacks to configure the hardware.

### 🔹 Part 3: Memory Models
*   `vb2_vmalloc`: Buffers allocated via `vmalloc`. Good for software generators (Virtual Cam).
*   `vb2_dma_contig`: Physically contiguous DMA. Good for simple hardware.
*   `vb2_dma_sg`: Scatter-Gather DMA. Good for advanced hardware.

---

## 💻 Implementation: Initializing vb2

> **Instruction:** Add the queue to our driver structure.

### 👨‍💻 Code Implementation

#### Step 1: Structure Update
```c
#include <media/videobuf2-v4l2.h>
#include <media/videobuf2-vmalloc.h> // For Virtual Cam

struct my_video_dev {
    // ...
    struct vb2_queue vb_vidq;
    struct list_head buf_list; // List of queued buffers
    spinlock_t slock;          // Spinlock for the list
};
```

#### Step 2: Queue Setup Callback
Called when userspace requests buffers (`REQBUFS`).
```c
static int my_queue_setup(struct vb2_queue *q,
                          unsigned int *num_buffers, unsigned int *num_planes,
                          unsigned int sizes[], struct device *alloc_devs[]) {
    struct my_video_dev *dev = vb2_get_drv_priv(q);
    unsigned int size = dev->fmt.fmt.pix.sizeimage;

    if (*num_planes) return sizes[0] < size ? -EINVAL : 0;

    *num_planes = 1;
    sizes[0] = size;
    
    return 0;
}
```

#### Step 3: Buffer Prepare
Called every time a buffer is queued (`QBUF`). Verify size.
```c
static int my_buf_prepare(struct vb2_buffer *vb) {
    struct my_video_dev *dev = vb2_get_drv_priv(vb->vb2_queue);
    unsigned int size = dev->fmt.fmt.pix.sizeimage;

    if (vb2_plane_size(vb, 0) < size) {
        return -EINVAL;
    }
    
    vb2_set_plane_payload(vb, 0, size);
    return 0;
}
```

#### Step 4: Buffer Queue
Called to give the buffer to the driver. Add it to the list.
```c
static void my_buf_queue(struct vb2_buffer *vb) {
    struct my_video_dev *dev = vb2_get_drv_priv(vb->vb2_queue);
    struct vb2_v4l2_buffer *vbuf = to_vb2_v4l2_buffer(vb);
    unsigned long flags;

    spin_lock_irqsave(&dev->slock, flags);
    list_add_tail(&vbuf->list, &dev->buf_list);
    spin_unlock_irqrestore(&dev->slock, flags);
}
```

#### Step 5: Start/Stop Streaming
```c
static int my_start_streaming(struct vb2_queue *q, unsigned int count) {
    struct my_video_dev *dev = vb2_get_drv_priv(q);
    // Enable Hardware / Timer
    return 0;
}

static void my_stop_streaming(struct vb2_queue *q) {
    struct my_video_dev *dev = vb2_get_drv_priv(q);
    struct vb2_v4l2_buffer *vbuf;
    unsigned long flags;

    // Return all buffers to vb2 with error state
    spin_lock_irqsave(&dev->slock, flags);
    while (!list_empty(&dev->buf_list)) {
        vbuf = list_first_entry(&dev->buf_list, struct vb2_v4l2_buffer, list);
        list_del(&vbuf->list);
        vb2_buffer_done(&vbuf->vb2_buf, VB2_BUF_STATE_ERROR);
    }
    spin_unlock_irqrestore(&dev->slock, flags);
}
```

#### Step 6: Registering the Queue (in Probe)
```c
// Init Spinlock & List
spin_lock_init(&dev->slock);
INIT_LIST_HEAD(&dev->buf_list);

// Init Queue
dev->vb_vidq.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
dev->vb_vidq.io_modes = VB2_MMAP | VB2_USERPTR | VB2_READ;
dev->vb_vidq.drv_priv = dev;
dev->vb_vidq.buf_struct_size = sizeof(struct vb2_v4l2_buffer);
dev->vb_vidq.ops = &my_vb2_ops;
dev->vb_vidq.mem_ops = &vb2_vmalloc_memops; // Use vmalloc
dev->vb_vidq.timestamp_flags = V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC;
dev->vb_vidq.lock = &dev->lock; // Use the same lock as IOCTLs

ret = vb2_queue_init(&dev->vb_vidq);
if (ret) return ret;

dev->vdev.queue = &dev->vb_vidq; // Link video_device to queue
```

---

## 💻 Implementation: Buffer IOCTLs

> **Instruction:** Connect the standard IOCTLs to vb2 helpers.

### 👨‍💻 Code Implementation

```c
static const struct v4l2_ioctl_ops my_ioctl_ops = {
    // ... previous ops ...
    .vidioc_reqbufs = vb2_ioctl_reqbufs,
    .vidioc_querybuf = vb2_ioctl_querybuf,
    .vidioc_qbuf = vb2_ioctl_qbuf,
    .vidioc_dqbuf = vb2_ioctl_dqbuf,
    .vidioc_streamon = vb2_ioctl_streamon,
    .vidioc_streamoff = vb2_ioctl_streamoff,
};
```

---

## 🔬 Lab Exercise: Lab 153.1 - Buffer Allocation

### 1. Lab Objectives
- Compile and load.
- Request buffers using `v4l2-ctl`.

### 2. Step-by-Step Guide
1.  Load module.
2.  Request 4 buffers:
    ```bash
    v4l2-ctl -d /dev/video0 --reqbufs=4
    ```
3.  Check `dmesg`. You should see calls to `queue_setup`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: User Pointer (`VB2_USERPTR`)
- **Goal:** Use memory allocated by userspace (malloc).
- **Task:**
    1.  Ensure `VB2_USERPTR` is in `io_modes`.
    2.  Use a test app that mallocs a buffer and passes it to the driver.

### Lab 3: DMABUF Import
- **Goal:** Zero-copy sharing (e.g., GPU -> Camera).
- **Task:**
    1.  Add `VB2_DMABUF` to `io_modes`.
    2.  This requires `vb2_dma_contig` or `vb2_dma_sg`. `vmalloc` doesn't support DMABUF well.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Device or resource busy" on REQBUFS
*   **Cause:** Buffers already allocated. You must request 0 buffers to free them first.

#### 2. "Invalid Argument" on QBUF
*   **Cause:** Buffer index out of range.
*   **Cause:** Buffer not mapped.
*   **Cause:** `buf_prepare` returned error (Size mismatch).

---

## ⚡ Optimization & Best Practices

### `vb2_buffer_done`
*   This function marks a buffer as filled and wakes up userspace (`DQBUF`).
*   **Context:** Can be called from Interrupt Context (ISR).
*   **State:** `VB2_BUF_STATE_DONE`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we need `spin_lock_irqsave` in `buf_queue`?
    *   **A:** `buf_queue` is called with the mutex held (Process Context), but the ISR (Interrupt Context) also accesses the list to remove buffers. We need a spinlock to protect against the ISR.
2.  **Q:** What is `VB2_MMAP`?
    *   **A:** Memory Mapping. The driver allocates the memory (kernel space) and maps it into the user process's virtual address space. This is the most common and efficient method for simple cameras.

### Challenge Task
> **Task:** "The Timestamp".
> *   In the next day's streaming logic, ensure you set `vb->timestamp = ktime_get_ns()`.
> *   Verify userspace receives the correct timestamp.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/media/v4l2-videobuf2.rst](https://www.kernel.org/doc/html/latest/driver-api/media/v4l2-videobuf2.html)

---
