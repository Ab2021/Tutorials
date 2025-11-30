# Day 177: USB URBs and Transfers
## Phase 2: Linux Kernel & Device Drivers | Week 26: USB Device Drivers

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
1.  **Allocate** and **Free** URBs (`usb_alloc_urb`, `usb_free_urb`).
2.  **Perform** Synchronous Transfers (`usb_bulk_msg`, `usb_control_msg`).
3.  **Perform** Asynchronous Transfers (URB Submission).
4.  **Implement** a Completion Callback.
5.  **Debug** URB errors (Status codes).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
    *   USB Device (e.g., Flash Drive for Bulk tests).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 176 (USB Basics).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Synchronous vs Asynchronous
*   **Synchronous:** `usb_bulk_msg`. Blocks the calling thread until done or timeout. Good for simple commands. Bad for high throughput.
*   **Asynchronous:** `usb_submit_urb`. Returns immediately. Calls a callback when done. Essential for streaming (Audio/Video/Network).

### 🔹 Part 2: URB Lifecycle
1.  **Created:** `usb_alloc_urb`.
2.  **Initialized:** `usb_fill_bulk_urb`.
3.  **Submitted:** `usb_submit_urb`. (Passed to Host Controller).
4.  **Active:** Hardware processing.
5.  **Completed:** Callback runs. Status updated (0 = Success, -EPIPE = Stall).
6.  **Resubmitted:** (Optional) If streaming, submit again in callback.
7.  **Freed:** `usb_free_urb`.

---

## 💻 Implementation: Synchronous Bulk Read

> **Instruction:** Read 512 bytes from a Bulk IN endpoint.

### 👨‍💻 Code Implementation

```c
#define BULK_EP_IN 0x81 // Example Endpoint Address

static int read_sync(struct usb_device *udev) {
    int ret;
    int actual_length;
    u8 *buf;

    buf = kmalloc(512, GFP_KERNEL);
    if (!buf) return -ENOMEM;

    // Pipe: usb_rcvbulkpipe(udev, EndpointAddr)
    ret = usb_bulk_msg(udev, usb_rcvbulkpipe(udev, BULK_EP_IN),
                       buf, 512, &actual_length, 5000); // 5000ms timeout

    if (ret) {
        pr_err("Bulk read failed: %d\n", ret);
    } else {
        pr_info("Read %d bytes: %02x %02x...\n", actual_length, buf[0], buf[1]);
    }

    kfree(buf);
    return ret;
}
```

---

## 💻 Implementation: Asynchronous Bulk Write

> **Instruction:** Send data without blocking.

### 👨‍💻 Code Implementation

#### Step 1: Completion Callback
```c
static void write_callback(struct urb *urb) {
    // Check status
    if (urb->status) {
        pr_err("URB failed: %d\n", urb->status);
    } else {
        pr_info("URB sent %d bytes\n", urb->actual_length);
    }

    // Free buffer allocated in submitter
    kfree(urb->transfer_buffer);
    
    // Free URB (or resubmit)
    usb_free_urb(urb);
}
```

#### Step 2: Submission
```c
static int write_async(struct usb_device *udev) {
    struct urb *urb;
    u8 *buf;
    int ret;

    urb = usb_alloc_urb(0, GFP_KERNEL); // 0 ISO packets
    if (!urb) return -ENOMEM;

    buf = kmalloc(512, GFP_KERNEL);
    // Fill buf...

    usb_fill_bulk_urb(urb, udev, usb_sndbulkpipe(udev, 0x02),
                      buf, 512, write_callback, NULL);

    ret = usb_submit_urb(urb, GFP_KERNEL);
    if (ret) {
        usb_free_urb(urb);
        kfree(buf);
        return ret;
    }

    return 0;
}
```

---

## 🔬 Lab Exercise: Lab 177.1 - Blinking an LED (Control Msg)

### 1. Lab Objectives
- Use `usb_control_msg` to send a vendor command.
- (Requires a programmable USB device like an Arduino or STM32, or just simulate).

### 2. Step-by-Step Guide
1.  **Command:** RequestType=Vendor, Request=0x01, Value=1 (On).
2.  **Code:**
    ```c
    usb_control_msg(udev, usb_sndctrlpipe(udev, 0),
                    0x01, USB_TYPE_VENDOR | USB_DIR_OUT,
                    1, 0, NULL, 0, 1000);
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Anchoring URBs
- **Goal:** Clean shutdown.
- **Task:**
    1.  Use `usb_anchor`.
    2.  `usb_anchor_urb(urb, &my_anchor)`.
    3.  In disconnect: `usb_kill_anchored_urbs(&my_anchor)`.
    4.  This ensures no callbacks run after the driver is unloaded.

### Lab 3: Isochronous Transfer
- **Goal:** Setup an ISO URB.
- **Task:**
    1.  `usb_alloc_urb(packets, GFP_KERNEL)`.
    2.  Set `urb->iso_frame_desc[i].offset` and `length`.
    3.  Submit.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. -EPIPE (32)
*   **Cause:** Endpoint Stalled (Halted).
*   **Fix:** `usb_clear_halt(udev, pipe)`.

#### 2. -ESHUTDOWN
*   **Cause:** Device disconnected or Host Controller disabled.

#### 3. -EINPROGRESS
*   **Cause:** URB is still running. (Normal for `urb->status` while active).

---

## ⚡ Optimization & Best Practices

### DMA Buffers
*   `usb_bulk_msg` automatically creates a DMA mapping if the buffer is `kmalloc`'d.
*   **Never** pass a stack variable (`u8 buf[64]`) to USB functions. DMA cannot access the stack on some archs, and it violates cache coherency rules.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What happens if I `kfree` the buffer before the callback runs?
    *   **A:** Use-after-free. The Host Controller might still be writing to that memory. Crash!
2.  **Q:** Why `GFP_ATOMIC` in completion callback?
    *   **A:** The callback runs in Interrupt Context (SoftIRQ). You cannot sleep. If you resubmit an URB, use `GFP_ATOMIC`.

### Challenge Task
> **Task:** "The Loopback".
> *   Submit a Bulk IN URB.
> *   In its callback, copy the data to a new Bulk OUT URB and submit it.
> *   Create a "Ping Pong" chain.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/usb/URB.rst](https://www.kernel.org/doc/html/latest/driver-api/usb/URB.html)

---
