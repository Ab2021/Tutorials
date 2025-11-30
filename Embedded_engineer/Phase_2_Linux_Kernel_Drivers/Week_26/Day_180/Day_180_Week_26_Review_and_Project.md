# Day 180: Week 26 Review & Project - The USB Thermometer
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
1.  **Synthesize** Week 26 concepts (USB Arch, URBs, Gadget, HID).
2.  **Architect** a complete USB system (Device + Host).
3.  **Implement** a USB Gadget that simulates a sensor.
4.  **Implement** a USB Host Driver that reads from that sensor.
5.  **Debug** the communication using Wireshark (usbmon).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC (Host).
    *   Raspberry Pi or `dummy_hcd` (Gadget).
*   **Software Required:**
    *   `wireshark`, `usbmon`.
*   **Prior Knowledge:**
    *   Week 26 Content.

---

## 🔄 Week 26 Review

### 1. USB Basics (Day 176)
*   **Host/Device:** Master/Slave.
*   **Endpoints:** Control, Bulk, Int, Iso.
*   **Driver:** `usb_driver`.

### 2. URBs (Day 177)
*   **Async:** `usb_submit_urb`.
*   **Sync:** `usb_bulk_msg`.

### 3. Gadget (Day 178)
*   **UDC:** Hardware driver.
*   **Function:** `usb_ep_queue`.

### 4. HID/MSC (Day 179)
*   **Classes:** Standard protocols.
*   **HID:** Reports.

---

## 🛠️ Project: The "USB Thermometer"

### 📋 Project Requirements
1.  **Gadget Side (`therm_gadget.ko`):**
    *   Simulates a thermometer.
    *   Vendor ID: 0x1234, Product ID: 0xAAAA.
    *   Endpoint: Bulk IN.
    *   Data: Sends "25.5 C" every time the host requests.
2.  **Host Side (`therm_driver.ko`):**
    *   Binds to 0x1234:0xAAAA.
    *   Registers a sysfs file `/sys/class/therm/temp`.
    *   When read, submits a Bulk IN URB, waits for data, and returns it.

---

## 💻 Implementation: Step-by-Step Guide

### 🔹 Phase 1: The Gadget

**`therm_gadget.c`**
```c
// ... Descriptors (Bulk IN) ...

static void therm_tx_complete(struct usb_ep *ep, struct usb_request *req) {
    // Free buffer or re-queue
}

// Handle Bulk IN token from Host
// Note: In legacy gadget API, we usually queue a request *before* the host asks.
// Or we use a thread to keep the FIFO full.
// For simplicity, let's queue one packet "25.5 C" in bind.
static int therm_bind(struct usb_gadget *gadget, struct usb_gadget_driver *driver) {
    // ... autoconfig ep ...
    req = usb_ep_alloc_request(ep, GFP_KERNEL);
    strcpy(req->buf, "25.5 C");
    req->length = 6;
    req->complete = therm_tx_complete;
    usb_ep_queue(ep, req, GFP_KERNEL);
    return 0;
}
```

### 🔹 Phase 2: The Host Driver

**`therm_driver.c`**
```c
struct therm_dev {
    struct usb_device *udev;
    struct urb *urb;
    u8 *buf;
    struct completion done;
};

static void therm_callback(struct urb *urb) {
    struct therm_dev *dev = urb->context;
    complete(&dev->done);
}

static ssize_t temp_show(struct device *d, struct device_attribute *attr, char *buf) {
    struct usb_interface *intf = to_usb_interface(d);
    struct therm_dev *dev = usb_get_intfdata(intf);
    
    // 1. Submit URB
    usb_fill_bulk_urb(dev->urb, dev->udev, usb_rcvbulkpipe(dev->udev, 0x81),
                      dev->buf, 64, therm_callback, dev);
    init_completion(&dev->done);
    usb_submit_urb(dev->urb, GFP_KERNEL);
    
    // 2. Wait
    wait_for_completion(&dev->done);
    
    // 3. Return Data
    return scnprintf(buf, PAGE_SIZE, "%s\n", dev->buf);
}
static DEVICE_ATTR_RO(temp);
```

### 🔹 Phase 3: Testing

> **Instruction:** Use `dummy_hcd` if you don't have two boards.

1.  **Load Dummy:** `modprobe dummy_hcd`.
2.  **Load Gadget:** `insmod therm_gadget.ko`.
3.  **Load Host:** `insmod therm_driver.ko`.
4.  **Read:**
    ```bash
    cat /sys/bus/usb/drivers/therm_driver/1-1:1.0/temp
    # Output: 25.5 C
    ```

---

## 📈 Grading Rubric

| Criteria | Excellent (A) | Good (B) | Needs Improvement (C) |
| :--- | :--- | :--- | :--- |
| **Completeness** | Both Host and Gadget work. | Only one side works. | Neither works. |
| **Robustness** | Handles disconnects gracefully. | Crashes on unplug. | Leaks memory. |
| **Code Style** | Proper indentation, comments. | Messy. | No comments. |

---

## 🔮 Looking Ahead: Phase 3
**Congratulations! You have completed Phase 2.**
You have written drivers for:
*   Char Devices, GPIO, Interrupts.
*   I2C, SPI, UART.
*   Block Devices, Filesystems.
*   Network, USB, Audio (ALSA), Video (V4L2).

**Next Step:**
We will transition to **Phase 3: Embedded Android**.
We will take these drivers and expose them to the Android Framework via HALs (Hardware Abstraction Layers).

---
