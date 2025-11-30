# Day 179: USB Mass Storage & HID
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
1.  **Explain** the USB Mass Storage Class (MSC) and Bulk-Only Transport (BOT).
2.  **Explain** the USB HID Class and Report Descriptors.
3.  **Write** a simple HID driver (`hid_driver`) to parse custom reports.
4.  **Understand** how `usb-storage` bridges USB to the SCSI subsystem.
5.  **Debug** HID reports using `hid-recorder` and `evtest`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
    *   USB HID Device (Mouse/Keyboard).
*   **Software Required:**
    *   `hid-tools` (optional).
*   **Prior Knowledge:**
    *   Day 176 (USB Basics).
    *   Day 147 (Input Subsystem).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: USB Mass Storage (MSC)
*   **Protocol:** Wraps SCSI commands (Read, Write, Inquiry) inside USB Bulk packets.
*   **CBW (Command Block Wrapper):** Header sent by Host. Contains the SCSI CDB.
*   **Data:** Bulk Data.
*   **CSW (Command Status Wrapper):** Footer sent by Device. Success/Fail.
*   **Linux:** `usb-storage.ko` handles the USB part and registers a SCSI Host. `sd_mod` (SCSI Disk) handles the disk part.

### 🔹 Part 2: USB HID (Human Interface Device)
*   **Report Descriptor:** A complex byte-code that describes the data (e.g., "3 buttons, X/Y axis 8-bit").
*   **Reports:** The actual data packets.
*   **Linux:** `usbhid` is the generic transport. `hid-generic` handles standard devices. Custom drivers (`hid-sony`, `hid-logitech`) handle quirks.

---

## 💻 Implementation: Custom HID Driver

> **Instruction:** We will write a driver that binds to a specific HID device and parses raw events.

### 👨‍💻 Code Implementation

#### Step 1: ID Table
```c
#include <linux/hid.h>
#include <linux/module.h>

static const struct hid_device_id my_hid_table[] = {
    { HID_USB_DEVICE(0x1234, 0x5678) },
    { }
};
MODULE_DEVICE_TABLE(hid, my_hid_table);
```

#### Step 2: Raw Event Callback
Called when a report arrives.
```c
static int my_raw_event(struct hid_device *hdev, struct hid_report *report,
                        u8 *data, int size) {
    // data[0] is Report ID
    // data[1...] is Payload
    
    if (size < 2) return 0;
    
    pr_info("HID Event: %02x %02x\n", data[0], data[1]);
    
    // If this is a custom vendor report, handle it
    if (data[0] == 0xAB) {
        // Do something custom
        return 1; // Swallow event (don't pass to generic)
    }
    
    return 0; // Pass to generic (Input subsystem)
}
```

#### Step 3: Driver Structure
```c
static struct hid_driver my_hid_driver = {
    .name = "my_hid",
    .id_table = my_hid_table,
    .raw_event = my_raw_event,
};

module_hid_driver(my_hid_driver);
```

---

## 💻 Implementation: Sending HID Reports

> **Instruction:** Send a command to the device (Output Report).

### 👨‍💻 Code Implementation

```c
static int send_command(struct hid_device *hdev, u8 cmd) {
    u8 *buf;
    int ret;
    
    buf = kzalloc(64, GFP_KERNEL);
    if (!buf) return -ENOMEM;
    
    buf[0] = 0x02; // Report ID
    buf[1] = cmd;
    
    ret = hid_hw_output_report(hdev, buf, 64);
    
    kfree(buf);
    return ret;
}
```

---

## 🔬 Lab Exercise: Lab 179.1 - HID Sniffer

### 1. Lab Objectives
- Bind the custom driver to a real mouse.
- Print X/Y coordinates to dmesg.

### 2. Step-by-Step Guide
1.  **Find ID:** `lsusb`.
2.  **Update ID Table.**
3.  **Load:** `insmod my_hid.ko`.
4.  **Move Mouse:**
    *   `dmesg` should flood with "HID Event: ...".
5.  **Decode:**
    *   Usually Byte 1 is Buttons, Byte 2 is X, Byte 3 is Y.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Force Feedback (FF)
- **Goal:** Vibrate a game controller.
- **Task:**
    1.  Implement `probe`.
    2.  `input_ff_create_memless(input_dev, NULL, my_ff_play)`.
    3.  In `my_ff_play`, send Output Report to device to enable motors.

### Lab 3: Mass Storage Quirks
- **Goal:** Understand `unusual_devs.h`.
- **Task:**
    1.  Read `drivers/usb/storage/unusual_devs.h`.
    2.  See how flags like `US_FL_FIX_CAPACITY` or `US_FL_IGNORE_RESIDUE` fix buggy USB sticks.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Driver not binding
*   **Cause:** `hid-generic` bound to it first.
*   **Fix:** You don't need to unbind `hid-generic` manually if your driver is loaded *before* the device is plugged in, or if you use `HID_QUIRK_HAVE_SPECIAL_DRIVER`. But usually, `modprobe` handles priority via `modules.alias`.

#### 2. "Report descriptor too short"
*   **Cause:** Device has a buggy descriptor.
*   **Fix:** Implement `rdesc_fixup` callback to patch the descriptor bytes in memory before parsing.

---

## ⚡ Optimization & Best Practices

### `hid_hw_start`
*   In `probe`, you must call `hid_hw_start(hdev, HID_CONNECT_DEFAULT)`.
*   This tells the HID core to start polling the device (if Interrupt endpoint) and connect it to the Input subsystem.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is HID preferred over raw USB for custom devices?
    *   **A:** Driverless support on Windows/macOS/Linux (using generic HID drivers). No need to install a kernel module for simple data exchange.
2.  **Q:** What is the difference between `hid_hw_output_report` and `hid_hw_raw_request`?
    *   **A:** `output_report` uses the Interrupt OUT endpoint (if available). `raw_request` uses the Control Endpoint (Set_Report).

### Challenge Task
> **Task:** "The LED Controller".
> *   Find a USB keyboard with LEDs (Num/Caps Lock).
> *   Write a driver that cycles the LEDs (Num -> Caps -> Scroll) every second.
> *   Use `hid_hw_request` to send Set_Report commands.

---

## 📚 Further Reading & References
- [Kernel Documentation: hid/hid-transport.rst](https://www.kernel.org/doc/html/latest/hid/hid-transport.html)
- [USB HID Usage Tables](https://usb.org/sites/default/files/hut1_2.pdf)

---
