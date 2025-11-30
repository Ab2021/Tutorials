# Day 176: USB Device Drivers Basics
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
1.  **Explain** the USB Architecture (Host, Device, Endpoints, Pipes).
2.  **Register** a `usb_driver` with the kernel.
3.  **Understand** the URB (USB Request Block) lifecycle.
4.  **Implement** `probe` and `disconnect` callbacks for a USB device.
5.  **Use** `lsusb` and `usb-devices` to inspect hardware.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
    *   A USB device (Mouse, Keyboard, or Flash Drive) for testing.
*   **Software Required:**
    *   `usbutils` (`lsusb`).
*   **Prior Knowledge:**
    *   Day 128 (Char Drivers).
    *   Basic USB concepts (Endpoints).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: USB Architecture
*   **Host Controller:** The master (e.g., xHCI). Initiates all transfers.
*   **Device:** The slave. Responds to requests.
*   **Configuration:** A set of interfaces (e.g., "High Power" vs "Low Power").
*   **Interface:** A logical function (e.g., "Mouse" or "Audio"). Drivers bind to Interfaces, not Devices.
*   **Endpoint:** A buffer on the device.
    *   **Control (EP0):** Setup packets.
    *   **Bulk:** Large data (Disk).
    *   **Interrupt:** Small, latency-sensitive (Mouse).
    *   **Isochronous:** Streaming (Audio/Video).

### 🔹 Part 2: The URB (USB Request Block)
The fundamental unit of communication.
1.  **Allocate:** `usb_alloc_urb`.
2.  **Fill:** `usb_fill_bulk_urb`.
3.  **Submit:** `usb_submit_urb` (Async).
4.  **Complete:** Callback function is called.

---

## 💻 Implementation: The Skeleton Driver

> **Instruction:** Create a driver that binds to a specific USB ID.

### 👨‍💻 Code Implementation

#### Step 1: ID Table
```c
#include <linux/module.h>
#include <linux/usb.h>

#define USB_VENDOR_ID  0x1234 // Replace with real ID
#define USB_PRODUCT_ID 0x5678

static const struct usb_device_id my_table[] = {
    { USB_DEVICE(USB_VENDOR_ID, USB_PRODUCT_ID) },
    { } // Terminating entry
};
MODULE_DEVICE_TABLE(usb, my_table);
```

#### Step 2: Probe & Disconnect
```c
static int my_probe(struct usb_interface *interface, const struct usb_device_id *id) {
    struct usb_device *udev = interface_to_usbdev(interface);
    
    dev_info(&interface->dev, "USB Device Probed: VID=0x%04x PID=0x%04x\n",
             le16_to_cpu(udev->descriptor.idVendor),
             le16_to_cpu(udev->descriptor.idProduct));
             
    return 0;
}

static void my_disconnect(struct usb_interface *interface) {
    dev_info(&interface->dev, "USB Device Disconnected\n");
}
```

#### Step 3: Driver Structure
```c
static struct usb_driver my_driver = {
    .name = "my_usb_driver",
    .probe = my_probe,
    .disconnect = my_disconnect,
    .id_table = my_table,
};

module_usb_driver(my_driver);
```

---

## 🔬 Lab Exercise: Lab 176.1 - Binding to a Mouse

### 1. Lab Objectives
- Find the VID/PID of your USB mouse.
- Modify the driver to bind to it.
- **Warning:** This will detach the default `usbhid` driver, so your mouse will stop working! Use a second mouse or SSH.

### 2. Step-by-Step Guide
1.  **Find ID:**
    ```bash
    lsusb
    # Bus 001 Device 004: ID 046d:c077 Logitech, Inc. Mouse
    ```
2.  **Update Code:**
    ```c
    #define USB_VENDOR_ID  0x046d
    #define USB_PRODUCT_ID 0xc077
    ```
3.  **Load:** `insmod my_usb.ko`.
4.  **Check:** `dmesg`.
    *   If `usbhid` grabbed it first, you might need to unbind it manually:
    *   `echo "1-1:1.0" > /sys/bus/usb/drivers/usbhid/unbind`
    *   `echo "1-1:1.0" > /sys/bus/usb/drivers/my_usb_driver/bind`

---

## 🧪 Additional / Advanced Labs

### Lab 2: Inspecting Endpoints
- **Goal:** Iterate over endpoints in `probe`.
- **Task:**
    1.  Access `interface->cur_altsetting`.
    2.  Loop `desc->bNumEndpoints`.
    3.  Print Address (`bEndpointAddress`) and Attributes (`bmAttributes`).

### Lab 3: Dynamic ID
- **Goal:** Bind to a new device without recompiling.
- **Task:**
    1.  Load driver.
    2.  `echo "1234 5678" > /sys/bus/usb/drivers/my_usb_driver/new_id`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Driver doesn't probe
*   **Cause:** Another driver (like `usb-storage` or `usbhid`) claimed it first.
*   **Fix:** Unload the other driver or use sysfs unbind.

#### 2. "Unknown symbol"
*   **Cause:** `CONFIG_USB` not enabled (unlikely on PC).

---

## ⚡ Optimization & Best Practices

### `interface_to_usbdev`
*   Helper to get the `struct usb_device` from `struct usb_interface`.
*   Remember: `usb_device` represents the whole physical stick. `usb_interface` is just one function (e.g., Keyboard part of a Combo dongle).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is Endpoint 0?
    *   **A:** The Control Endpoint. Every USB device has it. Used for enumeration and configuration.
2.  **Q:** Why do we bind to Interfaces and not Devices?
    *   **A:** Because a single USB device (Composite Device) can act as multiple things simultaneously (e.g., Audio + HID + Storage). Different drivers handle different interfaces.

### Challenge Task
> **Task:** "The Endpoint Finder".
> *   Write a function `find_bulk_endpoints`.
> *   It should scan the interface and store the addresses of the first Bulk IN and Bulk OUT endpoints it finds.
> *   Print them to dmesg.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/usb/index.rst](https://www.kernel.org/doc/html/latest/driver-api/usb/index.html)
- [USB Specification 2.0](https://www.usb.org/document-library/usb-20-specification)

---
