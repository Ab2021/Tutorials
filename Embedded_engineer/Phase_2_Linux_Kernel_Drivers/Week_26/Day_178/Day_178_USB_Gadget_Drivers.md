# Day 178: USB Gadget Drivers
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
1.  **Explain** the difference between Host and Gadget (Device) mode.
2.  **Register** a `usb_gadget_driver`.
3.  **Define** Descriptors (Device, Config, Interface, Endpoint) for the gadget.
4.  **Handle** Setup Packets (`setup` callback).
5.  **Enable** Endpoints and queue requests (`usb_ep_enable`, `usb_ep_queue`).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   A board with a USB Device/OTG port (Raspberry Pi Zero/4, BeagleBone).
    *   Or `dummy_hcd` module on PC (simulates a gadget controller).
*   **Software Required:**
    *   `modprobe dummy_hcd`.
*   **Prior Knowledge:**
    *   Day 176 (USB Arch).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Host vs Gadget
*   **Host:** Controls the bus. Initiates transfers. (PC).
*   **Gadget:** Responds to the host. (Phone, Flash Drive).
*   **OTG (On-The-Go):** Can switch roles.

### 🔹 Part 2: The Gadget API
*   **UDC (USB Device Controller):** The hardware driver (e.g., `dwc2`).
*   **Composite Framework:** Allows creating complex gadgets (Ethernet + Serial).
*   **Function:** A single capability (e.g., `f_mass_storage`).

---

## 💻 Implementation: The "Zero" Gadget (Simplified)

> **Instruction:** We will write a legacy gadget driver (pre-composite) to understand the core.

### 👨‍💻 Code Implementation

#### Step 1: Descriptors
```c
static struct usb_device_descriptor device_desc = {
    .bLength = sizeof(device_desc),
    .bDescriptorType = USB_DT_DEVICE,
    .bcdUSB = cpu_to_le16(0x0200),
    .bDeviceClass = USB_CLASS_VENDOR_SPEC,
    .idVendor = cpu_to_le16(0x1234),
    .idProduct = cpu_to_le16(0x5678),
    .bNumConfigurations = 1,
};

static struct usb_endpoint_descriptor ep_desc = {
    .bLength = USB_DT_ENDPOINT_SIZE,
    .bDescriptorType = USB_DT_ENDPOINT,
    .bEndpointAddress = USB_DIR_IN | 1,
    .bmAttributes = USB_ENDPOINT_XFER_BULK,
    .wMaxPacketSize = cpu_to_le16(512),
};
```

#### Step 2: Bind Callback
Called when the UDC driver loads.
```c
static int my_bind(struct usb_gadget *gadget, struct usb_gadget_driver *driver) {
    struct usb_ep *ep;

    // Find an available endpoint
    ep = usb_ep_autoconfig(gadget, &ep_desc);
    if (!ep) return -ENODEV;

    ep->driver_data = ep; // Save for later
    return 0;
}
```

#### Step 3: Setup Callback (EP0)
Handles Control Transfers from Host.
```c
static int my_setup(struct usb_gadget *gadget, const struct usb_ctrlrequest *ctrl) {
    struct usb_request *req = gadget->ep0->driver_data;
    u16 w_value = le16_to_cpu(ctrl->wValue);
    
    // Handle Standard Requests (Get Descriptor, Set Config)
    // Usually delegated to a helper or handled manually
    // For simplicity, we just ACK vendor requests
    if ((ctrl->bRequestType & USB_TYPE_MASK) == USB_TYPE_VENDOR) {
        return 0; // Success
    }
    
    return -EOPNOTSUPP;
}
```

#### Step 4: Driver Structure
```c
static struct usb_gadget_driver my_driver = {
    .function = "MyGadget",
    .max_speed = USB_SPEED_HIGH,
    .bind = my_bind,
    .setup = my_setup,
    .driver = {
        .name = "my_gadget",
    },
};

module_usb_gadget_driver(my_driver);
```

---

## 🔬 Lab Exercise: Lab 178.1 - Testing with Dummy HCD

### 1. Lab Objectives
- Load `dummy_hcd` (Virtual UDC).
- Load our gadget driver.
- Verify it enumerates on the host side.

### 2. Step-by-Step Guide
1.  **Load Dummy:**
    ```bash
    modprobe dummy_hcd
    ```
2.  **Load Gadget:**
    ```bash
    insmod my_gadget.ko
    ```
3.  **Check Host:**
    ```bash
    lsusb
    # Bus 00X Device 00Y: ID 1234:5678 ...
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: ConfigFS Gadget (Modern Way)
- **Goal:** Create a gadget from userspace without writing C code.
- **Task:**
    1.  Mount configfs: `mount -t configfs none /sys/kernel/config`.
    2.  `mkdir /sys/kernel/config/usb_gadget/g1`.
    3.  Set VID/PID.
    4.  `mkdir functions/mass_storage.0`.
    5.  Link function to config.
    6.  Enable UDC.

### Lab 3: Sending Data
- **Goal:** Send "Hello" when host requests.
- **Task:**
    1.  In `bind`, allocate a request (`usb_ep_alloc_request`).
    2.  Fill buffer with "Hello".
    3.  Enable EP.
    4.  Queue request (`usb_ep_queue`).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "No such device" on insmod
*   **Cause:** No UDC available.
*   **Fix:** Ensure `dummy_hcd` is loaded or you are on a real board with a UDC driver enabled.

#### 2. Host sees "Device Descriptor Request Failed"
*   **Cause:** `setup` callback didn't handle `USB_REQ_GET_DESCRIPTOR` correctly.
*   **Fix:** Use `composite_setup` helper or implement standard request handling carefully.

---

## ⚡ Optimization & Best Practices

### Composite Framework
*   Don't write raw gadget drivers unless necessary.
*   Use the **Composite Framework** (`usb_composite_driver`). It handles standard requests, config binding, and function management for you.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `usb_ep_autoconfig`?
    *   **A:** It asks the UDC "Do you have an endpoint that matches these requirements (Bulk, IN)?" and returns it.
2.  **Q:** Why do we need `dummy_hcd`?
    *   **A:** It connects a virtual Gadget Controller to a virtual Host Controller inside the same kernel. Allows testing gadget drivers on a PC.

### Challenge Task
> **Task:** "The Keyboard Emulator".
> *   Use ConfigFS.
> *   Create a HID gadget.
> *   Write a script to send keystrokes (e.g., "Hello World") to the host.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/usb/gadget.rst](https://www.kernel.org/doc/html/latest/driver-api/usb/gadget.html)
- [Linux USB Gadget ConfigFS](https://www.kernel.org/doc/Documentation/usb/gadget_configfs.txt)

---
