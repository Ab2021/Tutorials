# Day 135: Platform Device Model
## Phase 2: Linux Kernel & Device Drivers | Week 20: Platform Drivers & Device Tree

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
1.  **Explain** the separation between "Device" (Hardware) and "Driver" (Software).
2.  **Implement** a `platform_driver` with `probe` and `remove` callbacks.
3.  **Register** a `platform_device` manually (the "Old Way" before Device Tree).
4.  **Demonstrate** how the Kernel binds a Driver to a Device using name matching.
5.  **Pass** resources (Memory addresses, IRQs) to the driver via `platform_data`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 128 (Char Drivers).
    *   C Structures and Pointers.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Linux Device Model
In the early days, drivers looked for hardware at hardcoded addresses.
*   **Problem:** Not portable. The same UART might be at `0x1000` on Chip A and `0x2000` on Chip B.
*   **Solution:** The Device Model.
    *   **Bus:** A channel where devices live (PCI, USB, I2C, SPI).
    *   **Device:** The physical hardware description (Address, IRQ).
    *   **Driver:** The code that knows how to talk to the hardware.

### 🔹 Part 2: The Platform Bus
What about on-chip peripherals (UART, GPIO, RTC) that are not on a discoverable bus like PCI?
*   **The Virtual Bus:** Linux created the "Platform Bus" for these devices.
*   **Matching:** The bus core matches `platform_device.name` with `platform_driver.driver.name`. If they match, `probe()` is called.

### 🔹 Part 3: The Probe Function
`probe()` is the "Constructor" of your driver instance.
*   It is called **only** when a matching device is found.
*   **Responsibilities:**
    1.  Get resources (Memory, IRQ).
    2.  Register Character Device / Misc Device.
    3.  Initialize Hardware.

---

## 💻 Implementation: The Dummy Platform Driver

> **Instruction:** We will create two modules.
> 1.  `pcd_device_setup.c`: Registers 2 dummy devices.
> 2.  `pcd_platform_driver.c`: The driver that handles them.

### 👨‍💻 Code Implementation

#### Step 1: Device Registration (`pcd_device_setup.c`)

```c
#include <linux/module.h>
#include <linux/platform_device.h>

// Define Resources for Device 1
struct resource res_1[] = {
    [0] = {
        .start = 0x1000,
        .end   = 0x101F,
        .flags = IORESOURCE_MEM,
    },
    [1] = {
        .start = 10, // Fake IRQ
        .end   = 10,
        .flags = IORESOURCE_IRQ,
    }
};

// Define Resources for Device 2
struct resource res_2[] = {
    [0] = {
        .start = 0x2000,
        .end   = 0x201F,
        .flags = IORESOURCE_MEM,
    },
    [1] = {
        .start = 11,
        .end   = 11,
        .flags = IORESOURCE_IRQ,
    }
};

// Release callback (Required to avoid warnings)
void pcd_release(struct device *dev) {
    printk(KERN_INFO "Device released\n");
}

// Declare Devices
struct platform_device pdev_1 = {
    .name = "pseudo-char-device", // MUST MATCH DRIVER NAME
    .id = 0,
    .num_resources = ARRAY_SIZE(res_1),
    .resource = res_1,
    .dev = { .release = pcd_release }
};

struct platform_device pdev_2 = {
    .name = "pseudo-char-device",
    .id = 1,
    .num_resources = ARRAY_SIZE(res_2),
    .resource = res_2,
    .dev = { .release = pcd_release }
};

static int __init pcd_dev_init(void) {
    platform_device_register(&pdev_1);
    platform_device_register(&pdev_2);
    printk(KERN_INFO "PCD Devices Registered\n");
    return 0;
}

static void __exit pcd_dev_exit(void) {
    platform_device_unregister(&pdev_1);
    platform_device_unregister(&pdev_2);
    printk(KERN_INFO "PCD Devices Unregistered\n");
}

module_init(pcd_dev_init);
module_exit(pcd_dev_exit);
MODULE_LICENSE("GPL");
```

#### Step 2: Platform Driver (`pcd_platform_driver.c`)

```c
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/mod_devicetable.h>

// Probe: Called when device is found
static int pcd_probe(struct platform_device *pdev) {
    struct resource *res;
    
    printk(KERN_INFO "PCD Driver: Probe called for device %s.%d\n", pdev->name, pdev->id);
    
    // Get Memory Resource
    res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
    if (!res) {
        printk(KERN_ERR "PCD Driver: No MEM resource\n");
        return -ENODEV;
    }
    printk(KERN_INFO "PCD Driver: MEM Start=0x%llx, End=0x%llx\n", res->start, res->end);
    
    // Get IRQ Resource
    int irq = platform_get_irq(pdev, 0);
    printk(KERN_INFO "PCD Driver: IRQ=%d\n", irq);
    
    return 0;
}

// Remove: Called when device is removed
static int pcd_remove(struct platform_device *pdev) {
    printk(KERN_INFO "PCD Driver: Remove called for device %s.%d\n", pdev->name, pdev->id);
    return 0;
}

// Driver Structure
static struct platform_driver pcd_driver = {
    .probe = pcd_probe,
    .remove = pcd_remove,
    .driver = {
        .name = "pseudo-char-device", // MUST MATCH DEVICE NAME
        .owner = THIS_MODULE,
    }
};

static int __init pcd_driver_init(void) {
    return platform_driver_register(&pcd_driver);
}

static void __exit pcd_driver_exit(void) {
    platform_driver_unregister(&pcd_driver);
}

module_init(pcd_driver_init);
module_exit(pcd_driver_exit);
MODULE_LICENSE("GPL");
```

---

## 💻 Implementation: Testing the Binding

> **Instruction:** Load the modules and observe the dmesg logs.

### 👨‍💻 Command Line Steps

#### Step 1: Load Driver First
```bash
insmod pcd_platform_driver.ko
```
*   **Result:** Nothing happens. No devices exist yet.

#### Step 2: Load Devices
```bash
insmod pcd_device_setup.ko
```
*   **Result:**
    ```text
    PCD Devices Registered
    PCD Driver: Probe called for device pseudo-char-device.0
    PCD Driver: MEM Start=0x1000, End=0x101F
    PCD Driver: Probe called for device pseudo-char-device.1
    PCD Driver: MEM Start=0x2000, End=0x201F
    ```
    **Success!** The kernel matched the names and called probe twice.

#### Step 3: Unload Devices
```bash
rmmod pcd_device_setup
```
*   **Result:**
    ```text
    PCD Driver: Remove called for device pseudo-char-device.1
    PCD Driver: Remove called for device pseudo-char-device.0
    ```

---

## 🔬 Lab Exercise: Lab 135.1 - Private Data

### 1. Lab Objectives
- In `probe`, allocate a private structure (`struct pcd_private_data`) using `devm_kzalloc`.
- Store it in the device using `platform_set_drvdata(pdev, pdata)`.
- In `remove`, retrieve it using `platform_get_drvdata(pdev)`.

### 2. Step-by-Step Guide
1.  Define struct:
    ```c
    struct pcd_private {
        int size;
        char *buffer;
    };
    ```
2.  In `probe`:
    ```c
    struct pcd_private *pdata;
    pdata = devm_kzalloc(&pdev->dev, sizeof(*pdata), GFP_KERNEL);
    platform_set_drvdata(pdev, pdata);
    ```
3.  **Why `devm_`?** Managed resources. They are automatically freed when `probe` fails or `remove` is called. No memory leaks!

---

## 🧪 Additional / Advanced Labs

### Lab 2: Platform Data (`pdata`)
- **Goal:** Pass custom configuration (e.g., serial number) from Device to Driver.
- **Task:**
    1.  Define `struct pcd_platform_data { int serial; };`.
    2.  In `pcd_device_setup.c`:
        ```c
        static struct pcd_platform_data pdata_1 = { .serial = 12345 };
        pdev_1.dev.platform_data = &pdata_1;
        ```
    3.  In `probe`:
        ```c
        struct pcd_platform_data *pdata = dev_get_platdata(&pdev->dev);
        printk("Serial: %d", pdata->serial);
        ```

### Lab 3: Sysfs Attributes
- **Goal:** Expose the serial number in `/sys`.
- **Task:** Use `device_create_file` in `probe` to create `/sys/bus/platform/devices/pseudo-char-device.0/serial`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Probe not called
*   **Cause:** Name mismatch. `pdev.name` must be EXACTLY `pdrv.driver.name`.
*   **Cause:** Device not registered. Check `ls /sys/bus/platform/devices/`.

#### 2. "Resource busy"
*   **Cause:** You tried to `request_mem_region` for an address range already claimed by another driver.

---

## ⚡ Optimization & Best Practices

### Managed Resources (`devm_`)
*   ALWAYS use `devm_kzalloc`, `devm_request_irq`, `devm_ioremap`.
*   It simplifies error handling (no need for complex `goto` labels to free resources in reverse order).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we separate Device and Driver?
    *   **A:** Reusability. The same driver code can handle 10 identical UARTs. We just need 10 Device structures describing their addresses.
2.  **Q:** What happens if I load the Device module before the Driver module?
    *   **A:** The Device sits on the bus waiting. When the Driver loads, the bus core sees the match and calls `probe`. Order doesn't matter.

### Challenge Task
> **Task:** "The Multi-Instance Char Driver". Combine Day 128 and Day 135.
> *   In `probe`, create a `cdev`.
> *   Use the Memory Resource size to `kmalloc` the buffer.
> *   Result: `/dev/pcd0` accesses the 0x1000 buffer, `/dev/pcd1` accesses the 0x2000 buffer.

---

## 📚 Further Reading & References
- [Linux Device Drivers 3, Chapter 14: The Linux Device Model](https://lwn.net/Kernel/LDD3/)
- [Kernel Documentation: driver-model/platform.txt](https://www.kernel.org/doc/Documentation/driver-model/platform.txt)

---
