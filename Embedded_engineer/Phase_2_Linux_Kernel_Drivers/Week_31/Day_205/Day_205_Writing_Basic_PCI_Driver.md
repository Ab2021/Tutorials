# Day 205: Writing a Basic PCI Driver
## Phase 2: Linux Kernel & Device Drivers | Week 31: PCI & DMA

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
1.  **Register** a PCI driver using `struct pci_driver`.
2.  **Define** the ID Table (`MODULE_DEVICE_TABLE`) to bind to specific hardware.
3.  **Enable** the device (`pci_enable_device`) and Request regions.
4.  **Map** BARs to kernel virtual memory (`pci_iomap`).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC (QEMU with a virtual PCI device like `e1000` or `edu` is ideal).
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Day 204 (PCI Arch).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The PCI Driver Structure
*   **`id_table`:** Array of `struct pci_device_id`. Tells the kernel which devices this driver supports.
*   **`probe`:** Called when a matching device is found.
*   **`remove`:** Called when device is unplugged or driver unloaded.

### 🔹 Part 2: The Probe Sequence
1.  **Enable:** `pci_enable_device(pdev)`. (Wakes it up, enables IO/Mem bits).
2.  **Request:** `pci_request_regions(pdev, "name")`. (Reserves the BAR ranges).
3.  **Map:** `pci_iomap(pdev, bar_index, len)`. (Creates virtual mapping for MMIO).
4.  **Setup:** Initialize hardware, register subsystem (Net, Char, etc.).

---

## 💻 Implementation: A "Hello World" PCI Driver

> **Instruction:** Write a driver that binds to a QEMU "edu" device (1234:11e8) or Intel E1000 (8086:100e). We will use E1000 as an example but only print info.

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <linux/pci.h>

#define VENDOR_ID 0x8086
#define DEVICE_ID 0x100e // E1000 in QEMU

static struct pci_device_id my_pci_ids[] = {
    { PCI_DEVICE(VENDOR_ID, DEVICE_ID) },
    { 0, } // Null terminator
};
MODULE_DEVICE_TABLE(pci, my_pci_ids);

static void __iomem *mmio_base;

static int my_probe(struct pci_dev *pdev, const struct pci_device_id *id) {
    int ret;
    
    pr_info("MyPCI: Probing device %04x:%04x\n", pdev->vendor, pdev->device);
    
    // 1. Enable Device
    ret = pci_enable_device(pdev);
    if (ret) return ret;
    
    // 2. Request Regions (Exclusive ownership)
    ret = pci_request_regions(pdev, "my_pci_driver");
    if (ret) {
        pci_disable_device(pdev);
        return ret;
    }
    
    // 3. Map BAR 0 (Usually main registers)
    mmio_base = pci_iomap(pdev, 0, 0); // 0 length = map whole BAR
    if (!mmio_base) {
        pci_release_regions(pdev);
        pci_disable_device(pdev);
        return -ENOMEM;
    }
    
    pr_info("MyPCI: Mapped BAR0 to %p\n", mmio_base);
    
    // Read first register (Device Control usually)
    pr_info("MyPCI: Reg 0 = %08x\n", ioread32(mmio_base));
    
    return 0;
}

static void my_remove(struct pci_dev *pdev) {
    if (mmio_base) {
        pci_iounmap(pdev, mmio_base);
    }
    pci_release_regions(pdev);
    pci_disable_device(pdev);
    pr_info("MyPCI: Removed\n");
}

static struct pci_driver my_driver = {
    .name = "my_pci_driver",
    .id_table = my_pci_ids,
    .probe = my_probe,
    .remove = my_remove,
};

module_pci_driver(my_driver); // Helper for init/exit
MODULE_LICENSE("GPL");
```

---

## 🔬 Lab Exercise: Lab 205.1 - Binding and Unbinding

### 1. Lab Objectives
- Load the driver.
- Verify it binds to the device.
- Unbind it manually via Sysfs.

### 2. Step-by-Step Guide
1.  **Unbind Existing Driver:**
    *   If `e1000` is loaded, unbind it first.
    *   `echo "0000:00:03.0" > /sys/bus/pci/drivers/e1000/unbind`
2.  **Load Your Driver:**
    *   `insmod my_pci.ko`.
    *   `dmesg` should show "Probing device...".
3.  **Check Sysfs:**
    *   `ls -l /sys/bus/pci/drivers/my_pci_driver/`.
    *   Should see a symlink to `0000:00:03.0`.
4.  **Unbind:**
    *   `echo "0000:00:03.0" > /sys/bus/pci/drivers/my_pci_driver/unbind`.
    *   `dmesg` should show "Removed".

---

## 🧪 Additional / Advanced Labs

### Lab 2: The QEMU "edu" Device
- **Goal:** Interact with a device designed for education.
- **Task:**
    1.  Start QEMU with `-device edu`. (ID 1234:11e8).
    2.  Update driver IDs.
    3.  Edu device has a factorial calculator.
    4.  Write integer to offset 0x08 (Status/Factorial).
    5.  Read result.

### Lab 3: Multiple BARs
- **Goal:** Map multiple regions.
- **Task:**
    1.  Check `pci_resource_flags(pdev, bar)`.
    2.  If `IORESOURCE_MEM`, map it.
    3.  If `IORESOURCE_IO`, use `ioport_map` (or just `pci_iomap` handles it too).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Probe not called
*   **Cause:** Another driver (standard kernel driver) is already bound.
*   **Fix:** Blacklist the standard driver or unbind it manually.

#### 2. System Freeze on Read
*   **Cause:** Reading from an unmapped address or disabled device.
*   **Fix:** Ensure `pci_enable_device` succeeded.

---

## ⚡ Optimization & Best Practices

### `devm_pci_iomap`
*   Managed resources!
*   `pcim_enable_device(pdev)`.
*   `pcim_iomap_regions(pdev, BIT(0), "name")`.
*   Cleanup is automatic on remove. Code becomes 50% shorter.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we need `pci_request_regions`?
    *   **A:** To mark the memory range as "busy" in `/proc/iomem`. Prevents two drivers from accidentally mapping the same hardware.
2.  **Q:** What is `module_pci_driver`?
    *   **A:** A macro that generates the `module_init` and `module_exit` functions, calling `pci_register_driver` and `pci_unregister_driver`.

### Challenge Task
> **Task:** "The ID Scanner".
> *   Modify the driver to support *any* device from Vendor 0x8086 (`PCI_ANY_ID`).
> *   In probe, print "I found an Intel device!".
> *   (Note: This will try to grab your WiFi, GPU, etc. Be careful!).

---

## 📚 Further Reading & References
- [Kernel Documentation: PCI/pci.rst](https://www.kernel.org/doc/html/latest/PCI/pci.html)
- [QEMU Edu Device Spec](https://github.com/qemu/qemu/blob/master/docs/specs/edu.txt)

---
