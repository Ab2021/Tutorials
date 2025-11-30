# Day 204: PCI Bus Architecture & Enumeration
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
1.  **Explain** the PCI Configuration Space (Vendor ID, Device ID, BARs).
2.  **Describe** the enumeration process (How the kernel finds devices).
3.  **Inspect** PCI devices using `lspci` and `setpci`.
4.  **Understand** Memory Mapped I/O (MMIO) vs Port I/O in PCI.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC (Virtual Machine is fine).
*   **Software Required:**
    *   `pciutils` (`lspci`).
*   **Prior Knowledge:**
    *   Basic Computer Architecture.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The PCI Bus
*   **Peripheral Component Interconnect (PCI):** A high-speed bus for attaching hardware devices.
*   **Topology:** Tree structure. Root Complex -> Bridges -> Endpoints.
*   **Addressing:** Domain : Bus : Device : Function (DBDF). Example: `0000:01:00.0`.

### 🔹 Part 2: Configuration Space
Every PCI device has a 256-byte (PCI) or 4KB (PCIe) configuration space.
*   **Header:**
    *   **Vendor ID (16-bit):** Manufacturer (e.g., Intel = 0x8086).
    *   **Device ID (16-bit):** Product.
    *   **Command/Status:** Control bits (Enable Bus Master, Memory Space).
    *   **BARs (Base Address Registers):** Request memory or I/O ports.

### 🔹 Part 3: Enumeration
1.  **Scan:** Kernel probes every Bus/Device/Function.
2.  **Read ID:** Reads Vendor/Device ID. If 0xFFFF, device doesn't exist.
3.  **Size BARs:** Writes 0xFFFFFFFF to BARs to see how much memory they need.
4.  **Allocate:** Assigns physical addresses to BARs.
5.  **Driver Binding:** Matches IDs against registered drivers.

---

## 💻 Implementation: Inspecting PCI Config Space

> **Instruction:** We will write a module that manually reads the config space of a specific device.

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <linux/pci.h>

static struct pci_dev *pdev = NULL;

static int __init my_init(void) {
    // Find a device (e.g., Ethernet Controller)
    // You can find IDs via lspci -n
    // Example: Intel 82540EM (QEMU default) is 8086:100e
    pdev = pci_get_device(0x8086, 0x100e, NULL);
    
    if (!pdev) {
        pr_err("PCI Device not found\n");
        return -ENODEV;
    }
    
    u16 vendor, device;
    u32 bar0;
    
    pci_read_config_word(pdev, PCI_VENDOR_ID, &vendor);
    pci_read_config_word(pdev, PCI_DEVICE_ID, &device);
    pci_read_config_dword(pdev, PCI_BASE_ADDRESS_0, &bar0);
    
    pr_info("Found PCI Device: %04x:%04x\n", vendor, device);
    pr_info("BAR0 Raw Value: %08x\n", bar0);
    
    // Check resource start/len (Kernel's view)
    pr_info("Resource 0 Start: %llx, Len: %llx\n", 
            pci_resource_start(pdev, 0),
            pci_resource_len(pdev, 0));
            
    return 0;
}

static void __exit my_exit(void) {
    if (pdev) pci_dev_put(pdev);
}

module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
```

---

## 🔬 Lab Exercise: Lab 204.1 - Using lspci

### 1. Lab Objectives
- Use `lspci` to decode the configuration space.

### 2. Step-by-Step Guide
1.  **List Devices:**
    ```bash
    lspci
    ```
2.  **Verbose Output:**
    ```bash
    lspci -v -s 00:03.0
    ```
    *   Shows Interrupts, Kernel Driver in use, Capabilities.
3.  **Hex Dump:**
    ```bash
    lspci -x -s 00:03.0
    ```
    *   Dumps the first 64 bytes (Standard Header).
4.  **Tree View:**
    ```bash
    lspci -t
    ```
    *   Shows the bus topology.

---

## 🧪 Additional / Advanced Labs

### Lab 2: setpci
- **Goal:** Modify registers from userspace (Dangerous!).
- **Task:**
    1.  Read Vendor ID: `setpci -s 00:03.0 0.w` (Offset 0, Word).
    2.  Blink an LED? (If you know the GPIO register offset in BAR).
    3.  **Warning:** Writing to BARs or Command registers can crash the system instantly.

### Lab 3: PCIe Capabilities
- **Goal:** Inspect PCIe specific features.
- **Task:**
    1.  `lspci -vv`
    2.  Look for `Capabilities: [xx] Express Endpoint`.
    3.  Check Link Speed (e.g., "LnkSta: Speed 8GT/s, Width x4").

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Device not found
*   **Cause:** Hardware issue, or device is hidden (virtualization).
*   **Fix:** Check `dmesg | grep pci`.

#### 2. BAR collision
*   **Cause:** BIOS failed to assign addresses, or overlapping ranges.
*   **Fix:** Kernel usually fixes this with `pci=realloc`.

---

## ⚡ Optimization & Best Practices

### `pci_get_device`
*   Increments the reference count of the `struct pci_dev`.
*   Always call `pci_dev_put` when done to avoid memory leaks.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is a BAR?
    *   **A:** Base Address Register. It tells the OS how much memory (or I/O space) the device needs. The OS writes the physical address back to it.
2.  **Q:** What is the difference between PCI and PCIe?
    *   **A:** PCI is a parallel shared bus (slow, blocking). PCIe is a serial point-to-point link (fast, packet-based). Software sees them almost identically (Config Space is backward compatible).

### Challenge Task
> **Task:** "The Scanner".
> *   Write a module that iterates over *all* PCI devices in the system (`for_each_pci_dev`).
> *   Print their Vendor/Device IDs and whether they are a "Bridge" or an "Endpoint".

---

## 📚 Further Reading & References
- [Kernel Documentation: PCI/pci.rst](https://www.kernel.org/doc/html/latest/PCI/pci.html)
- [PCI Local Bus Specification](https://pcisig.com/)

---
