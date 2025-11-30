# Day 206: PCI Interrupts (MSI/MSI-X)
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
1.  **Distinguish** between Legacy INTx, MSI, and MSI-X.
2.  **Enable** MSI/MSI-X in a driver (`pci_alloc_irq_vectors`).
3.  **Request** the interrupt (`request_irq`).
4.  **Handle** the interrupt in the ISR.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC (QEMU).
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Day 205 (PCI Driver).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Legacy INTx
*   **Mechanism:** Physical wires (INTA, INTB, INTC, INTD) shared by multiple devices.
*   **Problem:** Shared IRQ means the kernel must call *every* ISR on that line to see who triggered it. Slow.
*   **Status:** Deprecated but still supported.

### 🔹 Part 2: MSI (Message Signaled Interrupts)
*   **Mechanism:** Device writes a specific data value to a specific address (LAPIC).
*   **Benefit:** No shared lines. Edge triggered. Faster.
*   **Limit:** Max 32 vectors per device.

### 🔹 Part 3: MSI-X
*   **Mechanism:** Extended MSI.
*   **Benefit:** Up to 2048 vectors. Can target specific CPUs (Affinity).
*   **Use Case:** High-performance NICs (1 queue per CPU core).

---

## 💻 Implementation: Enabling MSI

> **Instruction:** Modify our PCI driver to use MSI instead of Legacy IRQ.

### 👨‍💻 Code Implementation

```c
#include <linux/interrupt.h>

static irqreturn_t my_isr(int irq, void *dev_id) {
    struct pci_dev *pdev = dev_id;
    
    pr_info("MyPCI: Interrupt! IRQ %d\n", irq);
    
    // 1. Check Status Register (if shared or to clear it)
    // ioread32(mmio_base + STATUS_REG);
    
    // 2. Acknowledge/Clear Interrupt (Device specific)
    // iowrite32(ACK_BIT, mmio_base + CTRL_REG);
    
    return IRQ_HANDLED;
}

static int my_probe(struct pci_dev *pdev, const struct pci_device_id *id) {
    int ret;
    int nvec;
    
    // ... Enable and Map ...
    
    // 1. Allocate Vectors (Try MSI-X, then MSI, then fallback to Legacy)
    // min=1, max=1, flags=PCI_IRQ_ALL_TYPES
    nvec = pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_ALL_TYPES);
    if (nvec < 0) {
        pr_err("Failed to alloc IRQ vectors\n");
        return nvec;
    }
    
    // 2. Request IRQ
    // pdev->irq is updated automatically by pci_alloc_irq_vectors
    ret = request_irq(pdev->irq, my_isr, 0, "my_pci_driver", pdev);
    if (ret) {
        pci_free_irq_vectors(pdev);
        return ret;
    }
    
    pr_info("MyPCI: Using IRQ %d\n", pdev->irq);
    return 0;
}

static void my_remove(struct pci_dev *pdev) {
    free_irq(pdev->irq, pdev);
    pci_free_irq_vectors(pdev);
    // ... Unmap ...
}
```

---

## 🔬 Lab Exercise: Lab 206.1 - Triggering Interrupts

### 1. Lab Objectives
- Load driver.
- Trigger interrupt (if using QEMU Edu device).
- Verify in `/proc/interrupts`.

### 2. Step-by-Step Guide
1.  **Load:** `insmod my_pci.ko`.
2.  **Check IRQ Type:**
    *   `lspci -v` -> Look for `Capabilities: [xx] MSI: Enable+`.
    *   `cat /proc/interrupts | grep my_pci`.
    *   If MSI, it usually shows "PCI-MSI".
3.  **Trigger (Edu Device):**
    *   Write to the "Raise Interrupt" register (Offset 0x60).
    *   `devmem2` or modify driver to do it.
4.  **Observe:**
    *   `dmesg` shows "Interrupt!".
    *   Count in `/proc/interrupts` increments.

---

## 🧪 Additional / Advanced Labs

### Lab 2: MSI-X with Multiple Vectors
- **Goal:** Request 2 vectors.
- **Task:**
    1.  `pci_alloc_irq_vectors(pdev, 2, 2, PCI_IRQ_MSIX)`.
    2.  Use `pci_irq_vector(pdev, 0)` and `pci_irq_vector(pdev, 1)` to get the Linux IRQ numbers.
    3.  Call `request_irq` for each.
    4.  Assign different handlers or pass different data.

### Lab 3: IRQ Affinity
- **Goal:** Pin IRQ to CPU 1.
- **Task:**
    1.  `cat /proc/irq/<irq>/smp_affinity`.
    2.  `echo 2 > /proc/irq/<irq>/smp_affinity` (Bitmask 0x2 = CPU 1).
    3.  Generate load. Check `/proc/interrupts` to see CPU 1 handling it.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. `request_irq` fails with -EBUSY
*   **Cause:** IRQ conflict (Legacy) or device already grabbed.
*   **Fix:** Check `cat /proc/interrupts`.

#### 2. ISR never called
*   **Cause:** Device not generating interrupt (Masked in hardware), or Bus Master disabled.
*   **Fix:** Check `pci_set_master(pdev)`. Check device datasheet.

---

## ⚡ Optimization & Best Practices

### `pci_free_irq_vectors`
*   Always free vectors in `remove`.
*   Using `devm_request_irq` does *not* automatically free the vectors allocated by `pci_alloc_irq_vectors` (unless you use `pcim_` helpers which are newer/partial). Be careful with mixed managed/unmanaged resources.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is MSI better than INTx for virtualization?
    *   **A:** INTx requires the Hypervisor to emulate the IOAPIC wire toggling (expensive vmexit). MSI is just a memory write (posted interrupt), which can be delivered directly to the Guest without Hypervisor intervention (APICv).
2.  **Q:** What is `IRQF_SHARED`?
    *   **A:** Flag required for Legacy INTx. Tells kernel "I am willing to share this line". Not needed for MSI.

### Challenge Task
> **Task:** "The Multi-Queue Handler".
> *   Simulate a device with 4 queues.
> *   Allocate 4 MSI-X vectors.
> *   Create 4 threads (or workqueues).
> *   Map each ISR to wake up a specific thread.

---

## 📚 Further Reading & References
- [Kernel Documentation: PCI/msi-howto.rst](https://www.kernel.org/doc/html/latest/PCI/msi-howto.html)

---
