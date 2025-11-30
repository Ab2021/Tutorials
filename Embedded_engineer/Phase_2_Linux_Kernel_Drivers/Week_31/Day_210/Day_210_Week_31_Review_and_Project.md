# Day 210: Week 31 Review and Project - The PCI-DMA Data Mover
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
1.  **Synthesize** Week 31 concepts (PCI, MSI, DMA, SG).
2.  **Architect** a high-performance driver.
3.  **Implement** a DMA-based data transfer mechanism.
4.  **Debug** DMA issues using `dma_debug`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC (QEMU with `edu` device recommended).
*   **Software Required:**
    *   All Week 31 tools.
*   **Prior Knowledge:**
    *   Week 31 Content.

---

## 🔄 Week 31 Review

### 1. PCI Basics (Day 204-205)
*   Config Space, BARs, Enumeration.
*   `pci_driver`, `probe`, `pci_iomap`.

### 2. Interrupts (Day 206)
*   MSI/MSI-X vs Legacy.
*   `pci_alloc_irq_vectors`.

### 3. DMA (Day 207-208)
*   Coherent vs Streaming.
*   `dma_map_single`, `dma_map_sg`.
*   Scatter-Gather lists.

### 4. DMA Engine (Day 209)
*   Offloading copies.

---

## 🛠️ Project: The "PCI-DMA Data Mover"

### 📋 Project Requirements
1.  **Driver:** `pci_mover.ko`.
2.  **Device:** QEMU Edu Device (1234:11e8).
    *   *Note:* The Edu device supports DMA! It has a DMA Source, Dest, and Count register.
3.  **Functionality:**
    *   Allocate a 4KB Coherent Buffer (Source).
    *   Allocate a 4KB Coherent Buffer (Dest).
    *   Fill Source with pattern (0xDEADBEEF).
    *   Program Edu device to copy Source -> Dest via DMA.
    *   Wait for MSI interrupt.
    *   Verify Dest contains 0xDEADBEEF.
4.  **Interface:**
    *   Trigger via Sysfs file `/sys/.../start_dma`.

---

## 💻 Implementation: Step-by-Step Guide

### 🔹 Phase 1: PCI & MSI Setup
Probe the device, enable it, map BAR0, request MSI interrupt.

```c
// QEMU Edu Registers
#define REG_DMA_SRC  0x80
#define REG_DMA_DST  0x88
#define REG_DMA_CNT  0x90
#define REG_DMA_CMD  0x98
#define CMD_START_DMA 0x1
#define REG_INTR_STATUS 0x24
#define REG_INTR_RAISE  0x60
```

### 🔹 Phase 2: DMA Allocation
```c
// In probe
src_cpu = dma_alloc_coherent(&pdev->dev, 4096, &src_dma, GFP_KERNEL);
dst_cpu = dma_alloc_coherent(&pdev->dev, 4096, &dst_dma, GFP_KERNEL);
```

### 🔹 Phase 3: The Trigger
```c
static ssize_t start_dma_store(...) {
    // 1. Fill Data
    memset(src_cpu, 0xAA, 4096);
    memset(dst_cpu, 0x00, 4096);
    
    // 2. Program Device (Write Physical/Bus Addresses)
    iowrite32(lower_32_bits(src_dma), mmio + REG_DMA_SRC);
    iowrite32(lower_32_bits(dst_dma), mmio + REG_DMA_DST);
    iowrite32(4096, mmio + REG_DMA_CNT);
    
    // 3. Start
    iowrite32(CMD_START_DMA | 0x2, mmio + REG_DMA_CMD); // 0x2 = IRQ enable
    
    return count;
}
```

### 🔹 Phase 4: The ISR
```c
static irqreturn_t edu_isr(int irq, void *dev) {
    u32 status = ioread32(mmio + REG_INTR_STATUS);
    
    if (status & 0x100) { // DMA Done bit (check spec)
        pr_info("DMA Finished!\n");
        // Verify data...
        if (memcmp(src_cpu, dst_cpu, 4096) == 0)
            pr_info("Data Match!\n");
        else
            pr_err("Data Mismatch!\n");
            
        // Acknowledge
        iowrite32(status, mmio + REG_INTR_STATUS);
        return IRQ_HANDLED;
    }
    return IRQ_NONE;
}
```

---

## 📈 Grading Rubric

| Criteria | Excellent (A) | Good (B) | Needs Improvement (C) |
| :--- | :--- | :--- | :--- |
| **DMA Success** | Data copied correctly. | Copy fails or corrupts. | System crash. |
| **Interrupts** | MSI triggers ISR. | Polling used instead. | No completion check. |
| **Cleanup** | No memory leaks on remove. | Leaks DMA buffers. | Leaks IRQ. |

---

## 🔮 Looking Ahead: Week 32
Next week, we dive into **Block Device Drivers**.
You will learn how to write a RAM Disk, handle Requests, Bios, and integrate with the Linux Block Layer.

---
