# Day 207: DMA Mapping (Streaming vs Coherent)
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
1.  **Explain** DMA (Direct Memory Access) and why CPU copies are slow.
2.  **Distinguish** between Coherent (Consistent) and Streaming DMA.
3.  **Use** `dma_alloc_coherent` for control structures (Rings).
4.  **Use** `dma_map_single` for data buffers (Packets).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Virtual vs Physical Memory.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The DMA Concept
*   **CPU Copy:** CPU reads from RAM, writes to Device FIFO. Slow, high CPU usage.
*   **DMA:** CPU tells Device "Read 1KB from Address X". Device reads RAM directly. CPU is free.

### 🔹 Part 2: Virtual vs Bus Addresses
*   **Virtual Address (VA):** What the CPU sees (`void *`).
*   **Physical Address (PA):** What the RAM controller sees.
*   **Bus Address (BA):** What the Device sees. (Usually PA == BA, but not always, e.g., IOMMU).
*   **Rule:** NEVER pass a `void *` or `virt_to_phys()` to a device. ALWAYS use `dma_map_*`.

### 🔹 Part 3: Coherency
*   **Problem:** CPU has cache. RAM has data.
    *   If CPU writes to cache but not RAM, Device reads old data (Stale).
    *   If Device writes to RAM but CPU reads cache, CPU reads old data.
*   **Solution 1: Coherent DMA:** Disables caching (or uses hardware snooping). Good for long-lived structures (Descriptors).
*   **Solution 2: Streaming DMA:** Explicitly flushes/invalidates cache (`dma_sync`). Good for one-off buffers (Network packets).

---

## 💻 Implementation: Coherent DMA (The Ring Buffer)

> **Instruction:** Allocate a block of memory that both CPU and Device can access safely.

### 👨‍💻 Code Implementation

```c
#include <linux/dma-mapping.h>
#include <linux/pci.h>

struct my_descriptor {
    u32 addr;
    u32 len;
    u32 status;
};

static void *cpu_addr;
static dma_addr_t dma_handle;

static int my_probe(struct pci_dev *pdev, const struct pci_device_id *id) {
    // 1. Set Mask (Tell kernel we support 64-bit addressing)
    if (dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64))) {
        dev_warn(&pdev->dev, "No 64-bit DMA, trying 32-bit\n");
        dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
    }

    // 2. Allocate Coherent Memory (4KB)
    // cpu_addr: Virtual address for driver to read/write
    // dma_handle: Bus address to give to the device
    cpu_addr = dma_alloc_coherent(&pdev->dev, 4096, &dma_handle, GFP_KERNEL);
    
    if (!cpu_addr) return -ENOMEM;
    
    pr_info("DMA: CPU Addr: %p, Bus Addr: %llx\n", cpu_addr, dma_handle);
    
    // 3. Write Bus Address to Device Register
    // iowrite32(lower_32_bits(dma_handle), mmio + REG_RING_LO);
    // iowrite32(upper_32_bits(dma_handle), mmio + REG_RING_HI);
    
    return 0;
}

static void my_remove(struct pci_dev *pdev) {
    if (cpu_addr) {
        dma_free_coherent(&pdev->dev, 4096, cpu_addr, dma_handle);
    }
}
```

---

## 💻 Implementation: Streaming DMA (The Packet)

> **Instruction:** Map a buffer for a single transfer.

### 👨‍💻 Code Implementation

```c
static void send_packet(struct pci_dev *pdev, char *buffer, int len) {
    dma_addr_t dma_addr;
    
    // 1. Map (Flush cache)
    dma_addr = dma_map_single(&pdev->dev, buffer, len, DMA_TO_DEVICE);
    
    if (dma_mapping_error(&pdev->dev, dma_addr)) {
        pr_err("DMA Mapping failed\n");
        return;
    }
    
    // 2. Give dma_addr to device
    // iowrite32(dma_addr, ...);
    // kick_device(...);
    
    // 3. Wait for completion (Interrupt or Polling)
    
    // 4. Unmap (Release resources)
    dma_unmap_single(&pdev->dev, dma_addr, len, DMA_TO_DEVICE);
}
```

---

## 🔬 Lab Exercise: Lab 207.1 - DMA Allocation

### 1. Lab Objectives
- Load the driver.
- Verify allocation success.
- Check `/proc/iomem` or `dmesg`.

### 2. Step-by-Step Guide
1.  **Load:** `insmod my_dma.ko`.
2.  **Check Log:**
    *   "DMA: CPU Addr: ffff..., Bus Addr: 12345000".
3.  **Verify:**
    *   The Bus Address should look like a physical address (e.g., within RAM range).
    *   If IOMMU is on, it might look like a virtual IO address.

---

## 🧪 Additional / Advanced Labs

### Lab 2: DMA Direction
- **Goal:** Understand `DMA_TO_DEVICE` vs `DMA_FROM_DEVICE`.
- **Task:**
    *   `DMA_TO_DEVICE`: CPU writes, Device reads. (Cache Flush).
    *   `DMA_FROM_DEVICE`: Device writes, CPU reads. (Cache Invalidate).
    *   `DMA_BIDIRECTIONAL`: Both. (Expensive sync).

### Lab 3: DMA Sync
- **Goal:** Reuse a mapped buffer.
- **Task:**
    1.  Map buffer.
    2.  Device writes data.
    3.  `dma_sync_single_for_cpu(...)`.
    4.  CPU reads data.
    5.  `dma_sync_single_for_device(...)`.
    6.  Device writes new data.
    7.  Unmap eventually.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "DMA-API: device driver not checking mapping error"
*   **Cause:** Kernel debug option `CONFIG_DMA_API_DEBUG` is on, and you didn't call `dma_mapping_error`.
*   **Fix:** Always check for errors!

#### 2. Corrupted Data
*   **Cause:** Cache coherency issue. Using `virt_to_phys` instead of `dma_map`.
*   **Fix:** Use the API.

---

## ⚡ Optimization & Best Practices

### IOMMU Overhead
*   `dma_map` can be slow if IOMMU is enabled (it has to update page tables).
*   For high performance, use **Scatter-Gather** (Day 208) to map many pages at once, or reuse mappings.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why can't I just use `kmalloc` and pass the address to the device?
    *   **A:** 1. `kmalloc` returns a Virtual Address. 2. The memory might not be physically contiguous. 3. Cache issues.
2.  **Q:** What is `GFP_DMA`?
    *   **A:** A flag to allocate memory in the first 16MB (ISA DMA limit). Rarely needed for PCI devices (which can usually address 32-bit or 64-bit).

### Challenge Task
> **Task:** "The Loopback".
> *   Allocate a Coherent buffer.
> *   Write a pattern (0xAA) to it from CPU.
> *   (Simulate) Device reads it.
> *   (Simulate) Device writes 0x55.
> *   CPU reads 0x55.
> *   Prove that no explicit cache flush was needed (because it's Coherent).

---

## 📚 Further Reading & References
- [Kernel Documentation: core-api/dma-api.rst](https://www.kernel.org/doc/html/latest/core-api/dma-api.html)
- [DMA-API-HOWTO](https://www.kernel.org/doc/Documentation/DMA-API-HOWTO.txt)

---
