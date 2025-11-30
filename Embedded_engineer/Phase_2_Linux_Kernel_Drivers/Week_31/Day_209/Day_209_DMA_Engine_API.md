# Day 209: The DMA Engine API
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
1.  **Understand** the role of the DMA Engine subsystem (Hardware offload).
2.  **Request** a DMA channel (`dma_request_chan`).
3.  **Prepare** and **Submit** transactions (`dmaengine_prep_slave_sg`, `dma_async_issue_pending`).
4.  **Handle** completion callbacks.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   SoC Board (Pi, BeagleBone) is best (they have DMA controllers).
    *   PC can use `dmatest` module.
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Day 208 (SG DMA).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Slave DMA vs Async TX
*   **Async TX (Memcpy/XOR):** Offloading CPU tasks (e.g., RAID calculation, large memory copy) to a DMA engine.
*   **Slave DMA:** Peripheral DMA. Moving data from a device (UART, SPI, I2S) to Memory. The "Slave" is the peripheral.

### 🔹 Part 2: The Workflow
1.  **Request Channel:** Ask for a channel capable of doing what you need.
2.  **Configure:** Set bus width, burst size, direction.
3.  **Map Data:** Use `dma_map_sg`.
4.  **Prepare:** Get a descriptor (`struct dma_async_tx_descriptor`).
5.  **Submit:** Push to queue (`tx_submit`).
6.  **Issue:** Start the hardware (`dma_async_issue_pending`).

---

## 💻 Implementation: Using DMA Engine for Memcpy

> **Instruction:** Write a module that offloads a 1MB copy to the hardware DMA engine.

### 👨‍💻 Code Implementation

```c
#include <linux/dmaengine.h>
#include <linux/dma-mapping.h>
#include <linux/completion.h>

static struct dma_chan *chan;
static struct completion dma_done;

static void my_dma_callback(void *param) {
    complete(&dma_done);
}

static int test_dma_memcpy(void) {
    dma_cap_mask_t mask;
    struct dma_async_tx_descriptor *tx;
    dma_cookie_t cookie;
    char *src, *dst;
    dma_addr_t dma_src, dma_dst;
    
    // 1. Request Channel (Any DMA_MEMCPY capable channel)
    dma_cap_zero(mask);
    dma_cap_set(DMA_MEMCPY, mask);
    chan = dma_request_channel(mask, NULL, NULL);
    if (!chan) return -ENODEV;
    
    // 2. Alloc Buffers
    src = kzalloc(4096, GFP_KERNEL);
    dst = kzalloc(4096, GFP_KERNEL);
    strcpy(src, "DMA Rocks!");
    
    // 3. Map
    dma_src = dma_map_single(chan->device->dev, src, 4096, DMA_TO_DEVICE);
    dma_dst = dma_map_single(chan->device->dev, dst, 4096, DMA_FROM_DEVICE);
    
    // 4. Prepare
    tx = dmaengine_prep_dma_memcpy(chan, dma_dst, dma_src, 4096, DMA_CTRL_ACK | DMA_PREP_INTERRUPT);
    if (!tx) goto err_map;
    
    tx->callback = my_dma_callback;
    init_completion(&dma_done);
    
    // 5. Submit & Issue
    cookie = tx->tx_submit(tx);
    dma_async_issue_pending(chan);
    
    // 6. Wait
    wait_for_completion(&dma_done);
    
    pr_info("DMA Copy Done: %s\n", dst);
    
    // Cleanup...
    dma_unmap_single(chan->device->dev, dma_src, 4096, DMA_TO_DEVICE);
    dma_unmap_single(chan->device->dev, dma_dst, 4096, DMA_FROM_DEVICE);
    dma_release_channel(chan);
    return 0;

err_map:
    // Unmap...
    return -EIO;
}
```

---

## 🔬 Lab Exercise: Lab 209.1 - dmatest

### 1. Lab Objectives
- Use the built-in `dmatest` module to stress test the DMA controller.

### 2. Step-by-Step Guide
1.  **Load:**
    ```bash
    modprobe dmatest
    ```
2.  **Configure:**
    ```bash
    echo 2000 > /sys/module/dmatest/parameters/timeout
    echo 1 > /sys/module/dmatest/parameters/iterations
    echo 1 > /sys/module/dmatest/parameters/run
    ```
3.  **Check Log:**
    ```
    dmesg | grep dmatest
    # "dmatest: dma0chan0-copy0: summary 1 tests, 0 failures 1000 iops..."
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Slave DMA (SPI)
- **Goal:** Understand how SPI drivers use DMA.
- **Task:**
    *   Read `drivers/spi/spi-bcm2835.c` (Raspberry Pi SPI).
    *   Look for `dma_request_slave_channel`.
    *   Look for `dmaengine_prep_slave_sg`.
    *   The SPI controller has a DREQ (DMA Request) line connected to the DMA controller.

### Lab 3: Cyclic DMA (Audio)
- **Goal:** Audio Ring Buffer.
- **Task:**
    *   `dmaengine_prep_dma_cyclic`.
    *   The DMA engine loops over the buffer indefinitely, firing interrupts at period boundaries.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "No DMA channel available"
*   **Cause:** All channels used, or Device Tree configuration missing.
*   **Fix:** Check `/sys/class/dma/`. Check DT `dmas` property.

#### 2. Data Corruption
*   **Cause:** Cache coherency (forgot `dma_map`), or buffer freed before DMA completed.
*   **Fix:** Ensure buffer stays alive until callback fires.

---

## ⚡ Optimization & Best Practices

### Chain Submission
*   You can submit multiple descriptors before calling `issue_pending`.
*   This reduces MMIO overhead (starting the engine once for a batch).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is a "Cookie" in DMA Engine?
    *   **A:** A transaction ID. Used to check the status of a specific transfer (`dma_async_is_tx_complete`).
2.  **Q:** Why use DMA for `memcpy`?
    *   **A:** For small copies, CPU is faster (setup overhead of DMA is high). For large copies (MBs), DMA wins because it frees the CPU to do other things (Async).

### Challenge Task
> **Task:** "The Scatter-Gather Copier".
> *   Modify the memcpy example.
> *   Use `dmaengine_prep_slave_sg` (if supported) or just chain multiple `prep_dma_memcpy`.
> *   Copy a scattered source (vmalloc) to a contiguous destination.

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/dmaengine/provider.rst](https://www.kernel.org/doc/html/latest/driver-api/dmaengine/provider.html)
- [Kernel Documentation: driver-api/dmaengine/client.rst](https://www.kernel.org/doc/html/latest/driver-api/dmaengine/client.html)

---
