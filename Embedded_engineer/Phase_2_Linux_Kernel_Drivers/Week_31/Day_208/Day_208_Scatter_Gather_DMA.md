# Day 208: Scatter-Gather DMA
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
1.  **Explain** the fragmentation problem (why large buffers are rarely contiguous).
2.  **Use** `struct scatterlist` to describe a fragmented buffer.
3.  **Map** SG lists using `dma_map_sg`.
4.  **Iterate** over mapped SG entries (`for_each_sg`).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Day 207 (DMA).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Fragmentation Problem
*   **Scenario:** You want to send a 1MB buffer.
*   **Reality:** `kmalloc` might fail for 1MB. `vmalloc` returns 1MB virtually, but physically it is 256 separate 4KB pages scattered all over RAM.
*   **Solution:** Scatter-Gather.
    *   The driver creates a list: {Addr A, Len 4K}, {Addr B, Len 4K}, ...
    *   The Device (if SG-capable) reads the list and fetches data from A, then B, etc.

### 🔹 Part 2: The IOMMU Advantage
*   If an IOMMU is present, `dma_map_sg` can do magic.
*   It can take 10 scattered physical pages and map them to 1 contiguous Virtual Bus Address range.
*   Result: The device sees 1 contiguous block, even if RAM is fragmented.

---

## 💻 Implementation: Mapping an SG List

> **Instruction:** Create a scatterlist from a vmalloc buffer and map it.

### 👨‍💻 Code Implementation

```c
#include <linux/dma-mapping.h>
#include <linux/scatterlist.h>
#include <linux/vmalloc.h>

static void *my_buf;
static struct scatterlist *sgl;
static int n_ents;

static int map_sg_example(struct device *dev) {
    int i, count;
    struct scatterlist *sg;
    
    // 1. Allocate 1MB vmalloc buffer (physically fragmented)
    my_buf = vmalloc(1024 * 1024);
    
    // 2. Allocate SG Table
    // vmalloc_to_page needs to be called for each page
    // Helper: vmalloc_to_sg (not standard, usually we build sg from pages)
    // Let's assume we have a list of pages.
    
    // Simplified: Let's use a pre-existing SG table setup
    // (In real drivers, the block layer or net layer gives you the sg list)
    
    // ... Assume 'sgl' is populated ...
    
    // 3. Map SG
    // Returns number of contiguous segments device sees
    count = dma_map_sg(dev, sgl, n_ents, DMA_TO_DEVICE);
    
    if (count == 0) {
        pr_err("DMA Map SG failed\n");
        return -EIO;
    }
    
    pr_info("Mapped %d entries to %d DMA segments\n", n_ents, count);
    
    // 4. Iterate and give to device
    for_each_sg(sgl, sg, count, i) {
        dma_addr_t addr = sg_dma_address(sg);
        u32 len = sg_dma_len(sg);
        
        pr_info("Seg %d: Addr %llx, Len %d\n", i, addr, len);
        // write_to_device_descriptor(addr, len);
    }
    
    return 0;
}

static void unmap_sg_example(struct device *dev) {
    dma_unmap_sg(dev, sgl, n_ents, DMA_TO_DEVICE);
    vfree(my_buf);
}
```

---

## 🔬 Lab Exercise: Lab 208.1 - SG Construction

### 1. Lab Objectives
- Manually build an SG list from 2 pages.
- Map it.

### 2. Step-by-Step Guide
1.  **Alloc Pages:** `p1 = alloc_page(GFP_KERNEL); p2 = alloc_page(GFP_KERNEL);`
2.  **Init SG:**
    ```c
    struct scatterlist sg[2];
    sg_init_table(sg, 2);
    sg_set_page(&sg[0], p1, PAGE_SIZE, 0);
    sg_set_page(&sg[1], p2, PAGE_SIZE, 0);
    ```
3.  **Map:** `dma_map_sg(...)`.
4.  **Check:**
    *   If IOMMU is off, `count` should be 2.
    *   If IOMMU is on (and pages happen to be mapped contiguously), `count` *could* be 1.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Chained SG Lists
- **Goal:** Handle huge buffers.
- **Task:**
    *   A single array `struct scatterlist sg[N]` might be too big to alloc.
    *   Use `sg_chain` to link multiple arrays together.
    *   The DMA API handles the chaining automatically during mapping.

### Lab 3: Block Layer Integration
- **Goal:** See where SG comes from.
- **Task:**
    *   Look at a Block Driver `request_fn`.
    *   `blk_rq_map_sg(req_queue, req, sgl)`.
    *   The block layer constructs the SG list from the filesystem bio structs.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Using `sg->dma_address` directly
*   **Cause:** Accessing fields directly is bad practice.
*   **Fix:** Always use `sg_dma_address(sg)` and `sg_dma_len(sg)`. Macros handle architecture differences.

#### 2. Iterating wrong count
*   **Cause:** `for_each_sg` uses the *mapped* count, not the original count.
*   **Fix:** Use the return value of `dma_map_sg`.

---

## ⚡ Optimization & Best Practices

### SG Coalescing
*   Hardware often has a limit on the number of descriptors (e.g., 64).
*   If `dma_map_sg` returns > 64 segments, the request is too fragmented.
*   Driver must either fail or bounce-buffer (copy to a contiguous area).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `n_ents` (input) and `count` (output) in `dma_map_sg`?
    *   **A:** `n_ents` is the number of SG entries (physical fragments). `count` is the number of DMA descriptors needed. `count <= n_ents`. (IOMMU can merge them).
2.  **Q:** Does `dma_map_sg` copy data?
    *   **A:** No. It only sets up the address translation (IOMMU) and flushes caches.

### Challenge Task
> **Task:** "The Fragmenter".
> *   Alloc 4 pages.
> *   Fill them with "A", "B", "C", "D".
> *   Create an SG list: Page 0, Page 2, Page 1, Page 3. (Out of order).
> *   Simulate a DMA read.
> *   Verify the data comes out as "A", "C", "B", "D".

---

## 📚 Further Reading & References
- [Kernel Documentation: core-api/dma-api.rst](https://www.kernel.org/doc/html/latest/core-api/dma-api.html)

---
