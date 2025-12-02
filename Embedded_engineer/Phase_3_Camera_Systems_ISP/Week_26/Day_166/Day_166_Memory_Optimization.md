# Day 166: Memory Optimization (CMA & ION)
## Phase 3: Camera Systems & ISP | Week 26: Performance Optimization & Power

---

## 🎯 Learning Objectives
1.  **Understand** the Linux Memory Management for Multimedia: CMA (Contiguous Memory Allocator) and ION/DMA-BUF Heaps.
2.  **Analyze** `/proc/meminfo` and `/proc/slabinfo` to find leaks.
3.  **Configure** the CMA size in Device Tree to support large buffers (4K/8K).
4.  **Implement** a shared memory allocator using `dma_buf`.
5.  **Debug** OOM (Out of Memory) kills.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Jetson/Pi.
*   **Software:** Kernel Source, `dma-buf` utils.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Fragmentation Problem
*   **Virtual Memory:** Applications see contiguous memory, but physically it's scattered pages (4KB).
*   **DMA Requirement:** Most hardware (ISP, Encoder) requires **Physically Contiguous** memory.
*   **Problem:** After running for days, memory gets fragmented. You have 1GB free, but no 10MB contiguous block.
*   **Solution:** CMA (Contiguous Memory Allocator). A reserved region of RAM that is only used for movable pages, so it can be cleared when a big block is needed.

### 🔹 Part 2: ION / DMA-BUF Heaps
*   **Android ION:** The old standard for sharing buffers between CPU/GPU/ISP.
*   **DMA-BUF Heaps:** The new mainline Linux standard.
*   **Heaps:**
    *   `system`: Non-contiguous (Scatter-Gather).
    *   `cma`: Contiguous.

### 🔹 Part 3: Swap
*   **ZRAM:** Compressed RAM swap. Fast.
*   **Disk Swap:** Slow. Avoid for real-time camera systems (causes latency spikes).

---

## 💻 Implementation Examples

### Example 1: Configuring CMA (Device Tree)

Reserving 512MB for camera buffers.

```dts
/ {
    reserved-memory {
        #address-cells = <2>;
        #size-cells = <2>;
        ranges;

        linux,cma {
            compatible = "shared-dma-pool";
            reusable;
            size = <0x0 0x20000000>; // 512MB
            alignment = <0x0 0x2000>;
            linux,cma-default;
        };
    };
};
```

### Example 2: Allocating from DMA-BUF Heap (C++)

```cpp
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <fcntl.h>

int dmabuf_fd;

int allocate_buffer(size_t size) {
    int heap_fd = open("/dev/dma_heap/linux,cma", O_RDWR);
    if (heap_fd < 0) return -1;

    struct dma_heap_allocation_data data = {
        .len = size,
        .fd_flags = O_RDWR | O_CLOEXEC,
        .heap_flags = 0,
    };

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data) < 0) {
        perror("Alloc failed");
        return -1;
    }

    dmabuf_fd = data.fd;
    close(heap_fd);
    return dmabuf_fd;
}
```

### Example 3: Monitoring CMA Usage

```bash
# Check Total/Free
cat /proc/meminfo | grep CMA
# CmaTotal:       524288 kB
# CmaFree:        450000 kB

# Check who is using it (if debugfs enabled)
cat /sys/kernel/debug/cma/cma-reserved/bitmap
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Stress Test CMA

**Objective:** Fragment the memory.

**Steps:**
1.  Write a script that allocates random sized buffers (1MB to 50MB) and frees them in random order.
2.  Monitor `CmaFree`.
3.  Try to allocate a huge block (e.g., 100MB).
4.  **Observation:** It might take time (latency spike) as the kernel migrates pages to make room.

### Lab 2: OOM Killer Analysis

**Objective:** What happens when RAM is full?

**Steps:**
1.  Run `tail -f /var/log/syslog`.
2.  Run a program that `malloc`s 1GB in a loop.
3.  **Result:** "Out of memory: Killed process X".
4.  **Fix:** Adjust `oom_score_adj` for your critical Camera Process so it's the *last* to be killed.

### Lab 3: ZRAM Configuration

**Objective:** Squeeze more data.

**Steps:**
1.  Install `zram-tools`.
2.  Configure 50% of RAM as ZRAM.
3.  **Benefit:** You can fit larger AI models by compressing inactive pages.

---

## 🐛 Debugging Memory Issues

### Debug 1: "cma: allocation failed"

**Symptom:** `dmesg` shows CMA allocation failure.

**Cause:**
*   CMA region is full.
*   Fragmentation is too severe (pinned pages in CMA).
*   **Fix:** Increase CMA size in Device Tree. Or reduce buffer count in GStreamer (`num-buffers`).

### Debug 2: Memory Leak

**Symptom:** `MemAvailable` drops slowly over 24 hours.

**Cause:**
*   Application not freeing `dmabuf_fd`.
*   Driver not freeing `sk_buff` (Network).
*   **Fix:** Use Valgrind (for userspace) or `kmemleak` (for kernel).

---

## ⚡ Performance Optimization

### Optimization 1: Huge Pages

*   Use 2MB or 1GB pages instead of 4KB.
*   Reduces TLB (Translation Lookaside Buffer) misses.
*   Improves performance for large buffer access (4K video).

### Optimization 2: Pre-Allocation

*   Allocate all buffers at startup.
*   Never allocate/free during the streaming loop.
*   Deterministic performance.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "Pinned Memory"?** (Memory that cannot be swapped out or moved. Required for DMA).
2.  **Difference between `kmalloc` and `vmalloc`?** (`kmalloc` gives physically contiguous memory. `vmalloc` gives virtually contiguous but physically scattered memory).
3.  **Why is CMA better than `mem=...` kernel argument?** (CMA memory can be used by the OS for other things when not needed. `mem=` reserves it permanently, wasting RAM).

### Practical Challenges

1.  **Implement a Buffer Pool:** Write a C++ class that manages a pool of 10 DMA-BUFs. `acquire()` and `release()` methods.
2.  **Analyze Slab Info:** Run `slabtop`. Identify which kernel object consumes the most memory (e.g., `dentry`, `inode_cache`).

---

## 📚 Further Reading & Resources

### Documentation
*   **Linux DMA-BUF Documentation.**
*   **"Understanding the Linux Virtual Memory Manager" (Book).**

---

## 🎓 Summary

Today we covered:
- ✅ **Fragmentation:** The enemy of DMA.
- ✅ **CMA:** The solution.
- ✅ **DMA-BUF Heaps:** The API.
- ✅ **OOM:** When the limit is reached.
- ✅ **ZRAM:** Compression.

**Next:** Day 167 - Week 26 Review & Project (Optimized Pipeline).

---

**Day 166 Complete** | Phase 3: Camera Systems & ISP | Week 26: Performance Optimization & Power


