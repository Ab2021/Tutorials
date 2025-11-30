# Day 213: Handling Bios (Make Request)
## Phase 2: Linux Kernel & Device Drivers | Week 32: Block Device Drivers

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
1.  **Distinguish** between Request-based and Bio-based drivers.
2.  **Implement** a `submit_bio` callback.
3.  **Iterate** over `bio_vec`s using `bio_for_each_segment`.
4.  **Complete** a bio using `bio_endio`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Day 212 (RAM Disk).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Request vs Bio
*   **Request-based:** Standard. Kernel merges bios into requests. Good for HDDs (seek optimization) and standard SSDs.
*   **Bio-based:** Driver handles `bio` directly. No merging queue. Good for NVMe (hardware handles queues), RAM disks (no seek time), and stacking drivers (DM/MD).

### 🔹 Part 2: The `struct bio`
*   `bi_iter`: Iterator (sector, size).
*   `bi_io_vec`: Array of pages (Scatter-Gather).
*   `bi_private`: Driver private data.
*   `bi_end_io`: Completion callback.

---

## 💻 Implementation: Bio-based RAM Disk

> **Instruction:** Rewrite the RAM disk to use `submit_bio` instead of `blk-mq`.

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <linux/blkdev.h>
#include <linux/vmalloc.h>

#define DISK_SIZE (16 * 1024 * 1024)
#define SECTOR_SIZE 512

static struct gendisk *my_disk;
static struct request_queue *my_queue;
static u8 *dev_data;

static void my_bio_transfer(struct bio *bio) {
    struct bio_vec bvec;
    struct bvec_iter iter;
    loff_t pos = bio->bi_iter.bi_sector * SECTOR_SIZE;
    void *buffer;

    bio_for_each_segment(bvec, bio, iter) {
        size_t len = bvec.bv_len;
        void *kaddr = kmap_atomic(bvec.bv_page) + bvec.bv_offset;
        
        if (pos + len > DISK_SIZE) {
            // Error handling
            kunmap_atomic(kaddr);
            bio_io_error(bio);
            return;
        }

        buffer = dev_data + pos;

        if (bio_data_dir(bio) == WRITE)
            memcpy(buffer, kaddr, len);
        else
            memcpy(kaddr, buffer, len);

        kunmap_atomic(kaddr);
        pos += len;
    }
    
    // Complete the bio
    bio_endio(bio);
}

static void my_submit_bio(struct bio *bio) {
    // In bio-based drivers, we process immediately (or offload to workqueue)
    my_bio_transfer(bio);
}

static const struct block_device_operations my_fops = {
    .owner = THIS_MODULE,
    .submit_bio = my_submit_bio, // New API for bio-based
};

static int __init my_init(void) {
    dev_data = vmalloc(DISK_SIZE);
    
    // 1. Alloc Queue (No tags needed for bio-based)
    my_queue = blk_alloc_queue(NUMA_NO_NODE);
    
    // 2. Alloc Disk
    my_disk = alloc_disk(1);
    my_disk->queue = my_queue;
    my_disk->major = register_blkdev(0, "mybio");
    my_disk->first_minor = 0;
    my_disk->fops = &my_fops;
    sprintf(my_disk->disk_name, "mybio0");
    set_capacity(my_disk, DISK_SIZE / SECTOR_SIZE);
    
    // Important: Tell kernel this is a bio-based queue
    // (Actually, setting .submit_bio in fops is the key in newer kernels)
    // blk_queue_make_request(my_queue, my_make_request); // Legacy
    
    add_disk(my_disk);
    return 0;
}
// ... exit ...
```

---

## 🔬 Lab Exercise: Lab 213.1 - Performance Comparison

### 1. Lab Objectives
- Compare `myram` (blk-mq) vs `mybio` (bio-based).
- Use `fio` to benchmark.

### 2. Step-by-Step Guide
1.  **Install fio:** `apt install fio`.
2.  **Run Test:**
    ```bash
    fio --name=test --filename=/dev/mybio0 --rw=randwrite --bs=4k --ioengine=libaio --iodepth=32 --size=10M
    ```
3.  **Compare:**
    *   Bio-based should have slightly lower latency (no queue overhead).
    *   Blk-mq might handle high concurrency better on multi-core systems (due to per-cpu queues).

---

## 🧪 Additional / Advanced Labs

### Lab 2: Splitting Bios
- **Goal:** Handle hardware limits.
- **Task:**
    *   If hardware can only transfer 4KB at a time, but bio is 64KB.
    *   Use `blk_queue_max_hw_sectors`.
    *   Or manually split using `bio_split`.

### Lab 3: Stacking Driver (The Filter)
- **Goal:** Modify data in flight.
- **Task:**
    *   Write a driver that takes a bio, allocates a *new* bio, copies data (encrypts it), submits the new bio to a *real* disk (`submit_bio_noacct`), and waits.
    *   This is how `dm-crypt` works.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Recursion
*   **Cause:** Calling `submit_bio` on your own device from within `submit_bio`.
*   **Fix:** Don't do that. Submit to a *different* device (stacking) or handle it.

#### 2. Sleeping in `submit_bio`
*   **Cause:** `submit_bio` can be called from atomic context.
*   **Fix:** If you need to sleep (e.g., wait for hardware), offload to a Workqueue.

---

## ⚡ Optimization & Best Practices

### `bio_endio`
*   Must be called exactly once per bio.
*   If you split a bio, you must chain completions.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** When should I use Bio-based vs Request-based?
    *   **A:** Use Request-based for real hardware that benefits from merging/scheduling (HDD, SSD). Use Bio-based for virtual devices (RAM disk) or stacking drivers (LVM, RAID).
2.  **Q:** What is `bio_chain`?
    *   **A:** A helper to link a child bio to a parent bio, so the parent completes only when the child completes.

### Challenge Task
> **Task:** "The Faulty Disk".
> *   Modify `my_submit_bio`.
> *   Randomly fail 1% of requests (`bio_io_error`).
> *   Mount ext4 and see how it handles errors (it usually remounts RO).

---

## 📚 Further Reading & References
- [Kernel Documentation: block/biovecs.rst](https://www.kernel.org/doc/html/latest/block/biovecs.html)

---
