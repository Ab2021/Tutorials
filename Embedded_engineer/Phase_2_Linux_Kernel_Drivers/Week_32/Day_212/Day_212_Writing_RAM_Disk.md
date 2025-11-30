# Day 212: Writing a Simple RAM Disk (sbull)
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
1.  **Implement** a Block Device Driver using `blk-mq`.
2.  **Handle** I/O requests by copying data to/from a kernel buffer.
3.  **Use** `blk_mq_init_sq_queue` for simple drivers.
4.  **Format** and **Mount** the custom RAM disk.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source (5.x or newer recommended for `blk-mq`).
*   **Prior Knowledge:**
    *   Day 211 (Block Arch).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The RAM Disk Concept
*   **Storage:** A chunk of allocated kernel memory (`vmalloc`).
*   **Read:** Copy from `vmalloc` buffer to `bio` pages.
*   **Write:** Copy from `bio` pages to `vmalloc` buffer.
*   **Volatility:** Data is lost on unload/reboot.

### 🔹 Part 2: Blk-MQ (Multi-Queue) Basics
*   **Tag Set (`struct blk_mq_tag_set`):** Manages tags (identifiers) for requests.
*   **Queue (`struct request_queue`):** The interface.
*   **Ops (`struct blk_mq_ops`):** Functions to handle requests (`.queue_rq`).

---

## 💻 Implementation: The "MyRAM" Driver

> **Instruction:** Create a 16MB RAM disk.

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <linux/blk-mq.h>
#include <linux/vmalloc.h>

#define DISK_SIZE (16 * 1024 * 1024) // 16MB
#define SECTOR_SIZE 512

static struct gendisk *my_disk;
static struct request_queue *my_queue;
static struct blk_mq_tag_set tag_set;
static u8 *dev_data;

// The actual I/O logic
static void my_transfer(struct request *rq) {
    struct bio_vec bvec;
    struct req_iterator iter;
    loff_t pos = blk_rq_pos(rq) * SECTOR_SIZE;
    void *buffer;

    // Iterate over all segments in the request
    rq_for_each_segment(bvec, rq, iter) {
        size_t len = bvec.bv_len;
        void *kaddr = kmap_atomic(bvec.bv_page) + bvec.bv_offset;
        
        buffer = dev_data + pos;

        if (rq_data_dir(rq) == WRITE)
            memcpy(buffer, kaddr, len);
        else
            memcpy(kaddr, buffer, len);

        kunmap_atomic(kaddr);
        pos += len;
    }
}

// Queue Callback
static blk_status_t my_queue_rq(struct blk_mq_hw_ctx *hctx,
                                const struct blk_mq_queue_data *bd) {
    struct request *rq = bd->rq;

    blk_mq_start_request(rq);

    if (blk_rq_pos(rq) * SECTOR_SIZE + blk_rq_bytes(rq) > DISK_SIZE) {
        pr_err("MyRAM: Request beyond end of device\n");
        blk_mq_end_request(rq, BLK_STS_IOERR);
        return BLK_STS_OK;
    }

    my_transfer(rq);
    
    blk_mq_end_request(rq, BLK_STS_OK);
    return BLK_STS_OK;
}

static const struct blk_mq_ops my_mq_ops = {
    .queue_rq = my_queue_rq,
};

static const struct block_device_operations my_fops = {
    .owner = THIS_MODULE,
};

static int __init my_init(void) {
    dev_data = vmalloc(DISK_SIZE);
    if (!dev_data) return -ENOMEM;

    // 1. Init Tag Set
    tag_set.ops = &my_mq_ops;
    tag_set.nr_hw_queues = 1;
    tag_set.queue_depth = 128;
    tag_set.numa_node = NUMA_NO_NODE;
    tag_set.cmd_size = 0;
    tag_set.flags = BLK_MQ_F_SHOULD_MERGE;
    blk_mq_alloc_tag_set(&tag_set);

    // 2. Init Queue
    my_disk = blk_mq_alloc_disk(&tag_set, NULL);
    my_queue = my_disk->queue;

    // 3. Setup Disk
    my_disk->major = register_blkdev(0, "myram");
    my_disk->first_minor = 0;
    my_disk->fops = &my_fops;
    sprintf(my_disk->disk_name, "myram0");
    set_capacity(my_disk, DISK_SIZE / SECTOR_SIZE);

    add_disk(my_disk);
    return 0;
}

static void __exit my_exit(void) {
    del_gendisk(my_disk);
    blk_cleanup_disk(my_disk);
    blk_mq_free_tag_set(&tag_set);
    unregister_blkdev(my_disk->major, "myram");
    vfree(dev_data);
}

module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
```

---

## 🔬 Lab Exercise: Lab 212.1 - Using the RAM Disk

### 1. Lab Objectives
- Format the disk with ext4.
- Mount it.
- Write files.

### 2. Step-by-Step Guide
1.  **Load:** `insmod myram.ko`.
2.  **Verify:** `lsblk` should show `myram0` (16M).
3.  **Format:**
    ```bash
    mkfs.ext4 /dev/myram0
    ```
4.  **Mount:**
    ```bash
    mkdir /mnt/ram
    mount /dev/myram0 /mnt/ram
    ```
5.  **Test:**
    ```bash
    echo "Hello RAM" > /mnt/ram/hello.txt
    cat /mnt/ram/hello.txt
    ```
6.  **Unmount:** `umount /mnt/ram`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Persistence (Sort of)
- **Goal:** Pre-load data.
- **Task:**
    *   In `my_init`, `memcpy` a "Welcome" message into `dev_data` at offset 0 (or wherever).
    *   Or implement a "save to file" feature on unload (requires `filp_open` - tricky in kernel).

### Lab 3: Partitioning
- **Goal:** Create partitions.
- **Task:**
    *   Use `fdisk /dev/myram0`.
    *   Create 2 partitions.
    *   Format them separately.
    *   (Kernel handles partitions automatically if `alloc_disk` allows minors).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. System Crash on Copy
*   **Cause:** `kmap_atomic` context issue (sleeping) or buffer overflow.
*   **Fix:** `kmap_atomic` disables preemption. Do not sleep inside the loop. Ensure `pos` checks are correct.

#### 2. "Request beyond end of device"
*   **Cause:** Filesystem trying to read the last sector + 1 (read-ahead).
*   **Fix:** Ensure boundary checks are strict.

---

## ⚡ Optimization & Best Practices

### `blk_mq_start_request`
*   Must be called before processing.
*   `blk_mq_end_request` must be called after.

### `kmap_local_page`
*   Newer replacement for `kmap_atomic`. Preferred in modern kernels.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we need `kmap_atomic`?
    *   **A:** The pages in the `bio` might be in High Memory (on 32-bit systems) which is not permanently mapped in kernel space. `kmap` creates a temporary mapping.
2.  **Q:** What happens if I don't set `BLK_MQ_F_SHOULD_MERGE`?
    *   **A:** The block layer won't merge adjacent bios into a single request. You will get many small requests, hurting performance (though for a RAM disk, it matters less than for a HDD).

### Challenge Task
> **Task:** "The Encrypted RAM Disk".
> *   Modify `my_transfer`.
> *   XOR every byte with 0x55 during Read and Write.
> *   (Simple obfuscation/encryption).

---

## 📚 Further Reading & References
- [Kernel Documentation: block/blk-mq.rst](https://www.kernel.org/doc/html/latest/block/blk-mq.html)
- [LDD3 Chapter 16 (Block Drivers)](https://lwn.net/Kernel/LDD3/) (Note: LDD3 is very old, uses legacy API. Use only for concepts).

---
