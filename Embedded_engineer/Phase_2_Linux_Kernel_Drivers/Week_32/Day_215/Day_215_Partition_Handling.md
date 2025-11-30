# Day 215: Partition Handling & Geometry
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
1.  **Explain** how the kernel detects partitions (`check_partition`).
2.  **Implement** the `getgeo` callback (Cylinders, Heads, Sectors).
3.  **Manage** partition minors (Dynamic vs Static).
4.  **Create** a disk with a pre-filled Partition Table.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
    *   `fdisk`, `parted`.
*   **Prior Knowledge:**
    *   Day 212 (RAM Disk).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Partition Detection
*   **Scan:** When `add_disk` is called, the kernel reads the first sector (LBA 0).
*   **MBR:** Checks for 0x55AA signature and partition table at offset 446.
*   **GPT:** Checks for Protective MBR, then LBA 1 for GPT Header.
*   **Minors:** If partitions are found, the kernel allocates `struct block_device` for each partition (e.g., `sda1`, `sda2`).

### 🔹 Part 2: Geometry (CHS)
*   **Legacy:** Cylinders, Heads, Sectors.
*   **Relevance:** Mostly obsolete for LBA addressing, but `fdisk` and some BIOSes still query it for compatibility.
*   **Callback:** `fops->getgeo`.

---

## 💻 Implementation: Fake Geometry

> **Instruction:** Implement `getgeo` for our RAM disk to make it look like a real HDD.

### 👨‍💻 Code Implementation

```c
#include <linux/hdreg.h> // For struct hd_geometry

static int my_getgeo(struct block_device *bdev, struct hd_geometry *geo) {
    long size;
    
    // Calculate size in sectors
    size = get_capacity(my_disk);
    
    // Fake Geometry
    geo->cylinders = (size & ~0x3f) >> 6;
    geo->heads = 4;
    geo->sectors = 16;
    geo->start = 0;
    
    return 0;
}

static const struct block_device_operations my_fops = {
    .owner = THIS_MODULE,
    .getgeo = my_getgeo,
};
```

---

## 💻 Implementation: Pre-filled Partition Table

> **Instruction:** Write an MBR to the RAM disk buffer during init so it appears partitioned immediately.

### 👨‍💻 Code Implementation

```c
struct mbr_partition {
    u8  status;
    u8  chs_start[3];
    u8  type;
    u8  chs_end[3];
    u32 lba_start;
    u32 lba_size;
} __attribute__((packed));

static void create_mbr(u8 *data) {
    struct mbr_partition *p;
    
    // 1. Signature
    data[510] = 0x55;
    data[511] = 0xAA;
    
    // 2. Partition 1 (Offset 446)
    p = (struct mbr_partition *)(data + 446);
    
    p->status = 0x80; // Bootable
    p->type = 0x83;   // Linux
    p->lba_start = 2048; // Start at 1MB
    p->lba_size = 4096;  // 2MB size
    
    // ... Fill others with 0 ...
}

// Call create_mbr(dev_data) before add_disk()
```

---

## 🔬 Lab Exercise: Lab 215.1 - Verifying Partitions

### 1. Lab Objectives
- Load driver.
- Check if kernel detected `myram0p1`.

### 2. Step-by-Step Guide
1.  **Load:** `insmod myram.ko`.
2.  **Check:**
    ```bash
    lsblk
    # Should see:
    # myram0
    # └─myram0p1
    ```
3.  **Fdisk:**
    ```bash
    fdisk -l /dev/myram0
    ```
    *   Should show the geometry and the partition we created.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Dynamic Partitions
- **Goal:** Use `partprobe`.
- **Task:**
    1.  Use `fdisk` to delete p1 and create p2.
    2.  Write changes.
    3.  Kernel should automatically update `/dev/myram0p*` nodes because `fdisk` calls the `BLKRRPART` ioctl (Re-Read Partition Table).

### Lab 3: GPT Support
- **Goal:** Create a GPT disk.
- **Task:**
    *   Use `sgdisk` or `parted` to format as GPT.
    *   Verify kernel detects it (requires `CONFIG_EFI_PARTITION` enabled in kernel).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Partitions not showing up
*   **Cause:** `alloc_disk(1)` used (only 1 minor).
*   **Fix:** Use `alloc_disk(16)` or set `GENHD_FL_EXT_DEVT` (modern kernels handle minors dynamically).

#### 2. "Unknown partition table"
*   **Cause:** Signature 0x55AA missing.

---

## ⚡ Optimization & Best Practices

### Alignment
*   Ensure partitions start on 4KB boundaries (LBA 8 or 2048).
*   `blk_queue_io_min` and `blk_queue_io_opt` tell userspace about alignment requirements.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the `BLKRRPART` IOCTL?
    *   **A:** It tells the kernel to re-scan the partition table. Triggered by `blockdev --rereadpt` or `fdisk`.
2.  **Q:** Can I have partitions on a partition?
    *   **A:** Generally no (unless using LVM or Device Mapper). Standard MBR/GPT is flat (or nested via Extended partitions, which kernel presents as flat logic).

### Challenge Task
> **Task:** "The Virtual Floppy".
> *   Create a 1.44MB RAM disk.
> *   Implement `getgeo` to match a 3.5" floppy (80 tracks, 2 heads, 18 sectors).
> *   Format with FAT12 (`mkfs.vfat`).

---

## 📚 Further Reading & References
- [Kernel Documentation: block/](https://www.kernel.org/doc/html/latest/block/index.html)
- [MBR Specification](https://en.wikipedia.org/wiki/Master_boot_record)

---
