# Day 217: Week 32 Review and Project - The Fault-Tolerant RAID-1 Driver
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
1.  **Synthesize** Week 32 concepts (Gendisk, Bio, Blk-MQ).
2.  **Architect** a stacking block driver (like MD/DM).
3.  **Implement** RAID-1 (Mirroring) logic.
4.  **Handle** partial failures (one disk down).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   All Week 32 tools.
*   **Prior Knowledge:**
    *   Week 32 Content.

---

## 🔄 Week 32 Review

### 1. Architecture (Day 211)
*   `gendisk`, `request_queue`.

### 2. Implementation (Day 212-214)
*   `blk-mq`: Queue-based, high performance.
*   `submit_bio`: Bio-based, flexible.

### 3. Features (Day 215-216)
*   Partitions, Geometry.
*   Schedulers (BFQ, Kyber).

---

## 🛠️ Project: The "Simple RAID-1" (sraid)

### 📋 Project Requirements
1.  **Driver:** `sraid.ko`.
2.  **Underlying Devices:** Two RAM disks (e.g., `/dev/ram0`, `/dev/ram1` or our `myram0`, `myram1`).
    *   *Simplification:* Instead of opening other block devices (which is complex), we will just allocate **two internal vmalloc buffers** and treat them as "Disk A" and "Disk B".
3.  **Functionality:**
    *   **Write:** Write data to *both* buffers.
    *   **Read:** Read from Buffer A. If error, read from Buffer B.
    *   **Corrupt:** Add a sysfs file to manually "corrupt" Buffer A.
    *   **Recover:** Read should automatically fetch from B if A is corrupt.

---

## 💻 Implementation: Step-by-Step Guide

### 🔹 Phase 1: Dual Buffers
```c
static u8 *disk_a;
static u8 *disk_b;
static bool a_corrupt = false;

// In init
disk_a = vmalloc(SIZE);
disk_b = vmalloc(SIZE);
```

### 🔹 Phase 2: The Transfer Logic
```c
static void sraid_transfer(struct request *rq) {
    // ... Iterator ...
    void *buffer_a = disk_a + pos;
    void *buffer_b = disk_b + pos;
    
    if (rq_data_dir(rq) == WRITE) {
        // Mirroring
        memcpy(buffer_a, kaddr, len);
        memcpy(buffer_b, kaddr, len);
    } else {
        // Read
        if (!a_corrupt) {
            memcpy(kaddr, buffer_a, len);
        } else {
            pr_warn_ratelimited("SRAID: Reading from Mirror B!\n");
            memcpy(kaddr, buffer_b, len);
        }
    }
}
```

### 🔹 Phase 3: Corruption Trigger
```c
static ssize_t corrupt_store(struct kobject *kobj, ...) {
    a_corrupt = true;
    // Optionally memset disk_a to garbage
    memset(disk_a, 0xFF, SIZE);
    return count;
}
```

### 🔹 Phase 4: Testing
1.  **Mount:** `mount /dev/sraid0 /mnt`.
2.  **Write:** `echo "Important Data" > /mnt/file`.
3.  **Corrupt:** `echo 1 > /sys/kernel/sraid/corrupt`.
4.  **Read:** `cat /mnt/file`.
    *   Should still print "Important Data" (from B).
    *   `dmesg` should show "Reading from Mirror B!".

---

## 📈 Grading Rubric

| Criteria | Excellent (A) | Good (B) | Needs Improvement (C) |
| :--- | :--- | :--- | :--- |
| **Mirroring** | Writes go to both. | Writes only to one. | Data lost. |
| **Failover** | Reads switch to B on corruption. | Reads fail on corruption. | Crash. |
| **Performance** | No noticeable lag. | Very slow. | Hangs. |

---

## 🔮 Looking Ahead: Week 33
Next week, we start **Phase 3: Advanced Subsystems**.
We will cover **Network Device Drivers (Advanced)**, **NAPI**, and **10GbE Architecture**.

---
