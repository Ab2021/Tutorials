# Day 216: Advanced Block Layer (IO Schedulers, Tagging)
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
1.  **Compare** different IO Schedulers (MQ-Deadline, BFQ, Kyber).
2.  **Switch** schedulers at runtime via Sysfs.
3.  **Understand** Hardware Tagging (NCQ) vs Software Tagging.
4.  **Tune** the queue depth and scheduler parameters.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
    *   `fio`.
*   **Prior Knowledge:**
    *   Day 214 (Blk-MQ).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Need for Scheduling
*   **HDD:** Seek time is expensive (ms). Merging and sorting requests by LBA (Elevator algorithm) is critical.
*   **SSD:** Seek time is negligible (us). Latency and parallelism matter more.
*   **Blk-MQ Schedulers:**
    *   **None:** FIFO. Best for NVMe (hardware handles it).
    *   **MQ-Deadline:** Simple deadline. Good for SSDs.
    *   **BFQ (Budget Fair Queueing):** Complex, proportional share. Good for HDDs and desktop responsiveness.
    *   **Kyber:** Token-based, latency-targeted. Good for fast devices.

### 🔹 Part 2: Tagging
*   **Software Tags:** Kernel manages tags. Driver maps tag to hardware command.
*   **Hardware Tags:** Device supports tags natively (e.g., SATA NCQ, NVMe).
*   **Tag Depth:** How many requests can be "in flight" (pending) at once.

---

## 💻 Implementation: Tuning the Driver

> **Instruction:** Modify our RAM disk to support changing schedulers and queue depth.

### 👨‍💻 Code Implementation

```c
// In init:
tag_set.queue_depth = 64; // Can be tuned
tag_set.flags = BLK_MQ_F_SHOULD_MERGE; // Enable scheduler merging

// There is no specific code needed to "enable" schedulers.
// The block layer automatically attaches available schedulers if SHOULD_MERGE is set.
```

---

## 🔬 Lab Exercise: Lab 216.1 - Scheduler Switching

### 1. Lab Objectives
- Check available schedulers.
- Switch them.
- Benchmark.

### 2. Step-by-Step Guide
1.  **Check:**
    ```bash
    cat /sys/block/myram0/queue/scheduler
    # [none] mq-deadline kyber bfq
    ```
2.  **Switch:**
    ```bash
    echo bfq > /sys/block/myram0/queue/scheduler
    ```
3.  **Benchmark (Latency):**
    ```bash
    fio --name=lat --filename=/dev/myram0 --rw=randread --bs=4k --iodepth=1 --numjobs=1
    ```
4.  **Benchmark (Throughput):**
    ```bash
    fio --name=bw --filename=/dev/myram0 --rw=read --bs=1M --iodepth=64 --numjobs=4
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: BFQ Weights
- **Goal:** Prioritize one process over another.
- **Task:**
    1.  Run two `fio` jobs.
    2.  Use `ionice -c 1` (Realtime) for one, `ionice -c 3` (Idle) for the other.
    3.  Observe bandwidth distribution with BFQ. (With `none`, they fight equally).

### Lab 3: Queue Depth
- **Goal:** See the effect of `nr_requests`.
- **Task:**
    1.  `echo 1 > /sys/block/myram0/queue/nr_requests`.
    2.  Run `fio` with `iodepth=64`.
    3.  Performance should tank (serialization).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Scheduler not available
*   **Cause:** Kernel config (`CONFIG_IOSCHED_BFQ`, etc.) missing.
*   **Fix:** Recompile kernel.

#### 2. "None" is the only option
*   **Cause:** `BLK_MQ_F_NO_SCHED` flag set in driver, or single-queue device without `SHOULD_MERGE`.

---

## ⚡ Optimization & Best Practices

### Rotational Flag
*   `blk_queue_flag_set(QUEUE_FLAG_NONROT, q)` for SSDs/RAM Disks.
*   Tells the scheduler that seeking is cheap.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is `none` preferred for NVMe?
    *   **A:** NVMe is so fast that the CPU overhead of sorting/merging in software (lock contention) outweighs the benefit.
2.  **Q:** What is `kyber`?
    *   **A:** A scheduler designed for fast devices. It uses two queues (read/write) and limits the number of tokens (requests) sent to hardware to maintain target latencies.

### Challenge Task
> **Task:** "The Latency Spike".
> *   Write a script that constantly reads from the disk.
> *   Simultaneously, write a huge file.
> *   Compare responsiveness (latency of reads) between `mq-deadline` and `bfq`. BFQ should win.

---

## 📚 Further Reading & References
- [Kernel Documentation: block/bfq-iosched.rst](https://www.kernel.org/doc/html/latest/block/bfq-iosched.html)

---
