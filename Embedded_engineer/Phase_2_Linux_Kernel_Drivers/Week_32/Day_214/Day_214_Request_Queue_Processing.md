# Day 214: Request Queue Processing (blk-mq)
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
1.  **Explain** the architecture of `blk-mq` (Software Queues vs Hardware Queues).
2.  **Configure** `struct blk_mq_tag_set` for multi-queue devices.
3.  **Map** CPU queues to Hardware queues (`map_queues`).
4.  **Implement** `queue_rq` for a multi-queue simulated driver.

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

### 🔹 Part 1: The Scalability Problem
*   **Legacy Block Layer:** Single lock (`q->queue_lock`) for the entire queue. Bottleneck on many-core systems with high-IOPS SSDs.
*   **Blk-MQ Solution:**
    *   **Software Queues (ctx):** One per CPU. Lockless submission.
    *   **Hardware Queues (hctx):** One per hardware dispatch channel (e.g., NVMe submission queue).
    *   **Mapping:** M:N mapping. (e.g., 64 CPUs -> 4 HW Queues).

### 🔹 Part 2: Tag Sets
*   **Tags:** Instead of allocating `struct request` dynamically (slow), `blk-mq` pre-allocates a pool of requests identified by an integer (Tag).
*   **Tag Set:** Shared across all queues of a device (or multiple devices).

---

## 💻 Implementation: Multi-Queue RAM Disk

> **Instruction:** Simulate a device with 4 Hardware Queues.

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <linux/blk-mq.h>

#define NR_HW_QUEUES 4

static struct blk_mq_tag_set tag_set;

// Queue Callback
static blk_status_t my_mq_queue_rq(struct blk_mq_hw_ctx *hctx,
                                   const struct blk_mq_queue_data *bd) {
    struct request *rq = bd->rq;
    
    // Check which HW queue we are on
    // hctx->queue_num (0 to 3)
    
    blk_mq_start_request(rq);
    
    // Simulate processing (e.g., print which queue)
    // pr_info("Req on HW Queue %d\n", hctx->queue_num);
    
    // ... Transfer ...
    
    blk_mq_end_request(rq, BLK_STS_OK);
    return BLK_STS_OK;
}

static const struct blk_mq_ops my_mq_ops = {
    .queue_rq = my_mq_queue_rq,
    .map_queues = blk_mq_map_queues, // Default mapping
};

// ... Init ...
    tag_set.ops = &my_mq_ops;
    tag_set.nr_hw_queues = NR_HW_QUEUES; // 4 Queues
    tag_set.queue_depth = 128;
    tag_set.flags = BLK_MQ_F_SHOULD_MERGE;
    
    blk_mq_alloc_tag_set(&tag_set);
// ...
```

---

## 🔬 Lab Exercise: Lab 214.1 - Queue Mapping

### 1. Lab Objectives
- Inspect the mapping of CPUs to HW Queues.

### 2. Step-by-Step Guide
1.  **Load Module:** `insmod my_mq.ko`.
2.  **Check Sysfs:**
    ```bash
    ls /sys/block/myram0/mq/
    # Should see: 0, 1, 2, 3 (Directories for hctx)
    ```
3.  **Check CPU Mapping:**
    ```bash
    cat /sys/block/myram0/mq/0/cpu_list
    # e.g., 0, 1, 2, 3 (CPUs mapped to HW Queue 0)
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Custom Mapping
- **Goal:** Pin CPUs to Queues manually.
- **Task:**
    *   Implement `.map_queues` in `blk_mq_ops`.
    *   Use `blk_mq_map_queues` helper or write your own logic.
    *   Example: CPU 0 -> Queue 0, CPU 1 -> Queue 1.

### Lab 3: Blocking in Queue_RQ
- **Goal:** Handle "Device Busy".
- **Task:**
    *   Return `BLK_STS_RESOURCE` or `BLK_STS_DEV_RESOURCE`.
    *   The block layer will stop sending requests and retry later.
    *   Use `blk_mq_delay_kick_requeue_list` to restart.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. High CPU Usage
*   **Cause:** `queue_rq` returning `BLK_STS_RESOURCE` continuously (Busy Loop).
*   **Fix:** Only return Busy if you really can't process. Ensure you have a mechanism to wake up the queue (interrupt or timer).

---

## ⚡ Optimization & Best Practices

### `BLK_MQ_F_BLOCKING`
*   If your `queue_rq` needs to sleep (e.g., waiting for a mutex), set this flag in `tag_set.flags`.
*   If not set, `queue_rq` is called in atomic context (RCU read lock or spinlock held).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the benefit of `blk-mq` over the legacy block layer?
    *   **A:** Parallel submission. Multiple CPUs can submit I/O simultaneously without fighting for a single lock.
2.  **Q:** How does `blk-mq` handle merging?
    *   **A:** Merging happens in the Software Queue (per-CPU). Once a request is moved to the Hardware Queue, it is usually final (though some drivers support further merging).

### Challenge Task
> **Task:** "The Load Balancer".
> *   Create a driver with 2 HW Queues.
> *   In `map_queues`, assign even CPUs to Queue 0 and odd CPUs to Queue 1.
> *   Run `fio` pinned to specific CPUs and verify traffic goes to the correct HW Queue.

---

## 📚 Further Reading & References
- [Kernel Documentation: block/blk-mq.rst](https://www.kernel.org/doc/html/latest/block/blk-mq.html)
- [LWN: The Multi-Queue Block Layer](https://lwn.net/Articles/552904/)

---
