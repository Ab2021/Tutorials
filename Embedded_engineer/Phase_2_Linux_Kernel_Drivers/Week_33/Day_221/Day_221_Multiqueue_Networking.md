# Day 221: Multiqueue Networking (RSS/RPS)
## Phase 2: Linux Kernel & Device Drivers | Week 33: Advanced Network Drivers

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
1.  **Explain** RSS (Receive Side Scaling) and how it distributes load across CPUs.
2.  **Configure** RSS Indirection Tables.
3.  **Implement** Multiqueue support in a driver (`alloc_etherdev_mq`).
4.  **Understand** RPS/RFS (Software alternatives).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Day 214 (Blk-MQ - similar concept).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Single Queue Bottleneck
*   If all interrupts go to CPU 0, CPU 0 hits 100% while others are idle.
*   **RSS (Hardware):** NIC hashes the packet headers (IP/Port). Uses hash to select an RX Ring. Each Ring has its own MSI-X vector pinned to a different CPU.
*   **Result:** Traffic is load-balanced across cores *while preserving flow ordering* (packets from same connection go to same CPU).

### 🔹 Part 2: RPS (Receive Packet Steering)
*   **Software RSS:** If NIC is dumb (single queue), Kernel can hash the packet in the ISR and enqueue it to another CPU's backlog (`softnet_data`).
*   **Cost:** Inter-Processor Interrupt (IPI) overhead.

---

## 💻 Implementation: Multiqueue Driver

> **Instruction:** Initialize a driver with 4 TX and 4 RX queues.

### 👨‍💻 Code Implementation

```c
#include <linux/netdevice.h>

#define NUM_QUEUES 4

// In probe
struct net_device *netdev;

// 1. Alloc MQ Device
netdev = alloc_etherdev_mq(sizeof(struct my_priv), NUM_QUEUES);

// 2. Setup Queues
struct my_priv *priv = netdev_priv(netdev);
for (i = 0; i < NUM_QUEUES; i++) {
    // Alloc Ring i
    // Request MSI-X vector i
}

// 3. TX Selection (Optional, kernel default is hash)
u16 my_select_queue(struct net_device *dev, struct sk_buff *skb,
                    struct net_device *sb_dev) {
    // Custom logic or fallback
    return netdev_pick_tx(dev, skb, sb_dev);
}

static const struct net_device_ops my_netdev_ops = {
    .ndo_start_xmit = my_start_xmit,
    .ndo_select_queue = my_select_queue,
    // ...
};

// 4. In TX Path
static netdev_tx_t my_start_xmit(struct sk_buff *skb, struct net_device *netdev) {
    u16 queue_index = skb_get_queue_mapping(skb);
    struct my_ring *tx_ring = &priv->tx_rings[queue_index];
    
    // Lock this specific ring
    spin_lock(&tx_ring->lock);
    // ... transmit ...
    spin_unlock(&tx_ring->lock);
}
```

---

## 🔬 Lab Exercise: Lab 221.1 - Configuring RSS

### 1. Lab Objectives
- Use `ethtool` to inspect/change RSS hash key and indirection table.

### 2. Step-by-Step Guide
1.  **Show Channels:**
    ```bash
    ethtool -l eth0
    ```
2.  **Show Indirection Table:**
    ```bash
    ethtool -x eth0
    ```
3.  **Change Hash Key:**
    ```bash
    ethtool -X eth0 hkey <hex...>
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Configuring RPS (Software)
- **Goal:** Enable RPS on a single-queue NIC.
- **Task:**
    1.  `cat /sys/class/net/eth0/queues/rx-0/rps_cpus`.
    2.  `echo f > ...` (Enable on 4 CPUs).
    3.  Generate load. Watch `mpstat -P ALL 1`. CPU load should spread.

### Lab 3: RFS (Receive Flow Steering)
- **Goal:** Locality.
- **Task:**
    *   RFS tries to send the packet to the CPU where the *application* is running (reading the socket).
    *   Requires `CONFIG_RPS` and `CONFIG_RFS_ACCEL`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Out of Order Packets
*   **Cause:** Changing affinity/hash on the fly, or Round-Robin scheduling.
*   **Fix:** Always use hashing (Flow preservation).

#### 2. Lock Contention
*   **Cause:** Sharing locks between queues.
*   **Fix:** Use per-queue locks (`spin_lock`).

---

## ⚡ Optimization & Best Practices

### Cache Locality
*   Ensure the RX Ring for CPU N is allocated on the NUMA node of CPU N.
*   Ensure the MSI-X vector for Queue N is pinned to CPU N.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the Indirection Table?
    *   **A:** A table (e.g., 128 entries) that maps the Hash Value (modulo 128) to a Queue Index. Allows rebalancing weights without changing the hash function.
2.  **Q:** Why `alloc_etherdev_mq`?
    *   **A:** It allocates multiple `netdev_queue` structures, allowing the kernel stack to track state (like XOFF/XON) for each queue independently.

### Challenge Task
> **Task:** "The Manual Balancer".
> *   Write a script that monitors CPU usage.
> *   If one CPU is overloaded, use `ethtool -X` to update the indirection table and move flows away from that CPU.

---

## 📚 Further Reading & References
- [Kernel Documentation: networking/scaling.rst](https://www.kernel.org/doc/html/latest/networking/scaling.html)

---
