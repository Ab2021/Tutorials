# Day 219: NAPI Polling & Interrupt Mitigation
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
1.  **Explain** the "Receive Livelock" problem.
2.  **Implement** the NAPI `poll` callback.
3.  **Switch** between Interrupt and Polling modes dynamically.
4.  **Use** `netif_napi_add`, `napi_schedule`, `napi_complete`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Day 218 (Ring Buffers).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Livelock Problem
*   **Scenario:** High packet rate.
*   **Interrupt Mode:** Packet arrives -> IRQ -> ISR -> SoftIRQ -> Process.
*   **Issue:** If packets arrive faster than CPU can process, the CPU spends 100% time in ISR/Context Switch, never making progress on userspace. System freezes.

### 🔹 Part 2: NAPI Solution
*   **Hybrid:**
    1.  Packet arrives -> IRQ.
    2.  ISR disables NIC Interrupts and schedules NAPI Poll.
    3.  SoftIRQ calls `poll()` function.
    4.  `poll()` processes a batch of packets (budget, e.g., 64).
    5.  If ring empty: Re-enable Interrupts and quit.
    6.  If ring not empty: Return, keep Interrupts disabled, kernel calls `poll()` again later.

---

## 💻 Implementation: NAPI Driver

> **Instruction:** Add NAPI to a dummy network driver.

### 👨‍💻 Code Implementation

```c
#include <linux/netdevice.h>

struct my_priv {
    struct napi_struct napi;
    struct my_ring rx_ring;
    void __iomem *mmio;
};

// The Poll Function
static int my_poll(struct napi_struct *napi, int budget) {
    struct my_priv *priv = container_of(napi, struct my_priv, napi);
    int work_done = 0;
    
    // Process up to 'budget' packets
    while (work_done < budget) {
        if (!has_new_packets(&priv->rx_ring))
            break;
            
        process_one_packet(priv); // Alloc skb, netif_receive_skb
        work_done++;
    }
    
    // If we processed fewer than budget, we are done
    if (work_done < budget) {
        if (napi_complete_done(napi, work_done)) {
            // Re-enable Interrupts
            enable_irq_in_hw(priv);
        }
    }
    
    return work_done;
}

// The ISR
static irqreturn_t my_isr(int irq, void *dev_id) {
    struct net_device *netdev = dev_id;
    struct my_priv *priv = netdev_priv(netdev);
    
    // 1. Disable Interrupts in HW
    disable_irq_in_hw(priv);
    
    // 2. Schedule NAPI
    if (likely(napi_schedule_prep(&priv->napi))) {
        __napi_schedule(&priv->napi);
    }
    
    return IRQ_HANDLED;
}

// Init
static int my_open(struct net_device *netdev) {
    struct my_priv *priv = netdev_priv(netdev);
    netif_napi_add(netdev, &priv->napi, my_poll, 64); // 64 is default weight
    napi_enable(&priv->napi);
    // ... request_irq ...
    return 0;
}
```

---

## 🔬 Lab Exercise: Lab 219.1 - Observing NAPI

### 1. Lab Objectives
- Generate load.
- Watch `/proc/softirqs`.

### 2. Step-by-Step Guide
1.  **Load:** `iperf3` or `pktgen`.
2.  **Monitor:**
    ```bash
    watch -n 1 "cat /proc/softirqs | grep NET_RX"
    ```
3.  **Interpret:**
    *   The `NET_RX` count should increase rapidly.
    *   If you see CPU usage in `si` (SoftIRQ) but system remains responsive, NAPI is working.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Budget Tuning
- **Goal:** Change the budget.
- **Task:**
    *   Change `netif_napi_add(..., 16)`.
    *   Observe higher CPU usage (more context switches back to SoftIRQ loop) but lower latency.
    *   Change to `128`. Higher throughput, higher latency.

### Lab 3: Busy Poll
- **Goal:** Userspace polling.
- **Task:**
    *   `ethtool --set-priv-flags eth0 busy-poll on`.
    *   Allows `recv()` syscall to busy-wait on the NAPI context, bypassing the interrupt entirely for ultra-low latency.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. NAPI hangs
*   **Cause:** Forgetting to re-enable interrupts in `napi_complete`.
*   **Fix:** Check the `if (work_done < budget)` block.

#### 2. Interrupt Storm
*   **Cause:** Not disabling interrupts in ISR.
*   **Fix:** Ensure HW interrupt mask is set immediately.

---

## ⚡ Optimization & Best Practices

### GRO (Generic Receive Offload)
*   Use `napi_gro_receive` instead of `netif_receive_skb`.
*   The stack will merge adjacent TCP segments into one large skb *before* passing it up, reducing overhead significantly.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why `napi_schedule_prep`?
    *   **A:** It checks if NAPI is already scheduled (running on another CPU). If so, it returns false, preventing race conditions.
2.  **Q:** What is `NET_RX_SOFTIRQ`?
    *   **A:** The SoftIRQ vector used by NAPI. It runs `net_rx_action`, which iterates over the list of scheduled NAPI structs and calls their `poll` method.

### Challenge Task
> **Task:** "The Adaptive Poller".
> *   Implement a logic that dynamically changes the interrupt coalescing parameters based on NAPI work done.
> *   If `work_done == budget` consistently, increase HW interrupt delay (ITR).

---

## 📚 Further Reading & References
- [Kernel Documentation: networking/napi.rst](https://www.kernel.org/doc/html/latest/networking/napi.html)

---
