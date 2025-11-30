# Day 171: Receiving Packets & NAPI
## Phase 2: Linux Kernel & Device Drivers | Week 25: Network Device Drivers

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
1.  **Explain** the difference between Interrupt-driven RX and NAPI (Polling).
2.  **Implement** a NAPI poll function (`poll_controller`).
3.  **Schedule** NAPI execution (`napi_schedule`).
4.  **Allocate** SKBs for reception (`netdev_alloc_skb`).
5.  **Pass** packets to the stack (`netif_receive_skb`).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 170 (Netdev Basics).
    *   Interrupts (Day 132).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Interrupt Storm
In a 1Gbps network, packets arrive every few microseconds.
If we fire an IRQ for *every* packet:
1.  CPU stops.
2.  Context Switch.
3.  Handle Packet.
4.  Return.
**Result:** The CPU spends 100% time switching contexts (Livelock).

### 🔹 Part 2: NAPI (New API)
NAPI switches between Interrupts and Polling.
1.  **First Packet:** IRQ fires. Driver disables IRQ. Schedules NAPI poll.
2.  **Polling:** Kernel calls driver's `poll()` function.
3.  **Processing:** Driver reads N packets (budget).
4.  **Done:** If no more packets, re-enable IRQ.

---

## 💻 Implementation: NAPI Structure

> **Instruction:** Add NAPI support to our `snull` driver.

### 👨‍💻 Code Implementation

#### Step 1: Structure Update
```c
struct snull_priv {
    // ...
    struct napi_struct napi;
};
```

#### Step 2: The Poll Function
```c
static int snull_poll(struct napi_struct *napi, int budget) {
    struct snull_priv *priv = container_of(napi, struct snull_priv, napi);
    struct net_device *dev = priv->dev;
    int work_done = 0;
    
    // Simulate reading from HW FIFO
    while (work_done < budget) {
        // 1. Check if HW has data
        if (!hw_has_data(priv)) break;
        
        // 2. Allocate SKB
        struct sk_buff *skb = netdev_alloc_skb(dev, PACKET_SIZE + 2);
        if (!skb) {
            dev->stats.rx_dropped++;
            continue;
        }
        
        // 3. Alignment (IP header needs 16-byte alignment)
        skb_reserve(skb, 2);
        
        // 4. Copy Data (Simulated)
        hw_read_data(priv, skb_put(skb, PACKET_SIZE), PACKET_SIZE);
        
        // 5. Protocol ID
        skb->protocol = eth_type_trans(skb, dev);
        
        // 6. Hand to Stack
        netif_receive_skb(skb);
        
        dev->stats.rx_packets++;
        dev->stats.rx_bytes += PACKET_SIZE;
        work_done++;
    }
    
    // If we processed fewer than budget, we are done
    if (work_done < budget) {
        napi_complete_done(napi, work_done);
        // Re-enable Interrupts here
        // enable_rx_irq(priv);
    }
    
    return work_done;
}
```

#### Step 3: Initialization (in Probe)
```c
netif_napi_add(dev, &priv->napi, snull_poll, 64); // 64 is default weight
```

#### Step 4: Interrupt Handler (Simulated)
```c
static irqreturn_t snull_rx_interrupt(int irq, void *dev_id) {
    struct net_device *dev = dev_id;
    struct snull_priv *priv = netdev_priv(dev);
    
    // 1. Disable Interrupts
    // disable_rx_irq(priv);
    
    // 2. Schedule NAPI
    if (napi_schedule_prep(&priv->napi)) {
        __napi_schedule(&priv->napi);
    }
    
    return IRQ_HANDLED;
}
```

---

## 🔬 Lab Exercise: Lab 171.1 - Simulating RX

### 1. Lab Objectives
- Trigger RX via a timer (since we don't have real HW IRQs).
- Verify NAPI poll is called.

### 2. Step-by-Step Guide
1.  **Timer:** Add an `hrtimer` that fires every 100ms.
2.  **Callback:** In timer callback, call `napi_schedule(&priv->napi)`.
3.  **Poll:** In `poll`, create a fake packet (e.g., a Broadcast ARP).
4.  **Verify:** `tcpdump -i sn0`. You should see packets arriving.

---

## 🧪 Additional / Advanced Labs

### Lab 2: GRO (Generic Receive Offload)
- **Goal:** Improve performance.
- **Task:**
    1.  Use `napi_gro_receive` instead of `netif_receive_skb`.
    2.  This allows the kernel to merge multiple TCP packets into one large SKB before passing it up the stack.

### Lab 3: Budget Tuning
- **Goal:** Observe behavior under load.
- **Task:**
    1.  Reduce budget to 1.
    2.  Flood the driver with packets (simulated).
    3.  Observe that `poll` returns `budget` (1), and NAPI stays scheduled, preventing IRQ re-enable.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "NAPI poll called with IRQs enabled"
*   **Cause:** You forgot to disable the HW interrupt in the ISR.
*   **Consequence:** Race condition. IRQ fires while Poll is running.

#### 2. Memory Leaks
*   **Cause:** Allocating SKB but failing to pass it to stack (`netif_receive_skb` frees it on success/failure, but if you error out before calling it, YOU must free it).

---

## ⚡ Optimization & Best Practices

### `skb_reserve(skb, 2)`
*   Ethernet header is 14 bytes.
*   If we read directly into buffer[0], the IP header starts at byte 14 (Not 16-byte aligned).
*   Reserving 2 bytes puts the IP header at byte 16.
*   **Critical** for performance on ARM/MIPS.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the "Budget"?
    *   **A:** The maximum number of packets the driver is allowed to process in one poll call. Usually 64. Prevents one card from hogging the CPU.
2.  **Q:** Why `netif_receive_skb` vs `netif_rx`?
    *   **A:** `netif_rx` puts the packet in a per-CPU backlog queue (Software Interrupt context). `netif_receive_skb` processes it immediately (NAPI context). NAPI drivers should use `netif_receive_skb`.

### Challenge Task
> **Task:** "The Reflector".
> *   Combine TX and RX.
> *   When `snull_tx` receives a packet, put it into a queue.
> *   Fire the RX timer.
> *   In RX poll, take the packet from the queue and send it up via `netif_receive_skb`.
> *   Result: A working Loopback interface!

---

## 📚 Further Reading & References
- [Kernel Documentation: networking/napi.rst](https://www.kernel.org/doc/html/latest/networking/napi.html)

---
