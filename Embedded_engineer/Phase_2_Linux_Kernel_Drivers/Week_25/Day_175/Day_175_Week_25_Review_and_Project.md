# Day 175: Week 25 Review & Project - The Virtual Ethernet Switch
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
1.  **Synthesize** Week 25 concepts (Netdev, NAPI, DMA, PHY, VLAN).
2.  **Architect** a multi-interface network driver.
3.  **Implement** a Virtual Switch that forwards packets between two interfaces (`veth` style).
4.  **Demonstrate** NAPI polling in a virtual environment.
5.  **Debug** packet flow using `tcpdump` and `ethtool`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `iperf3` (for performance testing).
*   **Prior Knowledge:**
    *   Week 25 Content.

---

## 🔄 Week 25 Review

### 1. Netdev Basics (Day 170)
*   **Struct:** `net_device`.
*   **Ops:** `ndo_start_xmit`.
*   **SKB:** Socket Buffer management.

### 2. NAPI (Day 171)
*   **Poll:** `napi_poll`.
*   **Schedule:** `napi_schedule`.
*   **Benefit:** Reduces IRQ load.

### 3. DMA & Rings (Day 172)
*   **Descriptors:** Shared memory with HW.
*   **Mapping:** `dma_map_single`.

### 4. PHY & MDIO (Day 173)
*   **Bus:** `mdiobus_register`.
*   **Lib:** `phy_connect`.

### 5. Advanced Features (Day 174)
*   **VLAN:** HW Offload.
*   **Multicast:** Hash filtering.

---

## 🛠️ Project: The "vSwitch"

### 📋 Project Requirements
Create a driver `vswitch` that:
1.  **Creates** two interfaces: `vsw0` and `vsw1`.
2.  **Logic:** Packets sent on `vsw0` appear as RX on `vsw1`, and vice versa.
3.  **NAPI:** Use NAPI to process the "RX" side.
4.  **Stats:** Correctly track TX/RX bytes and packets.
5.  **VLAN:** Pass VLAN tags transparently.

---

## 💻 Implementation: Step-by-Step Guide

### 🔹 Phase 1: The Shared Data
We need a way for `vsw0` to find `vsw1`.
```c
struct vswitch_priv {
    struct net_device *dev;
    struct net_device *peer; // The other interface
    struct napi_struct napi;
    struct sk_buff_head rx_queue; // Packets waiting to be processed
    struct net_device_stats stats;
};
```

### 🔹 Phase 2: Transmit (The Cross-Over)
When `vsw0` transmits, we queue the SKB to `vsw1`'s RX queue and schedule `vsw1`'s NAPI.

```c
static netdev_tx_t vswitch_tx(struct sk_buff *skb, struct net_device *dev) {
    struct vswitch_priv *priv = netdev_priv(dev);
    struct vswitch_priv *peer = netdev_priv(priv->peer);
    
    // 1. Queue to Peer
    skb_queue_tail(&peer->rx_queue, skb);
    
    // 2. Update Stats
    priv->stats.tx_packets++;
    priv->stats.tx_bytes += skb->len;
    
    // 3. Trigger Peer NAPI
    if (napi_schedule_prep(&peer->napi))
        __napi_schedule(&peer->napi);
        
    return NETDEV_TX_OK;
}
```

### 🔹 Phase 3: NAPI Poll (Receive)
```c
static int vswitch_poll(struct napi_struct *napi, int budget) {
    struct vswitch_priv *priv = container_of(napi, struct vswitch_priv, napi);
    int work = 0;
    
    while (work < budget) {
        struct sk_buff *skb = skb_dequeue(&priv->rx_queue);
        if (!skb) break;
        
        // 1. Update SKB for RX
        skb->dev = priv->dev;
        skb->protocol = eth_type_trans(skb, priv->dev);
        skb->ip_summed = CHECKSUM_UNNECESSARY; // It's virtual, so it's correct
        
        // 2. Push to Stack
        netif_receive_skb(skb);
        
        priv->stats.rx_packets++;
        priv->stats.rx_bytes += skb->len;
        work++;
    }
    
    if (work < budget)
        napi_complete_done(napi, work);
        
    return work;
}
```

### 🔹 Phase 4: Initialization
```c
// In Init
dev1 = alloc_etherdev(sizeof(struct vswitch_priv));
dev2 = alloc_etherdev(sizeof(struct vswitch_priv));

priv1 = netdev_priv(dev1);
priv2 = netdev_priv(dev2);

priv1->peer = dev2;
priv2->peer = dev1;

// Register both...
```

---

## 💻 Implementation: Testing

> **Instruction:** Compile and Load.

### 👨‍💻 Command Line Steps

1.  **Load:** `insmod vswitch.ko`
2.  **Configure:**
    ```bash
    ip link set vsw0 up
    ip link set vsw1 up
    ip addr add 10.0.0.1/24 dev vsw0
    ip addr add 10.0.0.2/24 dev vsw1
    ```
3.  **Ping:**
    ```bash
    ping -I vsw0 10.0.0.2
    ```
    *   **Result:** Success! `vsw0` TX -> `vsw1` RX -> Kernel Reply -> `vsw1` TX -> `vsw0` RX.

4.  **Performance:**
    ```bash
    iperf3 -s -B 10.0.0.2 &
    iperf3 -c 10.0.0.2 -B 10.0.0.1
    ```
    *   Should achieve very high throughput (Gbps+) since it's memory copy only.

---

## 📈 Grading Rubric

| Criteria | Excellent (A) | Good (B) | Needs Improvement (C) |
| :--- | :--- | :--- | :--- |
| **Throughput** | > 1 Gbps. | < 100 Mbps. | Crashes under load. |
| **NAPI** | Correctly implemented. | IRQ context used for RX. | NAPI logic buggy. |
| **Cleanup** | Clean unload without leaks. | Memory leaks on rmmod. | Kernel Panic. |

---

## 🔮 Looking Ahead: Phase 3
Congratulations! You have completed **Week 25**.
Next week, we start **Phase 3: Embedded Android**.
*   We will explore the Android Open Source Project (AOSP).
*   We will learn how to build Android for an embedded board.
*   We will write HALs to connect our drivers to Android Frameworks.

---
