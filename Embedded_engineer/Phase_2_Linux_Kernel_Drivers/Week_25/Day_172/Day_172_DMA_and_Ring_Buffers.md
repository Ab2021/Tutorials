# Day 172: DMA and Ring Buffers (Network)
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
1.  **Explain** the Descriptor Ring architecture used by NICs.
2.  **Allocate** Coherent DMA memory for descriptors (`dma_alloc_coherent`).
3.  **Map** SKB data buffers for Streaming DMA (`dma_map_single`).
4.  **Implement** the TX Ring logic (Producer/Consumer).
5.  **Handle** DMA Unmapping and ownership bits.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 170/171.
    *   Day 131 (DMA Basics).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Descriptor Ring
Network cards don't read linked lists. They read arrays of "Descriptors".
*   **Descriptor:** A struct (shared with HW) containing:
    *   `addr`: Physical address of the data buffer.
    *   `len`: Length of data.
    *   `status`: Ownership bit (Who owns this? CPU or HW?).
*   **Ring:** A circular array of descriptors.

### 🔹 Part 2: TX Flow
1.  **CPU:** Gets SKB.
2.  **CPU:** Maps SKB data (`dma_map_single`).
3.  **CPU:** Writes address to next Descriptor.
4.  **CPU:** Flips ownership bit (CPU -> HW).
5.  **CPU:** Kicks the doorbell (Write to register).
6.  **HW:** Reads descriptor, reads data via DMA, sends packet.
7.  **HW:** Writes status back (TX Done), flips ownership (HW -> CPU), fires IRQ.

---

## 💻 Implementation: Descriptor Structure

> **Instruction:** Define a generic descriptor.

### 👨‍💻 Code Implementation

#### Step 1: The Hardware Struct
```c
struct my_desc {
    __le32 addr_low;
    __le32 addr_high;
    __le32 len_cmd; // Length + Command bits
    __le32 status;  // Status + Ownership
};

#define DESC_OWN_HW  (1 << 31)
#define DESC_CMD_EOP (1 << 30) // End of Packet
```

#### Step 2: The Ring Struct
```c
struct my_ring {
    struct my_desc *desc; // CPU Pointer
    dma_addr_t dma;       // DMA Address of the ring
    unsigned int size;    // Number of descriptors
    unsigned int head;    // Producer (CPU)
    unsigned int tail;    // Consumer (HW completion)
    struct sk_buff **skb_ring; // Shadow ring to hold SKB pointers
};
```

#### Step 3: Allocation (in Open)
```c
static int my_alloc_ring(struct snull_priv *priv) {
    priv->tx_ring.size = 256;
    
    // 1. Allocate Descriptors (Coherent)
    priv->tx_ring.desc = dma_alloc_coherent(priv->dev->dev.parent,
                                            sizeof(struct my_desc) * 256,
                                            &priv->tx_ring.dma, GFP_KERNEL);
    if (!priv->tx_ring.desc) return -ENOMEM;
    
    // 2. Allocate Shadow Ring (Normal Kernel Memory)
    priv->tx_ring.skb_ring = kcalloc(256, sizeof(struct sk_buff *), GFP_KERNEL);
    
    return 0;
}
```

---

## 💻 Implementation: TX Logic

> **Instruction:** Implement `start_xmit` with DMA.

### 👨‍💻 Code Implementation

```c
static netdev_tx_t snull_tx(struct sk_buff *skb, struct net_device *dev) {
    struct snull_priv *priv = netdev_priv(dev);
    struct my_ring *ring = &priv->tx_ring;
    struct my_desc *desc;
    dma_addr_t mapping;
    unsigned int idx = ring->head;
    
    // 1. Check for space
    if (ring_full(ring)) {
        netif_stop_queue(dev);
        return NETDEV_TX_BUSY;
    }
    
    // 2. Map SKB Data
    mapping = dma_map_single(dev->dev.parent, skb->data, skb->len, DMA_TO_DEVICE);
    if (dma_mapping_error(dev->dev.parent, mapping)) {
        dev_kfree_skb(skb);
        return NETDEV_TX_OK; // Drop
    }
    
    // 3. Fill Descriptor
    desc = &ring->desc[idx];
    desc->addr_low = cpu_to_le32(lower_32_bits(mapping));
    desc->addr_high = cpu_to_le32(upper_32_bits(mapping));
    desc->len_cmd = cpu_to_le32(skb->len | DESC_CMD_EOP);
    
    // 4. Save SKB for later unmapping
    ring->skb_ring[idx] = skb;
    
    // 5. Hand over to HW (Memory Barrier needed!)
    wmb(); 
    desc->status = cpu_to_le32(DESC_OWN_HW);
    
    // 6. Advance Head
    ring->head = (ring->head + 1) % ring->size;
    
    // 7. Kick Doorbell
    writel(ring->head, priv->io_base + REG_TX_HEAD);
    
    return NETDEV_TX_OK;
}
```

---

## 💻 Implementation: TX Completion (Cleanup)

> **Instruction:** Called from ISR or NAPI when HW is done.

### 👨‍💻 Code Implementation

```c
static void snull_clean_tx(struct snull_priv *priv) {
    struct my_ring *ring = &priv->tx_ring;
    unsigned int idx = ring->tail;
    
    while (idx != ring->head) {
        struct my_desc *desc = &ring->desc[idx];
        
        // Check if HW is done
        if (le32_to_cpu(desc->status) & DESC_OWN_HW) break;
        
        // Unmap
        struct sk_buff *skb = ring->skb_ring[idx];
        dma_unmap_single(priv->dev->dev.parent, 
                         le32_to_cpu(desc->addr_low), // Simplified
                         skb->len, DMA_TO_DEVICE);
                         
        dev_kfree_skb(skb);
        ring->skb_ring[idx] = NULL;
        
        idx = (idx + 1) % ring->size;
    }
    
    ring->tail = idx;
    
    // Wake queue if we freed enough space
    if (netif_queue_stopped(priv->dev))
        netif_wake_queue(priv->dev);
}
```

---

## 🔬 Lab Exercise: Lab 172.1 - Mocking DMA

### 1. Lab Objectives
- Since we don't have real HW, we will simulate the "HW Read" step.
- Verify the ring logic.

### 2. Step-by-Step Guide
1.  **Mock:** In `snull_tx`, instead of kicking doorbell, call a timer.
2.  **Timer:** In callback, iterate from `tail` to `head`.
3.  **Simulate:** Clear `DESC_OWN_HW` bit.
4.  **Clean:** Call `snull_clean_tx`.
5.  **Verify:** Send 1000 packets. Ensure memory usage doesn't explode (SKBs freed).

---

## 🧪 Additional / Advanced Labs

### Lab 2: RX Ring
- **Goal:** Implement the RX side.
- **Task:**
    1.  Allocate RX Ring.
    2.  Pre-fill it with empty SKBs (`netdev_alloc_skb` + `dma_map_single`).
    3.  Set ownership to HW.
    4.  When HW writes data, it clears ownership.
    5.  In NAPI poll, check ownership, unmap, pass to stack, allocate NEW skb, remap, give back to HW.

### Lab 3: Scatter-Gather TX
- **Goal:** Support `NETIF_F_SG`.
- **Task:**
    1.  Handle `skb_shinfo(skb)->frags`.
    2.  Use multiple descriptors for one packet (Start of Packet, Middle, End of Packet).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Memory Corruption
*   **Cause:** DMA writing to memory after we freed the SKB.
*   **Fix:** Ensure `DESC_OWN_HW` is checked correctly. Never free SKB until HW says it's done.

#### 2. Stalled Queue
*   **Cause:** `netif_stop_queue` called, but `netif_wake_queue` never called because `clean_tx` logic is buggy.
*   **Fix:** Check `tail` vs `head` pointers.

---

## ⚡ Optimization & Best Practices

### `wmb()` and `rmb()`
*   **Write Memory Barrier:** Ensures `desc->addr` is written to RAM *before* `desc->status`. Otherwise, HW might see "OWN_HW" but read garbage address.
*   **Read Memory Barrier:** Ensures we read `desc->status` *before* reading the data.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why `dma_alloc_coherent` for descriptors but `dma_map_single` for packets?
    *   **A:** Descriptors are accessed frequently by both CPU and HW (Control path). Coherent memory disables caching to ensure visibility. Packets are large and accessed once (Data path). Streaming mappings are more efficient (cache lines invalidated/flushed explicitly).
2.  **Q:** What is the "Shadow Ring"?
    *   **A:** The HW descriptor only stores the physical address. The driver needs to remember the `sk_buff *` virtual address to free it later. The shadow ring stores this mapping.

### Challenge Task
> **Task:** "The BQL (Byte Queue Limits)".
> *   Integrate BQL (`netdev_sent_queue`, `netdev_completed_queue`).
> *   This helps the networking stack avoid buffering too much data in the driver, reducing latency (Bufferbloat).

---

## 📚 Further Reading & References
- [Kernel Documentation: DMA-API-HOWTO.txt](https://www.kernel.org/doc/Documentation/DMA-API-HOWTO.txt)

---
