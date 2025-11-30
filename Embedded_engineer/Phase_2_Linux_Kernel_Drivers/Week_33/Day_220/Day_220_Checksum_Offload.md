# Day 220: Checksum Offload & TSO/GSO
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
1.  **Explain** the cost of software checksumming and segmentation.
2.  **Enable** Hardware Checksum Offload in a driver.
3.  **Implement** TSO (TCP Segmentation Offload) support.
4.  **Understand** GSO (Software fallback).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
    *   `ethtool`.
*   **Prior Knowledge:**
    *   Day 218 (10GbE Arch).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Checksumming
*   **Problem:** TCP/IP requires a checksum for data integrity. Calculating it in software (reading every byte) is expensive and pollutes CPU cache.
*   **Solution:** Hardware calculates it on the fly during DMA.
*   **RX:** HW verifies checksum and sets a bit in the descriptor.
*   **TX:** Driver tells HW where to insert the checksum.

### 🔹 Part 2: Segmentation (TSO)
*   **Problem:** MTU is usually 1500 bytes. Applications send 64KB buffers. CPU must split 64KB into 44 packets, copying headers for each.
*   **Solution:** CPU sends one huge 64KB packet (Super Packet) to NIC. NIC splits it into MTU-sized frames and replicates headers.
*   **Benefit:** Massive CPU savings.

---

## 💻 Implementation: Enabling Offloads

> **Instruction:** Configure `net_device` features.

### 👨‍💻 Code Implementation

```c
// In probe/setup
struct net_device *netdev;

// 1. Advertise Features
netdev->hw_features = NETIF_F_SG |          // Scatter-Gather (Required for TSO)
                      NETIF_F_IP_CSUM |     // IPv4 Checksum
                      NETIF_F_IPV6_CSUM |   // IPv6 Checksum
                      NETIF_F_TSO;          // TCP Segmentation

netdev->features |= netdev->hw_features;

// 2. Handling TX Checksum
static netdev_tx_t my_start_xmit(struct sk_buff *skb, struct net_device *netdev) {
    if (skb->ip_summed == CHECKSUM_PARTIAL) {
        // Tell HW to calculate checksum
        // skb->csum_start: Offset of checksum header
        // skb->csum_offset: Offset inside header to write result
        
        // Example for Intel NIC:
        // desc->cmd |= CMD_TX_CSUM;
        // desc->csum_offset = skb->csum_offset;
    }
    
    // 3. Handling TSO
    if (skb_is_gso(skb)) {
        // This is a Super Packet!
        // skb_shinfo(skb)->gso_size: The MSS (Maximum Segment Size)
        
        // desc->cmd |= CMD_TX_TSO;
        // desc->mss = skb_shinfo(skb)->gso_size;
        // desc->hdr_len = skb_transport_offset(skb) + tcp_hdrlen(skb);
    }
    
    // ... map dma ...
}
```

---

## 🔬 Lab Exercise: Lab 220.1 - Ethtool Verification

### 1. Lab Objectives
- Toggle offloads.
- Measure CPU usage.

### 2. Step-by-Step Guide
1.  **Check Features:**
    ```bash
    ethtool -k eth0
    ```
2.  **Disable TSO:**
    ```bash
    ethtool -K eth0 tso off
    ```
3.  **Benchmark:**
    *   Run `iperf3`.
    *   With TSO ON: High throughput, low CPU.
    *   With TSO OFF: Throughput might drop, CPU usage spikes (kernel doing segmentation).

---

## 🧪 Additional / Advanced Labs

### Lab 2: RX Checksum
- **Goal:** Verify RX offload.
- **Task:**
    *   In RX path:
        ```c
        if (desc->status & STATUS_CSUM_OK)
            skb->ip_summed = CHECKSUM_UNNECESSARY;
        else
            skb_checksum_none_assert(skb);
        ```

### Lab 3: GSO (Generic Segmentation Offload)
- **Goal:** What if HW doesn't support TSO?
- **Task:**
    *   Kernel does GSO in software just before the driver.
    *   It's still better than the application doing it, but worse than TSO.
    *   Driver doesn't need to do anything special (unless it claims `NETIF_F_TSO`, then it *must* handle it).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Bad Checksums
*   **Cause:** Wrong `csum_start` or `csum_offset`.
*   **Fix:** Dump the packet and offsets. Remember offsets are relative to the start of the packet (or sometimes transport header, depending on HW).

#### 2. TSO Hangs
*   **Cause:** Not handling the header length correctly. HW needs to know what to replicate.
*   **Fix:** Ensure `skb_transport_offset` is correct.

---

## ⚡ Optimization & Best Practices

### Scatter-Gather (SG)
*   TSO requires SG (`NETIF_F_SG`).
*   The Super Packet is usually non-linear (headers in linear part, data in frags).
*   Driver must handle `skb_shinfo(skb)->frags`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `CHECKSUM_PARTIAL`?
    *   **A:** It means the checksum is not yet complete. The HW (or next layer) needs to calculate it.
2.  **Q:** Why does TSO require SG?
    *   **A:** Because creating a contiguous 64KB buffer is hard/impossible. The data is usually scattered in page cache pages.

### Challenge Task
> **Task:** "The Software TSO".
> *   Disable TSO on your NIC.
> *   Write a module that intercepts packets (Netfilter).
> *   Print the size of packets.
> *   Verify they are all MTU sized (because GSO split them).

---

## 📚 Further Reading & References
- [Kernel Documentation: networking/segmentation-offloads.rst](https://www.kernel.org/doc/html/latest/networking/segmentation-offloads.html)

---
