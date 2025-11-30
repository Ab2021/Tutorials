# Day 174: Advanced Networking (VLAN, Bridge, Bonding)
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
1.  **Implement** Hardware VLAN Offload (`NETIF_F_HW_VLAN_CTAG_TX`).
2.  **Handle** Promiscuous Mode for Bridging.
3.  **Understand** how Bonding/Team drivers interact with the physical driver.
4.  **Implement** Multicast Filtering (`ndo_set_rx_mode`).
5.  **Debug** VLAN tag issues using `tcpdump -e`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `vconfig` (or `ip link add link ... type vlan`).
    *   `brctl` (or `ip link add ... type bridge`).
*   **Prior Knowledge:**
    *   Day 170 (Netdev).
    *   802.1Q VLAN Tagging.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: VLAN Offload
Standard Ethernet frames are 1514 bytes. VLAN adds 4 bytes (Tag).
*   **Software VLAN:** The kernel adds the 4 bytes to the payload. MTU issues ensue.
*   **Hardware VLAN:** The kernel passes the Tag in `skb->vlan_tci`. The Driver tells the NIC to insert it.
*   **RX:** NIC strips the tag and puts it in the descriptor. Driver puts it in `skb->vlan_tci`.

### 🔹 Part 2: Multicast Filtering
*   **Unicast:** Addressed to me.
*   **Broadcast:** Addressed to everyone (FF:FF:FF:FF:FF:FF).
*   **Multicast:** Addressed to a group (01:00:5E:...).
*   **Hardware Filter:** NICs usually have a Hash Table (64 bits) to filter multicast MACs. The driver must calculate the hash and update the registers.

---

## 💻 Implementation: VLAN Offload

> **Instruction:** Enable VLAN features.

### 👨‍💻 Code Implementation

#### Step 1: Features Flag
```c
// In Setup
dev->features |= NETIF_F_HW_VLAN_CTAG_TX | NETIF_F_HW_VLAN_CTAG_RX;
dev->hw_features |= NETIF_F_HW_VLAN_CTAG_TX | NETIF_F_HW_VLAN_CTAG_RX;
```

#### Step 2: TX Path
```c
static netdev_tx_t snull_tx(struct sk_buff *skb, struct net_device *dev) {
    // ...
    if (skb_vlan_tag_present(skb)) {
        u16 tag = skb_vlan_tag_get(skb);
        // Tell HW to insert tag
        desc->vlan_tag = cpu_to_le16(tag);
        desc->cmd |= DESC_CMD_INS_VLAN;
    }
    // ...
}
```

#### Step 3: RX Path
```c
// In Poll
if (desc->status & DESC_STAT_HAS_VLAN) {
    u16 tag = le16_to_cpu(desc->vlan_tag);
    __vlan_hwaccel_put_tag(skb, htons(ETH_P_8021Q), tag);
}
```

---

## 💻 Implementation: Multicast Filtering

> **Instruction:** Implement `ndo_set_rx_mode`.

### 👨‍💻 Code Implementation

```c
static void snull_set_rx_mode(struct net_device *dev) {
    struct snull_priv *priv = netdev_priv(dev);
    u32 mc_filter[2]; // 64-bit hash table
    struct netdev_hw_addr *ha;
    
    // 1. Promiscuous Mode
    if (dev->flags & IFF_PROMISC) {
        writel(MODE_PROMISC, priv->io_base + REG_RX_MODE);
        return;
    }
    
    // 2. All Multicast
    if (dev->flags & IFF_ALLMULTI) {
        writel(MODE_ALLMULTI, priv->io_base + REG_RX_MODE);
        return;
    }
    
    // 3. Specific Multicast List
    memset(mc_filter, 0, sizeof(mc_filter));
    
    netdev_for_each_mc_addr(ha, dev) {
        int bit_nr = ether_crc(ETH_ALEN, ha->addr) >> 26; // Top 6 bits
        mc_filter[bit_nr >> 5] |= (1 << (bit_nr & 31));
    }
    
    writel(mc_filter[0], priv->io_base + REG_MC_HASH_L);
    writel(mc_filter[1], priv->io_base + REG_MC_HASH_H);
}
```

---

## 🔬 Lab Exercise: Lab 174.1 - VLAN Testing

### 1. Lab Objectives
- Create a VLAN interface.
- Send packets.
- Verify offload.

### 2. Step-by-Step Guide
1.  **Create VLAN:**
    ```bash
    ip link add link sn0 name sn0.100 type vlan id 100
    ip link set sn0.100 up
    ip addr add 10.0.0.1/24 dev sn0.100
    ```
2.  **Ping:**
    ```bash
    ping 10.0.0.2
    ```
3.  **Debug:**
    *   In `snull_tx`, print `skb_vlan_tag_present(skb)`. It should be true.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Bridge Support
- **Goal:** Add interface to a bridge.
- **Task:**
    1.  `ip link add br0 type bridge`.
    2.  `ip link set sn0 master br0`.
    3.  The kernel automatically sets `IFF_PROMISC` on sn0.
    4.  Verify `snull_set_rx_mode` is called.

### Lab 3: Checksum Offload
- **Goal:** Enable `NETIF_F_IP_CSUM`.
- **Task:**
    1.  Add flag to features.
    2.  In TX, check `skb->ip_summed == CHECKSUM_PARTIAL`.
    3.  Tell HW to calculate checksum starting at `skb_checksum_start_offset(skb)`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. VLAN packets dropped
*   **Cause:** HW VLAN filtering enabled but VLAN ID not added to filter.
*   **Fix:** Implement `ndo_vlan_rx_add_vid` / `kill_vid` to update HW filter.

#### 2. Multicast not working
*   **Cause:** Hash calculation mismatch (CRC32 vs CRC32C vs Bit Reversal).
*   **Fix:** Check datasheet carefully for Hash algorithm.

---

## ⚡ Optimization & Best Practices

### `netdev_features_change`
*   If the user toggles features via `ethtool -K sn0 rx-vlan-offload off`, the kernel calls `ndo_fix_features` and `ndo_set_features`.
*   Driver must update HW state accordingly.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `ETH_P_8021Q`?
    *   **A:** The EtherType for VLAN (0x8100).
2.  **Q:** Why do we need Promiscuous mode for Bridging?
    *   **A:** Because the bridge might need to forward packets destined for a MAC address that is NOT the NIC's own MAC. The NIC must accept everything.

### Challenge Task
> **Task:** "The MacVLAN".
> *   Create a MacVLAN interface on top of sn0.
> *   `ip link add link sn0 name mac0 type macvlan`.
> *   This creates a virtual interface with a different MAC.
> *   Ensure your driver accepts packets for this new MAC (Unicast Filtering).

---

## 📚 Further Reading & References
- [Kernel Documentation: networking/netdev-features.rst](https://www.kernel.org/doc/html/latest/networking/netdev-features.html)

---
