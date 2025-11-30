# Day 170: Network Device Drivers Basics
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
1.  **Explain** the Linux Network Stack architecture (Socket -> Protocol -> Device).
2.  **Allocate** and **Register** a `net_device` (`alloc_etherdev`, `register_netdev`).
3.  **Implement** the `ndo_start_xmit` callback to transmit packets.
4.  **Manage** Socket Buffers (`sk_buff`) basics.
5.  **Configure** the interface using `ip link`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `iproute2` (`ip link`, `ip addr`).
    *   `tcpdump`.
*   **Prior Knowledge:**
    *   Day 133 (Tasklets/Workqueues - needed for RX later).
    *   DMA Concepts.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Network Stack
1.  **User:** `sendto()`
2.  **Socket Layer:** `sys_sendto`
3.  **Protocol Layer (TCP/IP):** Adds headers. Creates `sk_buff`.
4.  **Device Layer (Driver):** `ndo_start_xmit`.
5.  **Hardware:** DMA reads packet, sends to PHY.

### 🔹 Part 2: The `net_device` Structure
Represents a network interface (`eth0`, `wlan0`).
*   **Name:** "eth%d" (Auto-numbered).
*   **MAC Address:** `dev_addr`.
*   **Stats:** `net_device_stats` (RX packets, TX errors).
*   **Ops:** `net_device_ops` (Open, Stop, Xmit).

### 🔹 Part 3: The `sk_buff` (Socket Buffer)
The most important structure in networking.
*   **Head/Data:** Pointers to the buffer start.
*   **Tail/End:** Pointers to the buffer end.
*   **len:** Length of data.
*   **protocol:** Ethernet protocol ID (e.g., IPv4).

---

## 💻 Implementation: The Dummy Network Driver

> **Instruction:** We will create a loopback-style driver named `snull` (Simple Null).

### 👨‍💻 Code Implementation

#### Step 1: Structure
```c
#include <linux/module.h>
#include <linux/netdevice.h>
#include <linux/etherdevice.h> // alloc_etherdev

struct snull_priv {
    struct net_device_stats stats;
    struct sk_buff *skb;
    spinlock_t lock;
};
```

#### Step 2: Open/Stop
```c
static int snull_open(struct net_device *dev) {
    netif_start_queue(dev); // Allow kernel to send packets
    return 0;
}

static int snull_stop(struct net_device *dev) {
    netif_stop_queue(dev);
    return 0;
}
```

#### Step 3: Transmit (Xmit)
Called when the kernel wants to send a packet.
```c
static netdev_tx_t snull_tx(struct sk_buff *skb, struct net_device *dev) {
    struct snull_priv *priv = netdev_priv(dev);
    
    // 1. Simulate Hardware TX (Just free it for now)
    dev->stats.tx_packets++;
    dev->stats.tx_bytes += skb->len;
    
    // 2. Free the SKB (Driver owns it now)
    dev_kfree_skb(skb);
    
    return NETDEV_TX_OK;
}
```

#### Step 4: Ops & Setup
```c
static const struct net_device_ops snull_netdev_ops = {
    .ndo_open = snull_open,
    .ndo_stop = snull_stop,
    .ndo_start_xmit = snull_tx,
    // .ndo_get_stats is deprecated, use dev->stats directly
};

static void snull_setup(struct net_device *dev) {
    ether_setup(dev); // Fill in Ethernet defaults (MTU, Header Len)
    dev->netdev_ops = &snull_netdev_ops;
    dev->flags |= IFF_NOARP; // Don't use ARP (It's virtual)
    
    // Random MAC
    eth_hw_addr_random(dev);
}
```

#### Step 5: Registration (in Init)
```c
struct net_device *my_dev;

static int __init snull_init(void) {
    // Allocate (sizeof(priv) is extra space)
    my_dev = alloc_etherdev(sizeof(struct snull_priv));
    if (!my_dev) return -ENOMEM;
    
    snull_setup(my_dev);
    strcpy(my_dev->name, "sn0");
    
    int ret = register_netdev(my_dev);
    if (ret) {
        free_netdev(my_dev);
        return ret;
    }
    
    return 0;
}

static void __exit snull_exit(void) {
    unregister_netdev(my_dev);
    free_netdev(my_dev);
}
```

---

## 🔬 Lab Exercise: Lab 170.1 - Testing the Interface

### 1. Lab Objectives
- Compile and load.
- Configure IP.
- Send packets (Ping).

### 2. Step-by-Step Guide
1.  **Load:** `insmod snull.ko`.
2.  **Check:** `ip link show sn0`.
3.  **Up:** `ip link set sn0 up`.
4.  **IP:** `ip addr add 192.168.10.1/24 dev sn0`.
5.  **Ping:** `ping -c 3 192.168.10.2`.
    *   **Result:** It will fail (Destination Host Unreachable) because we are just dropping packets in TX. But TX stats should increase.
6.  **Stats:** `ip -s link show sn0`.
    *   TX packets: 3.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Loopback Mode
- **Goal:** Make Ping work (locally).
- **Task:**
    1.  In `snull_tx`, instead of freeing, call `netif_rx(skb)`.
    2.  This feeds the packet back into the RX stack.
    3.  Ping 192.168.10.1 (Self) should work.

### Lab 3: Promiscuous Mode
- **Goal:** Handle `ndo_set_rx_mode`.
- **Task:**
    1.  Implement the callback.
    2.  Check `dev->flags & IFF_PROMISC`.
    3.  Print "Promiscuous Mode ON" to dmesg.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Device not found"
*   **Cause:** `register_netdev` failed.
*   **Cause:** `alloc_etherdev` failed.

#### 2. Kernel Panic on TX
*   **Cause:** Accessing `skb` after freeing it.
*   **Rule:** Once you call `dev_kfree_skb`, you don't own it anymore.
*   **Rule:** Once you call `netif_rx`, you don't own it anymore.

---

## ⚡ Optimization & Best Practices

### `NETDEV_TX_BUSY`
*   If your hardware FIFO is full, return `NETDEV_TX_BUSY`.
*   **Crucial:** You must call `netif_stop_queue(dev)` *before* returning BUSY, otherwise the kernel will loop infinitely trying to send.
*   When HW is ready, call `netif_wake_queue(dev)`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `alloc_etherdev`?
    *   **A:** A helper wrapper around `alloc_netdev`. It sets up the device as an Ethernet device (Type ARPHRD_ETHER, MTU 1500, etc.).
2.  **Q:** Why `dev_kfree_skb` instead of `kfree`?
    *   **A:** `sk_buff` is a complex structure with reference counting and caches. Always use the specialized free functions.

### Challenge Task
> **Task:** "The Packet Modifier".
> *   In `snull_tx`, modify the packet data before looping it back.
> *   Change the first byte of the payload to 'X'.
> *   Verify with `tcpdump -i sn0 -X`.

---

## 📚 Further Reading & References
- [Kernel Documentation: networking/driver.rst](https://www.kernel.org/doc/html/latest/networking/driver.html)
- [Linux Device Drivers, Chapter 17](https://lwn.net/Kernel/LDD3/)

---
