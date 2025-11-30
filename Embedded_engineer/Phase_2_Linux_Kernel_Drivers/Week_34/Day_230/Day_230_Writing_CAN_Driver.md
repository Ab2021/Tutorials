# Day 230: Writing a CAN Controller Driver
## Phase 2: Linux Kernel & Device Drivers | Week 34: Wireless & Embedded Networking

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
1.  **Allocate** a CAN device using `alloc_candev`.
2.  **Register** the device with `register_candev`.
3.  **Implement** `ndo_start_xmit` for CAN frames.
4.  **Handle** Bit Timing calculations (`bittiming_const`).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Day 229 (SocketCAN).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The `can-dev` Framework
*   **Helper Library:** `drivers/net/can/dev.c`.
*   **Purpose:** Handles common tasks like Bit Timing calculation, Error State handling (Active/Passive/BusOff), and Statistics.
*   **Structure:** `struct can_priv` embedded in `net_device` private data.

### 🔹 Part 2: Bit Timing
*   **Complexity:** CAN requires precise timing (Quanta, Segments, Jump Width).
*   **Kernel:** User provides Bitrate (e.g., 500k). Kernel calculates register values based on `bittiming_const` provided by the driver (Clock freq, Max TQ, etc.).

---

## 💻 Implementation: The "MyCAN" Driver

> **Instruction:** Create a dummy CAN driver.

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <linux/can/dev.h>

struct my_can_priv {
    struct can_priv can; // Must be first
    // Hardware specific regs...
};

static const struct can_bittiming_const my_btc = {
    .name = "mycan",
    .tseg1_min = 1, .tseg1_max = 16,
    .tseg2_min = 1, .tseg2_max = 8,
    .sjw_max = 4,
    .brp_min = 1, .brp_max = 64,
    .brp_inc = 1,
};

static netdev_tx_t my_start_xmit(struct sk_buff *skb, struct net_device *dev) {
    if (can_dropped_invalid_skb(dev, skb))
        return NETDEV_TX_OK;

    // In real driver: Write ID, DLC, Data to registers.
    // Here: Loopback
    
    struct sk_buff *newskb = skb_clone(skb, GFP_ATOMIC);
    if (newskb) {
        newskb->dev = dev; // Loopback to same interface
        netif_rx(newskb);
    }
    
    dev_kfree_skb(skb);
    return NETDEV_TX_OK;
}

static const struct net_device_ops my_netdev_ops = {
    .ndo_open = my_open,
    .ndo_stop = my_stop,
    .ndo_start_xmit = my_start_xmit,
};

static int __init my_init(void) {
    struct net_device *dev;
    struct my_can_priv *priv;
    
    // 1. Alloc (Echo skb max = 4)
    dev = alloc_candev(sizeof(struct my_can_priv), 4);
    if (!dev) return -ENOMEM;
    
    priv = netdev_priv(dev);
    priv->can.bittiming_const = &my_btc;
    priv->can.clock.freq = 80000000; // 80 MHz
    
    dev->netdev_ops = &my_netdev_ops;
    strcpy(dev->name, "mycan%d");
    
    // 2. Register
    register_candev(dev);
    return 0;
}
// ... Exit ...
```

---

## 🔬 Lab Exercise: Lab 230.1 - Setting Bitrate

### 1. Lab Objectives
- Configure the bitrate.
- Bring up the interface.

### 2. Step-by-Step Guide
1.  **Load:** `insmod mycan.ko`.
2.  **Set Bitrate:**
    ```bash
    ip link set mycan0 type can bitrate 500000
    ```
    *   Kernel uses `my_btc` to calculate dividers.
3.  **Up:**
    ```bash
    ip link set up mycan0
    ```
4.  **Test:** `candump mycan0` & `cansend mycan0 ...`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Error Handling
- **Goal:** Simulate Bus Off.
- **Task:**
    *   Call `can_bus_off(dev)`.
    *   Observe state change in `ip -d link show mycan0`.
    *   Recover with `ip link set mycan0 type can restart-ms 100`.

### Lab 3: Echo Skb
- **Goal:** Local loopback of sent frames.
- **Task:**
    *   Use `can_put_echo_skb` in TX.
    *   Use `can_get_echo_skb` in TX Completion interrupt.
    *   This is how local apps see what was sent to the bus.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Bitrate not supported"
*   **Cause:** The requested bitrate cannot be achieved with the provided clock frequency and `bittiming_const` constraints.
*   **Fix:** Check your clock frequency and divider ranges.

---

## ⚡ Optimization & Best Practices

### NAPI
*   CAN traffic can be bursty (100% bus load).
*   Use NAPI for RX to avoid interrupt storms, just like Ethernet.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `alloc_candev`?
    *   **A:** A wrapper around `alloc_netdev` that sets up CAN-specific private data (`struct can_priv`) and default operations.
2.  **Q:** Why do we need `can_put_echo_skb`?
    *   **A:** CAN is a broadcast bus. The sender should see its own message if it was successfully arbitrated on the bus (Loopback). The driver manages this queue.

### Challenge Task
> **Task:** "The Error Injector".
> *   Add a debugfs file.
> *   When written to, generate a CAN Error Frame (`CAN_ERR_FLAG`).
> *   Send it up to the stack.
> *   Verify `candump` shows the error.

---

## 📚 Further Reading & References
- [Kernel Source: drivers/net/can/mcp251x.c](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/net/can/spi/mcp251x.c) (Popular SPI CAN controller).

---
