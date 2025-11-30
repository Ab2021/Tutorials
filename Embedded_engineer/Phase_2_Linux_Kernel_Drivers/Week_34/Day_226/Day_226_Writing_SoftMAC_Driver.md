# Day 226: Writing a SoftMAC WiFi Driver
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
1.  **Allocate** a `mac80211` hardware structure (`ieee80211_alloc_hw`).
2.  **Implement** the `start_xmit` callback for raw 802.11 frames.
3.  **Handle** RX path (`mac80211` receive).
4.  **Simulate** a wireless link (Loopback).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Day 225 (Wireless Subsystem).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The `mac80211` Ops
*   **`ops->start()`:** Bring up hardware.
*   **`ops->stop()`:** Shut down.
*   **`ops->tx()`:** Transmit a frame.
*   **`ops->config()`:** Change channel, power, etc.

### 🔹 Part 2: Frame Flow
*   **TX:** Stack -> `mac80211` (Adds 802.11 header, encrypts) -> Driver `tx()` -> Hardware.
*   **RX:** Hardware -> Driver -> `mac80211` (Decrypts, strips header) -> Stack (Ethernet).

---

## 💻 Implementation: The "SoftWiFi" Driver

> **Instruction:** Create a driver that allocates a `mac80211` hw and registers it.

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <net/mac80211.h>

struct my_priv {
    struct ieee80211_hw *hw;
};

static int my_start(struct ieee80211_hw *hw) {
    pr_info("MyWiFi: Start\n");
    return 0;
}

static void my_stop(struct ieee80211_hw *hw) {
    pr_info("MyWiFi: Stop\n");
}

static int my_tx(struct ieee80211_hw *hw, struct sk_buff *skb) {
    // In a real driver: DMA map and send to HW.
    // Here: Loopback for testing (Simulate RX)
    
    // We must consume the skb
    dev_kfree_skb(skb);
    return 0;
}

static struct ieee80211_ops my_ops = {
    .start = my_start,
    .stop = my_stop,
    .tx = my_tx,
};

static int __init my_init(void) {
    struct ieee80211_hw *hw;
    struct my_priv *priv;
    int ret;

    // 1. Alloc HW
    hw = ieee80211_alloc_hw(sizeof(struct my_priv), &my_ops);
    if (!hw) return -ENOMEM;
    
    priv = hw->priv;
    priv->hw = hw;
    
    // 2. Setup Bands (Required!)
    // ... (Omitted for brevity, but must populate hw->wiphy->bands) ...
    
    // 3. Register
    ret = ieee80211_register_hw(hw);
    if (ret) {
        ieee80211_free_hw(hw);
        return ret;
    }
    
    return 0;
}
// ... Exit ...
```

---

## 🔬 Lab Exercise: Lab 226.1 - Bringing Interface Up

### 1. Lab Objectives
- Use `ip link`.
- Observe `start` callback.

### 2. Step-by-Step Guide
1.  **Load:** `insmod mywifi.ko`.
2.  **Find Interface:** `ip link` (usually `wlan0` or `wlan1`).
3.  **Up:**
    ```bash
    ip link set wlan0 up
    ```
4.  **Dmesg:** Should see "MyWiFi: Start".

---

## 🧪 Additional / Advanced Labs

### Lab 2: RX Simulation
- **Goal:** Feed packets to stack.
- **Task:**
    *   In `my_tx`, instead of freeing, call `ieee80211_rx(hw, skb, rx_status)`.
    *   This creates a loopback.
    *   Note: You need to set `rx_status` (signal strength, rate) correctly.

### Lab 3: Monitor Mode
- **Goal:** Sniff packets.
- **Task:**
    *   `iw dev wlan0 set type monitor`.
    *   `ip link set wlan0 up`.
    *   `tcpdump -i wlan0`.
    *   If your `tx` function loops back, you should see your own packets.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "No supported bands"
*   **Cause:** `ieee80211_register_hw` fails if bands are not populated.
*   **Fix:** You MUST fill `hw->wiphy->bands[NL80211_BAND_2GHZ]`.

#### 2. Kernel Panic on TX
*   **Cause:** Not freeing `skb` in `tx` callback.
*   **Fix:** The driver owns the skb once `tx` is called. Free it or pass it to HW.

---

## ⚡ Optimization & Best Practices

### Scatter-Gather
*   Wifi frames can be large (A-MPDU). Use SG to avoid copying.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `A-MPDU`?
    *   **A:** Aggregated Mac Protocol Data Unit. Grouping multiple frames into one large burst to reduce overhead. `mac80211` handles the aggregation logic.
2.  **Q:** Why `ieee80211_alloc_hw`?
    *   **A:** It allocates both the `struct ieee80211_hw` and the underlying `struct wiphy` (cfg80211) in one go, linking them together.

### Challenge Task
> **Task:** "The Beacon Generator".
> *   Implement a timer in the driver.
> *   Every 100ms, construct a raw 802.11 Beacon Frame.
> *   Pass it to `ieee80211_rx`.
> *   Use `iw scan` on *another* interface to see if it detects your fake AP.

---

## 📚 Further Reading & References
- [mac80211 Driver Developer's Guide](https://wireless.wiki.kernel.org/en/developers/documentation/mac80211)

---
