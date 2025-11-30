# Day 228: Writing a Bluetooth HCI Driver
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
1.  **Allocate** and **Register** an `hci_dev`.
2.  **Implement** `open`, `close`, `send`, and `flush` callbacks.
3.  **Feed** data from hardware to the stack (`hci_recv_frame`).
4.  **Create** a Virtual HCI Loopback driver.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
*   **Prior Knowledge:**
    *   Day 227 (Bluetooth Subsystem).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The `struct hci_dev`
*   Represents a Bluetooth Controller.
*   **Type:** `HCI_PRIMARY` (Real), `HCI_AMP` (High Speed).
*   **Bus:** `HCI_USB`, `HCI_UART`, `HCI_VIRTUAL`.
*   **Queues:** `cmd_q` (Commands), `rx_q` (Events/Data).

### 🔹 Part 2: Data Flow
*   **TX (Host -> Controller):** Kernel calls `hdev->send(skb)`. Driver sends `skb->data` to hardware.
*   **RX (Controller -> Host):** Driver receives data, allocates `skb`, sets `bt_cb(skb)->pkt_type`, calls `hci_recv_frame(hdev, skb)`.

---

## 💻 Implementation: The "MyHCI" Driver

> **Instruction:** Create a virtual HCI driver that loops commands back as events (minimal simulation).

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <net/bluetooth/bluetooth.h>
#include <net/bluetooth/hci_core.h>

static struct hci_dev *hdev;

static int my_hci_open(struct hci_dev *hdev) {
    set_bit(HCI_RUNNING, &hdev->flags);
    return 0;
}

static int my_hci_close(struct hci_dev *hdev) {
    clear_bit(HCI_RUNNING, &hdev->flags);
    return 0;
}

static int my_hci_flush(struct hci_dev *hdev) {
    return 0;
}

static int my_hci_send_frame(struct hci_dev *hdev, struct sk_buff *skb) {
    // In a real driver: Send to HW.
    // Here: Just print and free.
    
    int type = hci_skb_pkt_type(skb);
    pr_info("MyHCI: Sending packet type %d len %d\n", type, skb->len);
    
    // Simulate a Command Complete event for Reset (Opcode 0x0C03)
    // This is hard to fake without a full state machine.
    // For now, just drop it.
    
    kfree_skb(skb);
    return 0;
}

static int __init my_init(void) {
    hdev = hci_alloc_dev();
    if (!hdev) return -ENOMEM;
    
    hdev->bus = HCI_VIRTUAL;
    hdev->dev_type = HCI_PRIMARY;
    hdev->open = my_hci_open;
    hdev->close = my_hci_close;
    hdev->flush = my_hci_flush;
    hdev->send = my_hci_send_frame;
    
    if (hci_register_dev(hdev) < 0) {
        hci_free_dev(hdev);
        return -ENODEV;
    }
    
    return 0;
}

static void __exit my_exit(void) {
    hci_unregister_dev(hdev);
    hci_free_dev(hdev);
}

module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
```

---

## 🔬 Lab Exercise: Lab 228.1 - Verifying the Device

### 1. Lab Objectives
- Use `hciconfig` to see the new device.

### 2. Step-by-Step Guide
1.  **Load:** `insmod myhci.ko`.
2.  **Check:**
    ```bash
    hciconfig -a
    ```
    *   Should see `hciX` (Bus: Virtual).
3.  **Up:**
    ```bash
    hciconfig hciX up
    ```
    *   It might timeout because we aren't replying to the "Reset" command that the kernel sends on init.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Responding to Reset
- **Goal:** Make `hciconfig up` work.
- **Task:**
    *   In `send_frame`: Check if it's a Command (0x01) and Opcode is Reset (0x0C03).
    *   If so, construct an Event packet (0x04):
        *   Event Code: Command Complete (0x0E).
        *   Param: Num HCI Command Packets (1), Opcode (0x0C03), Status (0x00 - Success).
    *   Call `hci_recv_frame`.

### Lab 3: SPI Transport
- **Goal:** Sketch an SPI driver.
- **Task:**
    *   In `send_frame`: `spi_sync_transfer`.
    *   In IRQ handler: `spi_sync_transfer` (Read), then `hci_recv_frame`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Timeout on Init
*   **Cause:** Kernel sends commands (Reset, Read Local Version) and waits for events. If driver drops them, init fails.
*   **Fix:** Use `HCI_QUIRK_RAW_DEVICE` if you just want a raw channel without kernel management, or implement the state machine.

---

## ⚡ Optimization & Best Practices

### Skb Headroom
*   HCI drivers often need to prepend a 1-byte packet type before sending to UART/USB.
*   Ensure `skb` has headroom (`skb_reserve`).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `hci_skb_pkt_type(skb)`?
    *   **A:** A macro to access the packet type stored in `cb` (Control Buffer) of the skb. It's not in the data payload itself when passed from core to driver.
2.  **Q:** Why `HCI_VIRTUAL`?
    *   **A:** Used for testing or for devices that don't fit standard buses (like a pure software simulator).

### Challenge Task
> **Task:** "The Echo Dongle".
> *   Implement the Reset response (Lab 2).
> *   Implement the "Read Local Version" response.
> *   Get `hciconfig hciX up` to succeed and show "RUNNING".

---

## 📚 Further Reading & References
- [Kernel Source: drivers/bluetooth/btusb.c](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/bluetooth/btusb.c) (The reference USB driver).

---
