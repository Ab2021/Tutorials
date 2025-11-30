# Day 227: Bluetooth Subsystem (BlueZ & HCI)
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
1.  **Explain** the Bluetooth Stack (BlueZ, Kernel, Hardware).
2.  **Understand** HCI (Host Controller Interface) packets (Command, Event, ACL, SCO).
3.  **Use** `hciconfig`, `hcitool`, and `btmon`.
4.  **Visualize** the flow of a Bluetooth connection.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC with Bluetooth (USB dongle or built-in).
*   **Software Required:**
    *   `bluez`, `bluez-tools`.
*   **Prior Knowledge:**
    *   Day 176 (USB - since most BT is USB).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Stack
*   **Userspace (BlueZ):** `bluetoothd`. Handles pairing, profiles (A2DP, HFP), and DBus API.
*   **Kernel (L2CAP/RFCOMM/HCI):** Multiplexing, segmentation, and hardware abstraction.
*   **Hardware (Controller):** The radio chip. Talks HCI.

### 🔹 Part 2: HCI (Host Controller Interface)
*   **Standardized Protocol:** Unlike WiFi (where every vendor is different), Bluetooth Controllers speak a standard language (HCI).
*   **Packet Types:**
    1.  **Command (0x01):** Host -> Controller (e.g., "Start Scan").
    2.  **ACL Data (0x02):** Asynchronous Connection-Less (Normal Data).
    3.  **SCO Data (0x03):** Synchronous Connection-Oriented (Voice).
    4.  **Event (0x04):** Controller -> Host (e.g., "Scan Result").

---

## 💻 Implementation: Sniffing HCI

> **Instruction:** Use `btmon` to see the raw HCI traffic.

### 👨‍💻 Code Implementation

```bash
# This is a terminal exercise, not C code yet.
sudo btmon
```

---

## 🔬 Lab Exercise: Lab 227.1 - Manual HCI Commands

### 1. Lab Objectives
- Send raw commands using `hcitool`.

### 2. Step-by-Step Guide
1.  **Reset Device:**
    ```bash
    sudo hciconfig hci0 reset
    ```
2.  **Scan:**
    ```bash
    sudo hcitool scan
    ```
3.  **Raw Command (Read Local Version):**
    *   Opcode: OGF (0x04 - Info Param) | OCF (0x0001 - Read Local Version).
    *   `hcitool cmd 0x04 0x0001`
    *   Watch `btmon` to see the Command and the Event response.

---

## 🧪 Additional / Advanced Labs

### Lab 2: L2CAP Ping
- **Goal:** Test data plane.
- **Task:**
    *   `l2ping <BD_ADDR>`.
    *   Observe ACL packets in `btmon`.

### Lab 3: Virtual Controller
- **Goal:** Use `hci_vhci`.
- **Task:**
    *   Load `hci_vhci` module.
    *   Open `/dev/vhci`.
    *   Write HCI Event packets to it.
    *   Kernel thinks it's a real Bluetooth controller.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Operation not possible due to RF-kill"
*   **Cause:** Hardware switch or software block.
*   **Fix:** `rfkill list`, `rfkill unblock bluetooth`.

#### 2. Firmware Missing
*   **Cause:** Many BT chips (Intel, Broadcom) need firmware loaded via HCI before they work.
*   **Fix:** Check `dmesg | grep firmware`.

---

## ⚡ Optimization & Best Practices

### UART vs USB
*   USB BT uses standard USB endpoints.
*   UART BT (common in embedded) requires a "Line Discipline" (`hci_uart`) to attach the TTY to the Bluetooth stack (`btattach`).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is L2CAP?
    *   **A:** Logical Link Control and Adaptation Protocol. It sits on top of HCI and multiplexes multiple logical channels (sockets) onto a single HCI connection.
2.  **Q:** What is the difference between BR/EDR and LE?
    *   **A:** BR/EDR (Basic Rate/Enhanced Data Rate) is "Classic" Bluetooth (Audio, File Transfer). LE (Low Energy) is for IoT (Sensors). They use different PHYs but share the HCI.

### Challenge Task
> **Task:** "The Beacon Scanner".
> *   Write a Python script (using `bluepy` or raw sockets) that listens for LE Advertising packets.
> *   Print the UUIDs of nearby beacons.

---

## 📚 Further Reading & References
- [Bluetooth Core Specification (Vol 4: HCI)](https://www.bluetooth.com/specifications/specs/)
- [BlueZ Documentation](http://www.bluez.org/)

---
