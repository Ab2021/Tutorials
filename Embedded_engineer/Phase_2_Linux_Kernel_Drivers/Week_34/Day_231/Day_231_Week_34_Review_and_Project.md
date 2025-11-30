# Day 231: Week 34 Review and Project - The Universal Wireless Bridge
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
1.  **Synthesize** Week 34 concepts (WiFi, Bluetooth, CAN).
2.  **Architect** a multi-protocol bridge.
3.  **Implement** a userspace application that talks to multiple kernel subsystems.
4.  **Demonstrate** data flow from CAN to Bluetooth.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Week 34 tools (`can-utils`, `bluez`, `iw`).
*   **Prior Knowledge:**
    *   Week 34 Content.

---

## 🔄 Week 34 Review

### 1. WiFi (Day 225-226)
*   `mac80211` (SoftMAC) vs `cfg80211`.
*   Netlink configuration.

### 2. Bluetooth (Day 227-228)
*   HCI (Command/Event/ACL).
*   BlueZ stack.

### 3. CAN Bus (Day 229-230)
*   SocketCAN.
*   `can-dev` framework.

---

## 🛠️ Project: "CAN-over-Bluetooth" Bridge

### 📋 Project Requirements
1.  **Goal:** Tunnel CAN frames over a Bluetooth RFCOMM socket.
2.  **Components:**
    *   **Virtual CAN (`vcan0`):** Source of vehicle data.
    *   **Bluetooth Server:** Listens on RFCOMM Channel 1.
    *   **Bridge App:** Reads `vcan0`, encapsulates in JSON/Binary, sends to BT Client.
3.  **Scenario:** A mechanic walks up to a car with a tablet (BT Client) and reads engine RPM (CAN).

---

## 💻 Implementation: Step-by-Step Guide

### 🔹 Phase 1: The CAN Reader
```c
int s = socket(PF_CAN, SOCK_RAW, CAN_RAW);
// ... bind vcan0 ...
read(s, &frame, sizeof(frame));
```

### 🔹 Phase 2: The Bluetooth Server
```c
int bt_sock = socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM);
struct sockaddr_rc loc_addr = { 0 };
loc_addr.rc_family = AF_BLUETOOTH;
loc_addr.rc_bdaddr = *BDADDR_ANY;
loc_addr.rc_channel = 1;
bind(bt_sock, (struct sockaddr *)&loc_addr, sizeof(loc_addr));
listen(bt_sock, 1);
client = accept(bt_sock, ...);
```

### 🔹 Phase 3: The Bridge Loop
```c
while (1) {
    // 1. Read CAN
    nbytes = read(can_sock, &frame, sizeof(frame));
    
    // 2. Format
    sprintf(buf, "ID: %03X Data: %02X %02X...\n", frame.can_id, frame.data[0]...);
    
    // 3. Send BT
    write(client, buf, strlen(buf));
}
```

---

## 📈 Grading Rubric

| Criteria | Excellent (A) | Good (B) | Needs Improvement (C) |
| :--- | :--- | :--- | :--- |
| **Functionality** | Real-time streaming. | High latency. | Data corruption. |
| **Robustness** | Handles BT disconnects. | Crashes on disconnect. | Leaks sockets. |
| **Formatting** | Clean JSON/Text. | Raw binary dump. | Unreadable. |

---

## 🔮 Looking Ahead: Week 35
Next week, we start **Phase 3: Advanced Storage & Filesystems**.
We will cover **VFS Architecture**, **Writing a Filesystem**, and **Flash Storage (MTD)**.

---
