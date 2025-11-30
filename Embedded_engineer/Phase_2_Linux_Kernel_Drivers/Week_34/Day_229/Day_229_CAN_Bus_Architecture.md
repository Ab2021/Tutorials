# Day 229: CAN Bus Architecture (SocketCAN)
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
1.  **Explain** the CAN protocol (ID, DLC, Data, Arbitration).
2.  **Understand** the SocketCAN architecture (Network Interface vs Char Device).
3.  **Use** `can-utils` (`candump`, `cansend`, `cangen`).
4.  **Setup** a Virtual CAN (`vcan`) interface.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   `can-utils`.
*   **Prior Knowledge:**
    *   Day 170 (Net Dev).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The CAN Protocol
*   **Broadcast:** Everyone hears everything.
*   **ID:** 11-bit (Standard) or 29-bit (Extended). Determines priority (Lower ID = Higher Priority).
*   **Payload:** 0 to 8 bytes (Classic CAN). Up to 64 bytes (CAN FD).
*   **No MAC Address:** Addressed by Content (Message ID), not Node ID.

### 🔹 Part 2: SocketCAN Philosophy
*   **Old Way:** Character devices (`/dev/can0`). Hard to share, non-standard API.
*   **Linux Way (SocketCAN):** Treat CAN like Ethernet.
    *   Interface: `can0`, `vcan0`.
    *   API: Berkeley Sockets (`socket(PF_CAN, SOCK_RAW, CAN_RAW)`).
    *   Tools: `ip link`, `ifconfig`.

---

## 💻 Implementation: Virtual CAN Setup

> **Instruction:** Set up a virtual CAN network for testing.

### 👨‍💻 Code Implementation

```bash
# 1. Load Module
sudo modprobe vcan

# 2. Add Interface
sudo ip link add dev vcan0 type vcan

# 3. Bring Up
sudo ip link set up vcan0

# 4. Verify
ip link show vcan0
```

---

## 🔬 Lab Exercise: Lab 229.1 - Sending & Receiving

### 1. Lab Objectives
- Use `can-utils` to communicate over `vcan0`.

### 2. Step-by-Step Guide
1.  **Terminal 1 (Sniffer):**
    ```bash
    candump vcan0
    ```
2.  **Terminal 2 (Sender):**
    ```bash
    # ID 123, Data [DE AD BE EF]
    cansend vcan0 123#DEADBEEF
    ```
3.  **Observe:**
    *   Terminal 1 shows: `vcan0  123   [4]  DE AD BE EF`

---

## 🧪 Additional / Advanced Labs

### Lab 2: CAN FD (Flexible Data-Rate)
- **Goal:** Send larger packets.
- **Task:**
    *   `ip link set vcan0 mtu 72`.
    *   `cansend vcan0 123##0DEADBEEF...` (Note double hash `##` for FD).

### Lab 3: C Programming (Raw Socket)
- **Goal:** Send CAN frame from C.
- **Task:**
    *   `socket(PF_CAN, SOCK_RAW, CAN_RAW)`.
    *   `bind` to `vcan0`.
    *   `write` a `struct can_frame`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Network is down"
*   **Cause:** Forgot `ip link set up`.
*   **Fix:** Bring it up.

#### 2. No traffic
*   **Cause:** Bitrate mismatch (on real hardware).
*   **Fix:** `ip link set can0 type can bitrate 500000`. (Virtual CAN ignores bitrate).

---

## ⚡ Optimization & Best Practices

### Filtering
*   Use kernel-level filters (`setsockopt(..., CAN_RAW_FILTER, ...)`) instead of filtering in userspace.
*   Saves CPU by not waking up the app for irrelevant IDs.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why use SocketCAN instead of a char driver?
    *   **A:** Multi-user access, standard API, integration with Linux networking stack (qdisc, routing), and tool support (`tcpdump`, `wireshark`).
2.  **Q:** What is the maximum data length in Classic CAN?
    *   **A:** 8 bytes.

### Challenge Task
> **Task:** "The Traffic Generator".
> *   Use `cangen vcan0 -g 10` to flood the bus.
> *   Write a Python script (using `python-can`) to calculate the bus load (frames per second).

---

## 📚 Further Reading & References
- [Kernel Documentation: networking/can.rst](https://www.kernel.org/doc/html/latest/networking/can.html)
- [can-utils GitHub](https://github.com/linux-can/can-utils)

---
