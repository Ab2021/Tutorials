# Day 48: LwIP Stack Basics (TCP/IP)
## Phase 1: Core Embedded Engineering Foundations | Week 7: Advanced Peripherals

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
1.  **Explain** the TCP/IP layers (Link, Network, Transport, Application).
2.  **Integrate** the LwIP (Lightweight IP) stack into an STM32 project.
3.  **Configure** LwIP for NO_SYS mode (Raw API, no RTOS).
4.  **Implement** a TCP Echo Server that replies to client messages.
5.  **Understand** the role of `ethernetif.c` (The glue between MAC and LwIP).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Board with Ethernet (RMII).
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   [LwIP Source Code](https://git.savannah.nongnu.org/git/lwip.git)
    *   Netcat (nc) or Telnet Client
*   **Prior Knowledge:**
    *   Day 47 (Ethernet MAC)
*   **Datasheets:**
    *   [LwIP Wiki](https://lwip.fandom.com/wiki/LwIP_Wiki)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: LwIP Architecture
LwIP is designed for embedded systems with limited RAM.
*   **APIs:**
    *   **Raw API:** Callback-based. Fastest, lowest memory. No RTOS required. (We use this).
    *   **Netconn API:** Sequential. Requires RTOS.
    *   **Socket API:** BSD-like. Requires RTOS.
*   **Pbufs:** Packet Buffers. A chain of RAM blocks holding the packet data. This avoids copying data between layers.

### 🔹 Part 2: The Glue (ethernetif.c)
LwIP is hardware-agnostic. We must provide:
*   `low_level_init()`: Setup MAC/DMA.
*   `low_level_output()`: Send pbuf to MAC DMA.
*   `low_level_input()`: Read MAC DMA to pbuf.

---

## 💻 Implementation: TCP Echo Server

> **Instruction:** We will setup a static IP (192.168.1.10) and listen on Port 7.

### 👨‍💻 Code Implementation

#### Step 1: LwIP Initialization (`main.c`)

```c
#include "lwip/init.h"
#include "lwip/netif.h"
#include "lwip/tcp.h"
#include "netif/etharp.h"

struct netif gnetif;

void LwIP_Init(void) {
    ip_addr_t ipaddr, netmask, gw;
    
    // Static IP: 192.168.1.10
    IP4_ADDR(&ipaddr, 192, 168, 1, 10);
    IP4_ADDR(&netmask, 255, 255, 255, 0);
    IP4_ADDR(&gw, 192, 168, 1, 1);
    
    lwip_init();
    
    // Add Network Interface
    // ethernetif_init is defined in the glue file provided by ST or written by us
    netif_add(&gnetif, &ipaddr, &netmask, &gw, NULL, &ethernetif_init, &ethernet_input);
    
    // Set as Default and Up
    netif_set_default(&gnetif);
    netif_set_up(&gnetif);
}
```

#### Step 2: TCP Callbacks
```c
err_t tcp_recv_callback(void *arg, struct tcp_pcb *tpcb, struct pbuf *p, err_t err) {
    if (p == NULL) {
        // Connection closed by remote
        tcp_close(tpcb);
        return ERR_OK;
    }
    
    // Echo data back
    tcp_write(tpcb, p->payload, p->len, 1);
    
    // Acknowledge processing
    tcp_recved(tpcb, p->tot_len);
    
    // Free buffer
    pbuf_free(p);
    
    return ERR_OK;
}

err_t tcp_accept_callback(void *arg, struct tcp_pcb *newpcb, err_t err) {
    // Set Recv Callback for this new connection
    tcp_recv(newpcb, tcp_recv_callback);
    return ERR_OK;
}

void TCP_Echo_Init(void) {
    struct tcp_pcb *pcb = tcp_new();
    tcp_bind(pcb, IP_ADDR_ANY, 7); // Port 7
    pcb = tcp_listen(pcb);
    tcp_accept(pcb, tcp_accept_callback);
}
```

#### Step 3: Main Loop (Polling)
Since we are NO_SYS, we must poll LwIP periodically.
```c
int main(void) {
    HAL_Init(); // Or custom init
    SystemClock_Config();
    
    LwIP_Init();
    TCP_Echo_Init();
    
    while(1) {
        // Check for RX packets
        ethernetif_input(&gnetif);
        
        // Handle Timers (TCP retransmission, ARP, etc.)
        sys_check_timeouts();
    }
}
```

---

## 🔬 Lab Exercise: Lab 48.1 - Ping & Echo

### 1. Lab Objectives
- Verify network connectivity.
- Test TCP Echo.

### 2. Step-by-Step Guide

#### Phase A: Ping
1.  Connect Board to PC (Direct or via Switch).
2.  Set PC IP to 192.168.1.5.
3.  Cmd: `ping 192.168.1.10`.
4.  **Success:** "Reply from 192.168.1.10: bytes=32 time<1ms".

#### Phase B: Echo
1.  Cmd: `telnet 192.168.1.10 7` (or use Putty Raw mode).
2.  Type "Hello".
3.  **Success:** You see "Hello" appear back.

### 3. Verification
If Ping fails, check Link LED. Check IP subnet. Check Windows Firewall.

---

## 🧪 Additional / Advanced Labs

### Lab 2: DHCP Client
- **Goal:** Get IP automatically.
- **Task:**
    1.  Enable `LWIP_DHCP` in `lwipopts.h`.
    2.  Call `dhcp_start(&gnetif)`.
    3.  In main loop, check `gnetif.ip_addr`. Print it when assigned.

### Lab 3: Web Server (HTTP)
- **Goal:** Serve a simple HTML page.
- **Task:**
    1.  Listen on Port 80.
    2.  On Recv, check for "GET /".
    3.  Send "HTTP/1.1 200 OK\r\n\r\n<html><body><h1>Hello STM32</h1></body></html>".

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. HardFault in `pbuf_free`
*   **Cause:** Double free, or memory corruption.
*   **Solution:** Ensure you don't free `p` if you passed it to `tcp_write` with copy=0 (Reference). But usually `tcp_write` copies data, so you MUST free `p` in the recv callback.

#### 2. Low Throughput
*   **Cause:** Small TCP Window.
*   **Solution:** Increase `TCP_WND` in `lwipopts.h`.

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **Zero-Copy:** In `low_level_output`, point the DMA descriptors directly to the pbuf payload (if aligned) instead of copying to a TX buffer. This is tricky (chained pbufs) but fast.

### Code Quality
- **lwipopts.h:** This is the most important file. Tune memory sizes (`MEM_SIZE`, `PBUF_POOL_SIZE`) to fit your RAM.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is ARP?
    *   **A:** Address Resolution Protocol. Maps IP (192.168.1.10) to MAC (00:80:E1:...). LwIP handles this automatically.
2.  **Q:** Why use Raw API instead of Sockets?
    *   **A:** Sockets require a thread per connection (blocking I/O), consuming huge RAM for stacks. Raw API is event-driven and runs on a single thread.

### Challenge Task
> **Task:** Implement a "UDP Broadcast Listener". Listen on Port 5000. If a packet "DISCOVER" is received, reply with "STM32 HERE". This is how devices are found on a LAN.

---

## 📚 Further Reading & References
- [LwIP Raw API Tutorial](https://lwip.fandom.com/wiki/Raw/TCP)

---
