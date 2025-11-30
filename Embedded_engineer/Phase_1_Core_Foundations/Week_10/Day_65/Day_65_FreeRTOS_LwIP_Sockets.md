# Day 65: FreeRTOS + LwIP (Socket API)
## Phase 1: Core Embedded Engineering Foundations | Week 10: Advanced RTOS & IoT

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
1.  **Integrate** LwIP with FreeRTOS (`NO_SYS = 0`).
2.  **Explain** the role of the `tcpip_thread` and how it serializes network operations.
3.  **Implement** a TCP Client using the standard BSD Socket API (`socket`, `connect`, `send`).
4.  **Manage** multiple socket connections in separate tasks.
5.  **Tune** LwIP memory settings for RTOS usage.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board with Ethernet.
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   FreeRTOS + LwIP
    *   Netcat (PC).
*   **Prior Knowledge:**
    *   Day 48 (LwIP Raw)
    *   Day 57 (RTOS)
*   **Datasheets:**
    *   [LwIP Socket API](https://lwip.fandom.com/wiki/Socket_API)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Raw vs Netconn vs Sockets
*   **Raw API (Day 48):** No RTOS. Callback based. Fast, hard to use.
*   **Netconn API:** RTOS required. Sequential API (`netconn_recv`). Type-safe.
*   **Socket API:** RTOS required. Standard BSD style (`recv`). Wrapper around Netconn.
    *   **Pros:** Portable code (same code runs on Linux/Windows).
    *   **Cons:** Higher overhead (copying data).

### 🔹 Part 2: The TCPIP Thread
LwIP core is **NOT** thread-safe. You cannot call LwIP functions from multiple tasks simultaneously.
*   **Solution:** The `tcpip_thread`.
*   **Mechanism:** When you call `send()`, the data is put into a Mailbox. The `tcpip_thread` wakes up, reads the Mailbox, and performs the actual protocol work.
*   **Implication:** You must initialize LwIP using `tcpip_init()`, not `lwip_init()`.

---

## 💻 Implementation: TCP Client Task

> **Instruction:** Connect to PC (192.168.1.5) on Port 8080. Send "Hello".

### 👨‍💻 Code Implementation

#### Step 1: LwIP Init (RTOS)
```c
void LwIP_RTOS_Init(void) {
    tcpip_init(NULL, NULL);
    
    // Add Netif (same as Day 48 but inside tcpip_thread context usually)
    // Actually, netif_add can be called here if tcpip_init is done.
    netif_add(&gnetif, &ipaddr, &netmask, &gw, NULL, &ethernetif_init, &tcpip_input);
    
    netif_set_default(&gnetif);
    netif_set_up(&gnetif);
    
    // DHCP (Optional)
    dhcp_start(&gnetif);
}
```

#### Step 2: Client Task
```c
#include "lwip/sockets.h"

void vTaskClient(void *p) {
    int sock;
    struct sockaddr_in server_addr;
    char *msg = "Hello from RTOS\n";
    
    // Wait for IP (DHCP)
    while (gnetif.ip_addr.addr == 0) vTaskDelay(100);
    
    while(1) {
        // 1. Create Socket
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            printf("Socket Error\n");
            vTaskDelay(1000);
            continue;
        }
        
        // 2. Connect
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(8080);
        server_addr.sin_addr.s_addr = inet_addr("192.168.1.5");
        
        if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            printf("Connect Error\n");
            close(sock);
            vTaskDelay(1000);
            continue;
        }
        
        // 3. Send
        send(sock, msg, strlen(msg), 0);
        
        // 4. Receive Response
        char buf[64];
        int len = recv(sock, buf, sizeof(buf)-1, 0);
        if (len > 0) {
            buf[len] = 0;
            printf("Server: %s\n", buf);
        }
        
        // 5. Close
        close(sock);
        
        vTaskDelay(5000);
    }
}
```

---

## 🔬 Lab Exercise: Lab 65.1 - Chat Server

### 1. Lab Objectives
- Create a TCP Server on STM32.
- Accept connections from PC.
- Echo messages back.

### 2. Step-by-Step Guide

#### Phase A: Server Task
```c
void vTaskServer(void *p) {
    int listen_sock, client_sock;
    struct sockaddr_in addr, client_addr;
    socklen_t client_len;
    
    listen_sock = socket(AF_INET, SOCK_STREAM, 0);
    
    addr.sin_family = AF_INET;
    addr.sin_port = htons(7); // Echo Port
    addr.sin_addr.s_addr = INADDR_ANY;
    
    bind(listen_sock, (struct sockaddr*)&addr, sizeof(addr));
    listen(listen_sock, 5);
    
    while(1) {
        // Block until client connects
        client_sock = accept(listen_sock, (struct sockaddr*)&client_addr, &client_len);
        
        if (client_sock >= 0) {
            // Spawn a worker task or handle here
            // For simple echo, handle here
            char buf[128];
            int len;
            while ((len = recv(client_sock, buf, sizeof(buf), 0)) > 0) {
                send(client_sock, buf, len, 0);
            }
            close(client_sock);
        }
    }
}
```

#### Phase B: Test
1.  Run STM32.
2.  PC: `telnet 192.168.1.10 7`.
3.  Type "Test".
4.  See "Test" return.

### 3. Verification
If `accept` returns -1, check memory (heap). Each socket needs a Netconn struct and buffers.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Multi-Threaded Server
- **Goal:** Handle multiple clients.
- **Task:**
    1.  When `accept` returns a socket, `xTaskCreate` a new `vWorkerTask`.
    2.  Pass `client_sock` as parameter.
    3.  Worker handles the session, then closes socket and deletes itself.
    4.  **Note:** Watch out for Heap exhaustion! Limit max clients.

### Lab 3: UDP Broadcast
- **Goal:** Announce presence.
- **Task:**
    1.  Create UDP socket.
    2.  `setsockopt(SO_BROADCAST)`.
    3.  Send to `255.255.255.255`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Stack Overflow in LwIP Thread
*   **Cause:** `TCPIP_THREAD_STACKSIZE` too small in `lwipopts.h`.
*   **Solution:** Increase it (at least 1024 words).

#### 2. HardFault on `socket()`
*   **Cause:** `tcpip_init` not called or not finished.
*   **Solution:** Wait for LwIP init to complete (use a Semaphore signaled in the init callback).

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Zero Copy:** Sockets force a copy. If performance is critical (e.g., Camera Stream), use Netconn or Raw API even with RTOS.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why does `recv()` block?
    *   **A:** It waits on a Semaphore/Mailbox inside the Netconn layer until data arrives from the `tcpip_thread`.
2.  **Q:** Can I use `select()`?
    *   **A:** Yes, LwIP supports `select()` to wait on multiple sockets.

### Challenge Task
> **Task:** Implement a "HTTP Get". Connect to `google.com` (use IP), send `GET / HTTP/1.1\r\nHost: google.com\r\n\r\n`. Print the HTML response.

---

## 📚 Further Reading & References
- [LwIP Application Developers Manual](https://www.nongnu.org/lwip/2_1_x/index.html)

---
