# Day 82: Network & Cloud Task Implementation
## Phase 1: Core Embedded Engineering Foundations | Week 12: Capstone Project Phase 1

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
1.  **Implement** a robust State Machine for the Network Task (DHCP -> DNS -> TLS -> MQTT).
2.  **Integrate** mbedTLS with LwIP to create secure sockets.
3.  **Develop** an MQTT Client wrapper that handles automatic reconnection and subscription resync.
4.  **Synchronize** local device state with the Cloud Shadow (AWS/Azure).
5.  **Handle** network errors gracefully without crashing the RTOS.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Ethernet Connection
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   AWS IoT Core Account (Certificates ready)
*   **Prior Knowledge:**
    *   Day 80 (Middleware)
    *   Day 68 (Cloud Integration)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Network State Machine
A linear "Connect" function is bad. If it blocks for 30s, the task is unresponsive.
We need a State Machine:
*   **NET_STATE_INIT:** Wait for PHY Link Up.
*   **NET_STATE_DHCP:** Wait for IP Address.
*   **NET_STATE_DNS:** Resolve `a3xyz.iot.us-east-1.amazonaws.com`.
*   **NET_STATE_TLS:** Perform SSL Handshake.
*   **NET_STATE_MQTT:** Send CONNECT packet.
*   **NET_STATE_RUNNING:** Publish/Subscribe loop.
*   **NET_STATE_ERROR:** Backoff timer, then retry.

### 🔹 Part 2: Secure Socket Wrapper
We need a unified API that wraps standard Sockets and mbedTLS.
*   `Net_Connect(host, port, secure)`
*   `Net_Send(data, len)`
*   `Net_Recv(buf, max_len)`
*   `Net_Close()`

---

## 💻 Implementation: app_net.c

> **Instruction:** Implement the Network Task and State Machine.

### 👨‍💻 Code Implementation

#### Step 1: Definitions & States
```c
#include "app_net.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include "mbedtls/ssl.h"
#include "mqtt_client.h" // Our wrapper

typedef enum {
    NET_STATE_INIT,
    NET_STATE_DHCP,
    NET_STATE_DNS,
    NET_STATE_TLS,
    NET_STATE_MQTT,
    NET_STATE_RUNNING,
    NET_STATE_ERROR
} NetState_t;

static NetState_t gState = NET_STATE_INIT;
static char *brokerHost = "example.iot.us-east-1.amazonaws.com";
static ip_addr_t brokerIp;
```

#### Step 2: The Network Task
```c
void vTaskNet(void *p) {
    // Init LwIP (Day 80)
    BSP_Net_Init();
    
    while(1) {
        switch(gState) {
            case NET_STATE_INIT:
                if (BSP_Net_IsLinkUp()) {
                    gState = NET_STATE_DHCP;
                }
                vTaskDelay(1000);
                break;
                
            case NET_STATE_DHCP:
                if (BSP_Net_GetIP() != 0) {
                    printf("IP Acquired!\n");
                    gState = NET_STATE_DNS;
                }
                vTaskDelay(500);
                break;
                
            case NET_STATE_DNS:
                if (Net_Resolve(brokerHost, &brokerIp) == 0) {
                    gState = NET_STATE_TLS;
                } else {
                    printf("DNS Failed. Retrying...\n");
                    vTaskDelay(5000);
                }
                break;
                
            case NET_STATE_TLS:
                if (Net_Secure_Connect(brokerIp, 8883) == 0) {
                    gState = NET_STATE_MQTT;
                } else {
                    gState = NET_STATE_ERROR;
                }
                break;
                
            case NET_STATE_MQTT:
                if (MQTT_Connect("STM32_Hub") == 0) {
                    MQTT_Subscribe("cmd/clap");
                    gState = NET_STATE_RUNNING;
                } else {
                    Net_Close();
                    gState = NET_STATE_ERROR;
                }
                break;
                
            case NET_STATE_RUNNING:
                MQTT_Process(); // Yield, KeepAlive
                
                // Check for Outgoing Messages
                NetMsg_t msg;
                if (xQueueReceive(hNetQueue, &msg, 10)) {
                    Handle_Net_Msg(&msg);
                }
                
                if (!MQTT_IsConnected()) {
                    gState = NET_STATE_ERROR;
                }
                break;
                
            case NET_STATE_ERROR:
                Net_Close();
                vTaskDelay(10000); // Backoff
                gState = NET_STATE_DNS; // Restart from DNS
                break;
        }
    }
}
```

#### Step 3: DNS Resolver
```c
int Net_Resolve(char *host, ip_addr_t *ip) {
    struct hostent *h = gethostbyname(host);
    if (h == NULL) return -1;
    
    ip->addr = ((struct in_addr*)h->h_addr)->s_addr;
    return 0;
}
```

#### Step 4: Secure Connect (Simplified)
```c
int Net_Secure_Connect(ip_addr_t ip, uint16_t port) {
    // 1. Create Socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    
    // 2. Connect TCP
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = ip.addr;
    
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        return -1;
    }
    
    // 3. Start mbedTLS Handshake (Day 67 logic)
    // ... mbedtls_ssl_handshake ...
    
    return 0; // Success
}
```

---

## 💻 Implementation: Shadow Sync

> **Instruction:** When connected, sync the "Clap Count" to the Cloud.

### 👨‍💻 Code Implementation

#### Step 1: Shadow Update
```c
void Send_Shadow_Update(int clapCount) {
    char json[128];
    snprintf(json, sizeof(json), 
        "{\"state\":{\"reported\":{\"claps\":%d}}}", clapCount);
    
    MQTT_Publish("$aws/things/STM32_Hub/shadow/update", json);
}
```

#### Step 2: Handle App Message
```c
void Handle_Net_Msg(NetMsg_t *msg) {
    switch(msg->type) {
        case NET_CMD_REPORT_CLAP:
            Send_Shadow_Update(msg->param);
            break;
    }
}
```

---

## 🔬 Lab Exercise: Lab 82.1 - Cloud Connection

### 1. Lab Objectives
- Configure AWS Endpoint and Certs.
- Flash and run.
- Verify connection in AWS Console.

### 2. Step-by-Step Guide

#### Phase A: Config
1.  Put `client-cert.pem.crt` and `private.pem.key` in `app_config.h` (as const strings).
2.  Set `BROKER_HOST`.

#### Phase B: Run
1.  Terminal: "IP Acquired... DNS OK... TLS Handshake... MQTT Connected".
2.  AWS Console: MQTT Test Client -> Subscribe to `#`.
3.  **Observation:** See Keep Alive packets or Shadow updates.

### 3. Verification
If TLS fails (-0x2700), check:
1.  Time (NTP). If board thinks it's 1970, certs fail.
2.  Root CA.
3.  Heap Size (mbedTLS needs ~30KB).

---

## 🧪 Additional / Advanced Labs

### Lab 2: NTP Time Sync
- **Goal:** Set RTC.
- **Task:**
    1.  Before TLS state, add `NET_STATE_NTP`.
    2.  Send UDP packet to `pool.ntp.org` (Port 123).
    3.  Parse response (Seconds since 1900).
    4.  Set STM32 RTC.

### Lab 3: OTA Trigger
- **Goal:** Remote Update.
- **Task:**
    1.  Subscribe to `cmd/update`.
    2.  If message received `{"url": "..."}`, spawn `vTaskOTA`.
    3.  `vTaskOTA` downloads file and writes to Flash (Day 69).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. HardFault in mbedTLS
*   **Cause:** Stack Overflow. `mbedtls_ssl_context` is huge.
*   **Solution:** Allocate context on Heap (`calloc`), not Stack. Or increase Task Stack to 8KB.

#### 2. DNS Fail
*   **Cause:** DHCP didn't provide DNS server?
*   **Solution:** Hardcode Google DNS (8.8.8.8) as fallback.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Non-Blocking:** The `MQTT_Process` loop should not block for long. Use `select()` with a short timeout (e.g., 100ms) to check for incoming data, then yield to other tasks.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we need a separate `NET_STATE_DNS`?
    *   **A:** `gethostbyname` can block for seconds. We want to explicitely manage this step and retry if needed, rather than burying it inside "Connect".
2.  **Q:** What happens if the Broker IP changes?
    *   **A:** The State Machine handles this. If connection drops (`NET_STATE_ERROR`), we go back to `NET_STATE_DNS` to resolve the hostname again.

### Challenge Task
> **Task:** Implement "Offline Queue". If `NET_STATE_ERROR`, don't drop Telemetry. Queue it in RAM (Ring Buffer). When `NET_STATE_RUNNING`, flush the queue to Cloud.

---

## 📚 Further Reading & References
- [AWS IoT Device Shadow REST API](https://docs.aws.amazon.com/iot/latest/developerguide/device-shadow-rest-api.html)

---
