# Day 115: Middleware Integration (MQTT & FatFs)
## Phase 1: Core Embedded Engineering Foundations | Week 17: Final Project - The Smart Home Hub

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
1.  **Integrate** the MQTT Protocol (Day 103) into the project middleware layer.
2.  **Implement** a `Logger` module using FatFs (Day 93) to persist system events.
3.  **Secure** communication using AES encryption (Day 104) within the MQTT payload.
4.  **Manage** state persistence (saving configuration to Flash/SD).
5.  **Design** a robust "Retry & Backoff" mechanism for network connections.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   ESP8266, SD Card.
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 114 (BSP)
    *   Day 103 (MQTT)
    *   Day 93 (FatFs)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Middleware Architecture
Middleware sits between BSP and App. It doesn't know about hardware pins (BSP does), and it doesn't know about business logic (App does).
*   **MQTT Client:** Knows how to packetize CONNECT/PUBLISH. Uses `BSP_WiFi_Send`.
*   **Logger:** Knows how to format strings and call `f_write`. Uses `BSP_SD_Init`.
*   **Config Manager:** Knows struct layout. Uses `lfs_read`/`f_read`.

### 🔹 Part 2: Robustness Strategies
*   **Exponential Backoff:** If MQTT fails, wait 1s, then 2s, 4s, 8s... to avoid flooding the network.
*   **Circular Logging:** If SD card is full, delete oldest file or wrap around.
*   **Atomic Saves:** Write to `config.tmp`, then rename to `config.dat`.

---

## 💻 Implementation: MQTT Middleware (`mqtt_mgr.c`)

> **Instruction:** Manage the connection state and packet building.

### 👨‍💻 Code Implementation

#### Step 1: Header (`mqtt_mgr.h`)
```c
#ifndef MQTT_MGR_H
#define MQTT_MGR_H

#include <stdbool.h>

void MQTT_Mgr_Init(void);
bool MQTT_Mgr_Connect(const char *client_id);
bool MQTT_Mgr_Publish(const char *topic, const char *msg);
void MQTT_Mgr_Process(void); // Call periodically

#endif
```

#### Step 2: Source (`mqtt_mgr.c`)
```c
#include "mqtt_mgr.h"
#include "bsp_wifi.h"
#include "aes.h" // Day 104

static bool is_connected = false;

bool MQTT_Mgr_Connect(const char *client_id) {
    if (!BSP_WiFi_IsConnected()) return false;
    
    // Build Packet (Day 103 code)
    uint8_t buf[128];
    int len = MQTT_BuildConnect(buf, client_id);
    
    if (BSP_WiFi_Send(buf, len)) {
        // Wait for CONNACK (Simplified)
        uint8_t rx[4];
        if (BSP_WiFi_Receive(rx, 4) == 4) {
            if (rx[0] == 0x20 && rx[3] == 0x00) {
                is_connected = true;
                return true;
            }
        }
    }
    return false;
}

bool MQTT_Mgr_Publish(const char *topic, const char *msg) {
    if (!is_connected) return false;
    
    // 1. Encrypt Payload
    uint8_t cipher[64];
    uint8_t iv[16];
    RNG_Get(iv);
    AES_Encrypt(msg, cipher, iv, SECRET_KEY);
    
    // 2. Build Packet (Topic + IV + Cipher)
    uint8_t buf[256];
    // ... Build logic ...
    
    return BSP_WiFi_Send(buf, total_len);
}
```

---

## 💻 Implementation: Logger Middleware (`logger.c`)

> **Instruction:** Thread-safe(ish) logging to SD Card.

### 👨‍💻 Code Implementation

#### Step 1: Header (`logger.h`)
```c
typedef enum { LOG_INFO, LOG_WARN, LOG_ERROR } LogLevel_t;

void Logger_Init(void);
void Logger_Log(LogLevel_t level, const char *fmt, ...);
```

#### Step 2: Source (`logger.c`)
```c
#include "logger.h"
#include "fatfs.h"
#include <stdarg.h>
#include <stdio.h>

static FIL logFile;
static bool fs_ok = false;

void Logger_Init(void) {
    if (f_mount(&SDFatFS, "", 1) == FR_OK) {
        // Open/Create log.txt
        if (f_open(&logFile, "log.txt", FA_OPEN_ALWAYS | FA_WRITE) == FR_OK) {
            f_lseek(&logFile, f_size(&logFile)); // Append
            fs_ok = true;
        }
    }
}

void Logger_Log(LogLevel_t level, const char *fmt, ...) {
    if (!fs_ok) return;
    
    char buf[128];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, 128, fmt, args);
    va_end(args);
    
    // Add Timestamp & Level
    char final_buf[160];
    const char *lvl_str = (level == LOG_ERROR) ? "ERR" : "INF";
    int len = snprintf(final_buf, 160, "[%lu] %s: %s\n", HAL_GetTick(), lvl_str, buf);
    
    UINT bw;
    f_write(&logFile, final_buf, len, &bw);
    f_sync(&logFile); // Flush to disk
}
```

---

## 🔬 Lab Exercise: Lab 115.1 - Integration Test

### 1. Lab Objectives
- Initialize Logger.
- Connect MQTT.
- Log connection status to SD Card.

### 2. Step-by-Step Guide

#### Phase A: Test Code
```c
void Test_Middleware(void) {
    Logger_Init();
    Logger_Log(LOG_INFO, "System Boot");
    
    if (MQTT_Mgr_Connect("STM32_Hub")) {
        Logger_Log(LOG_INFO, "MQTT Connected");
        MQTT_Mgr_Publish("home/status", "Online");
    } else {
        Logger_Log(LOG_ERROR, "MQTT Fail");
    }
}
```

#### Phase B: Run
1.  Insert SD Card.
2.  Ensure WiFi credentials are correct in BSP.
3.  Run.
4.  **Verification:** Check SD Card `log.txt` on PC. Check MQTT Broker for "Online" message (encrypted).

### 3. Verification
If `f_open` fails, check SD Card formatting (FAT32). If MQTT fails, check WiFi connection first.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Config Manager
- **Goal:** Load WiFi SSID from SD Card.
- **Task:**
    1.  Create `config.ini` on SD: `SSID=MyWiFi\nPASS=Secret`.
    2.  Write `Config_Load()` to parse this file.
    3.  Pass these values to `BSP_WiFi_ConnectAP`.

### Lab 3: Offline Buffering
- **Goal:** No data loss.
- **Task:**
    1.  If MQTT Publish fails, write message to `queue.dat` on SD.
    2.  In `MQTT_Mgr_Process`, if connected, read `queue.dat` and publish.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Stack Overflow
*   **Cause:** `vsnprintf` and large buffers in `Logger_Log`.
*   **Solution:** Increase Stack Size in linker script (0x400 -> 0x800 or 0x1000).

#### 2. SD Card Corruption
*   **Cause:** Removing card while writing.
*   **Solution:** Use `f_sync` often, but be aware of latency. Or add a "Eject" button that calls `f_close`.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Dependency Injection:** Pass the `WiFi_Send` function pointer to `MQTT_Init`. This allows unit testing MQTT logic without real WiFi hardware (Mocking).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why use `f_sync`?
    *   **A:** `f_write` only writes to RAM buffer. `f_sync` flushes it to the physical SD card. Essential to prevent data loss on power cut.
2.  **Q:** What is `va_list`?
    *   **A:** Variable Argument List. Allows creating functions like `printf` that take variable number of arguments.

### Challenge Task
> **Task:** "Secure Bootloader". (Advanced). Use the SD Card to load a new firmware binary (`app.bin`). Verify its HMAC signature before flashing it to the Application region.

---

## 📚 Further Reading & References
- [FatFs Module Application Note](http://elm-chan.org/fsw/ff/00index_e.html)
- [MQTT Essentials: Packet Structure](https://www.hivemq.com/blog/mqtt-essentials-part-2-publish-subscribe/)

---
