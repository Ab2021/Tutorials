# Day 114: HAL & BSP Implementation
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
1.  **Implement** a robust Board Support Package (BSP) layer to abstract hardware details.
2.  **Develop** `bsp_wifi.c` to handle ESP8266 AT commands with timeouts and error checking.
3.  **Develop** `bsp_sensor.c` to aggregate data from internal ADC and external I2C sensors.
4.  **Develop** `bsp_bt.c` for interrupt-driven Bluetooth command reception.
5.  **Verify** individual BSP modules using unit tests.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   ESP8266, HC-05, Sensors.
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 113 (Architecture)
    *   Day 100 (UART Modules)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The BSP Philosophy
The Application layer should never call `HAL_UART_Transmit`.
*   **Bad:** `HAL_UART_Transmit(&huart2, "AT", 2, 100);` (App knows about UART handle and AT commands).
*   **Good:** `BSP_WiFi_IsReady();` (App only knows intent).
*   **Why?** If we change ESP8266 to an SPI-based WiFi module, we only rewrite `bsp_wifi.c`. The App code remains untouched.

### 🔹 Part 2: Module Interfaces
*   **WiFi:** `Init`, `Connect`, `Send`, `Receive`, `GetStatus`.
*   **BT:** `Init`, `Send`, `RegisterCallback`.
*   **Sensor:** `Init`, `ReadTemp`, `ReadLight`.

---

## 💻 Implementation: WiFi BSP (`bsp_wifi.c`)

> **Instruction:** Wrap the AT Command Parser (Day 100) into a clean API.

### 👨‍💻 Code Implementation

#### Step 1: Header (`bsp_wifi.h`)
```c
#ifndef BSP_WIFI_H
#define BSP_WIFI_H

#include <stdint.h>
#include <stdbool.h>

bool BSP_WiFi_Init(void);
bool BSP_WiFi_ConnectAP(const char *ssid, const char *pass);
bool BSP_WiFi_ConnectTCP(const char *ip, uint16_t port);
bool BSP_WiFi_Send(const uint8_t *data, uint16_t len);
int  BSP_WiFi_Receive(uint8_t *buf, uint16_t max_len);

#endif
```

#### Step 2: Source (`bsp_wifi.c`)
```c
#include "bsp_wifi.h"
#include "usart.h" // HAL Handle

// Internal AT Helper (Private)
static bool SendAT(const char *cmd, const char *expect, uint32_t timeout) {
    // ... Implementation from Day 100 ...
}

bool BSP_WiFi_Init(void) {
    // Reset Module
    HAL_GPIO_WritePin(ESP_RST_PORT, ESP_RST_PIN, 0);
    HAL_Delay(10);
    HAL_GPIO_WritePin(ESP_RST_PORT, ESP_RST_PIN, 1);
    HAL_Delay(500);
    
    return SendAT("AT", "OK", 1000);
}

bool BSP_WiFi_ConnectAP(const char *ssid, const char *pass) {
    char cmd[128];
    snprintf(cmd, 128, "AT+CWJAP=\"%s\",\"%s\"", ssid, pass);
    return SendAT(cmd, "OK", 10000); // Long timeout for WiFi Join
}
```

---

## 💻 Implementation: Sensor BSP (`bsp_sensor.c`)

> **Instruction:** Aggregate Internal Temp (ADC) and Light Sensor (ADC or I2C).

### 👨‍💻 Code Implementation

#### Step 1: Header (`bsp_sensor.h`)
```c
typedef struct {
    float temp_c;
    float light_lux;
    float vbat_v;
} SensorData_t;

void BSP_Sensor_Init(void);
void BSP_Sensor_ReadAll(SensorData_t *data);
```

#### Step 2: Source (`bsp_sensor.c`)
```c
#include "bsp_sensor.h"
#include "adc.h"

void BSP_Sensor_Init(void) {
    MX_ADC1_Init();
    // MX_I2C1_Init(); // If using external sensor
}

void BSP_Sensor_ReadAll(SensorData_t *data) {
    // 1. Read Internal Temp
    HAL_ADC_Start(&hadc1);
    HAL_ADC_PollForConversion(&hadc1, 10);
    uint32_t raw_temp = HAL_ADC_GetValue(&hadc1);
    
    // Convert (Formula from Datasheet)
    data->temp_c = ((raw_temp * 3.3 / 4096.0) - 0.76) / 0.0025 + 25.0;
    
    // 2. Read Light (Simulated on Potentiometer or Photoresistor)
    // Switch Channel...
    data->light_lux = 500.0f; // Dummy
    
    // 3. Vbat
    data->vbat_v = 3.3f;
}
```

---

## 💻 Implementation: Bluetooth BSP (`bsp_bt.c`)

> **Instruction:** Interrupt-driven command receiver.

### 👨‍💻 Code Implementation

#### Step 1: Callback Type
```c
typedef void (*BT_CmdCallback_t)(char *cmd);
static BT_CmdCallback_t app_cb = NULL;
```

#### Step 2: ISR Handling
```c
void BSP_BT_Init(BT_CmdCallback_t cb) {
    app_cb = cb;
    // Start RX Interrupt
    HAL_UART_Receive_IT(&huart2, &rx_byte, 1);
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart->Instance == USART2) {
        static char buf[32];
        static int idx = 0;
        
        if (rx_byte == '\n') {
            buf[idx] = 0; // Null terminate
            if (app_cb) app_cb(buf); // Notify App
            idx = 0;
        } else {
            if (idx < 31) buf[idx++] = rx_byte;
        }
        HAL_UART_Receive_IT(&huart2, &rx_byte, 1);
    }
}
```

---

## 🔬 Lab Exercise: Lab 114.1 - BSP Unit Tests

### 1. Lab Objectives
- Verify each BSP module independently.
- Ensure hardware connections are correct.

### 2. Step-by-Step Guide

#### Phase A: Test WiFi
```c
void Test_WiFi(void) {
    if (BSP_WiFi_Init()) printf("WiFi Init OK\n");
    else printf("WiFi Init FAIL\n");
    
    if (BSP_WiFi_ConnectAP("SSID", "PASS")) printf("WiFi Connect OK\n");
    else printf("WiFi Connect FAIL\n");
}
```

#### Phase B: Test Sensors
```c
void Test_Sensors(void) {
    SensorData_t d;
    BSP_Sensor_ReadAll(&d);
    printf("Temp: %.2f, Light: %.2f\n", d.temp_c, d.light_lux);
}
```

#### Phase C: Test BT
```c
void My_BT_Callback(char *cmd) {
    printf("BT Received: %s\n", cmd);
}

void Test_BT(void) {
    BSP_BT_Init(My_BT_Callback);
    while(1); // Wait for phone command
}
```

### 3. Verification
Run each test function in `main()`. Don't try to run the full app yet. Isolate hardware issues now.

---

## 🧪 Additional / Advanced Labs

### Lab 2: SD Card BSP
- **Goal:** Abstract FatFs.
- **Task:**
    1.  `BSP_SD_Init()`: Mounts FS.
    2.  `BSP_SD_Log(char *msg)`: Appends to `log.txt`.
    3.  Handle "Card Missing" gracefully.

### Lab 3: Watchdog Integration
- **Goal:** Reliability.
- **Task:**
    1.  `BSP_WDT_Init(timeout_ms)`.
    2.  `BSP_WDT_Kick()`.
    3.  Call Kick in the main loop.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. WiFi Timeout
*   **Cause:** Power supply insufficient for ESP8266 (needs 200mA).
*   **Solution:** External 3.3V supply.

#### 2. Sensor Noise
*   **Cause:** ADC jitter.
*   **Solution:** Software averaging in `BSP_Sensor_ReadAll`. Take 10 samples, average them.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Error Codes:** Instead of `bool`, return `int` (0=OK, -1=Timeout, -2=Error).
- **Non-Blocking:** `BSP_WiFi_Connect` currently blocks for 10s. In a real OS, this would be a state machine. For now, it's acceptable during Init, but bad during Loop.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why use `static` for `SendAT`?
    *   **A:** To limit its scope to `bsp_wifi.c`. It's a private helper function, not part of the public API.
2.  **Q:** What happens if `app_cb` is NULL in the ISR?
    *   **A:** Crash (HardFault) if we call it. Always check `if (app_cb)` before calling.

### Challenge Task
> **Task:** "Async WiFi". Rewrite `BSP_WiFi_Connect` to be non-blocking. It should return `BUSY` immediately, and we poll `BSP_WiFi_GetStatus()` until it returns `CONNECTED` or `ERROR`.

---

## 📚 Further Reading & References
- [Layered Architecture Pattern](https://www.oreilly.com/library/view/software-architecture-patterns/9781491971437/ch01.html)

---
