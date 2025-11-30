# Day 83: Application Logic & State Machine
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
1.  **Design** the central Application Task (`vTaskApp`) that coordinates Audio and Network tasks.
2.  **Implement** a hierarchical State Machine (HSM) or Flat State Machine for system control.
3.  **Handle** asynchronous events (Clap Detected, Cloud Command, Button Press) via a unified Event Queue.
4.  **Manage** persistent settings (Volume, WiFi Creds) using Flash Emulation.
5.  **Create** a CLI (Command Line Interface) for debugging and control.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 81 (Audio)
    *   Day 82 (Network)
    *   Day 54 (Flash)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Coordinator Pattern
We have two "Worker" tasks (Audio, Net) and one "Coordinator" (App).
*   **Audio Task:** "I heard a clap." (Event -> App).
*   **Net Task:** "Cloud says turn on light." (Event -> App).
*   **App Task:** Decides what to do.
    *   If Clap -> Tell Net to send MQTT. Tell Audio to play "Beep".
    *   If Cloud Cmd -> Tell GPIO to toggle LED.

### 🔹 Part 2: Event Queue
A single queue `hAppQueue` receives messages from all sources.
```c
typedef enum {
    EVT_BUTTON_PRESS,
    EVT_CLAP_DETECTED,
    EVT_CLOUD_CMD,
    EVT_TIMER_TICK
} AppEventType_t;

typedef struct {
    AppEventType_t type;
    uint32_t param;
} AppMsg_t;
```

### 🔹 Part 3: System States
*   **SYS_BOOT:** Init hardware.
*   **SYS_NORMAL:** Running.
*   **SYS_MUTE:** Audio disabled.
*   **SYS_ERROR:** Hardware failure.
*   **SYS_OTA:** Updating firmware.

---

## 💻 Implementation: app_main.c

> **Instruction:** Implement the Application Task and Event Loop.

### 👨‍💻 Code Implementation

#### Step 1: Definitions
```c
#include "app_main.h"
#include "app_audio.h"
#include "app_net.h"
#include "bsp_gpio.h"

QueueHandle_t hAppQueue;

typedef enum {
    SYS_BOOT,
    SYS_NORMAL,
    SYS_MUTE,
    SYS_OTA
} SysState_t;

static SysState_t gSysState = SYS_BOOT;
static int clapCount = 0;
```

#### Step 2: The App Task
```c
void vTaskApp(void *p) {
    hAppQueue = xQueueCreate(20, sizeof(AppMsg_t));
    
    // Init Logic
    printf("System Booting...\n");
    gSysState = SYS_NORMAL;
    
    AppMsg_t msg;
    
    while(1) {
        // Block waiting for event
        if (xQueueReceive(hAppQueue, &msg, portMAX_DELAY)) {
            Process_App_Event(&msg);
        }
    }
}
```

#### Step 3: Event Processor
```c
void Process_App_Event(AppMsg_t *msg) {
    switch(msg->type) {
        case EVT_CLAP_DETECTED:
            if (gSysState == SYS_NORMAL) {
                printf("Clap! Count: %d\n", ++clapCount);
                
                // 1. Feedback Sound
                AudioMsg_t audioMsg = { .type = AUDIO_CMD_PLAY_TONE, .param = 1000 };
                xQueueSend(hAudioQueue, &audioMsg, 0);
                
                // 2. Update Cloud
                NetMsg_t netMsg = { .type = NET_CMD_REPORT_CLAP, .param = clapCount };
                xQueueSend(hNetQueue, &netMsg, 0);
                
                // 3. Visual
                BSP_LED_Toggle(LED_GREEN);
            }
            break;
            
        case EVT_BUTTON_PRESS:
            // Toggle Mute
            if (gSysState == SYS_NORMAL) {
                gSysState = SYS_MUTE;
                printf("System Muted\n");
                BSP_LED_On(LED_ORANGE);
            } else if (gSysState == SYS_MUTE) {
                gSysState = SYS_NORMAL;
                printf("System Normal\n");
                BSP_LED_Off(LED_ORANGE);
            }
            break;
            
        case EVT_CLOUD_CMD:
            // Param 1 = ON, 0 = OFF
            if (msg->param) BSP_LED_On(LED_BLUE);
            else BSP_LED_Off(LED_BLUE);
            break;
    }
}
```

---

## 💻 Implementation: CLI Task

> **Instruction:** A simple UART shell for debugging.

### 👨‍💻 Code Implementation

#### Step 1: CLI Task
```c
void vTaskCLI(void *p) {
    char rxBuf[64];
    int idx = 0;
    
    printf("Shell Ready > ");
    
    while(1) {
        char c;
        if (BSP_UART_GetChar(&c)) { // Non-blocking check
            if (c == '\r' || c == '\n') {
                rxBuf[idx] = 0;
                printf("\n");
                Process_CLI_Cmd(rxBuf);
                idx = 0;
                printf("> ");
            } else {
                rxBuf[idx++] = c;
                printf("%c", c); // Echo
                if (idx >= 63) idx = 0;
            }
        }
        vTaskDelay(10);
    }
}
```

#### Step 2: Command Parser
```c
void Process_CLI_Cmd(char *cmd) {
    if (strcmp(cmd, "status") == 0) {
        printf("State: %d, Claps: %d\n", gSysState, clapCount);
        printf("Heap: %d bytes\n", xPortGetFreeHeapSize());
    } 
    else if (strcmp(cmd, "reboot") == 0) {
        NVIC_SystemReset();
    }
    else if (strncmp(cmd, "vol ", 4) == 0) {
        int vol = atoi(cmd + 4);
        AudioMsg_t msg = { .type = AUDIO_CMD_SET_VOL, .param = vol };
        xQueueSend(hAudioQueue, &msg, 0);
    }
    else {
        printf("Unknown command\n");
    }
}
```

---

## 🔬 Lab Exercise: Lab 83.1 - Integration Test

### 1. Lab Objectives
- Verify Event Flow: Clap -> App -> Audio/Net.
- Verify State Change: Button -> Mute.
- Verify CLI.

### 2. Step-by-Step Guide

#### Phase A: Clap Test
1.  Clap hands.
2.  **Observation:**
    *   Green LED Toggles.
    *   Speaker Beeps.
    *   UART: "Clap! Count: X".
    *   AWS Console: Shadow updates.

#### Phase B: Mute Test
1.  Press User Button.
2.  **Observation:** Orange LED ON. UART: "System Muted".
3.  Clap hands.
4.  **Observation:** Nothing happens (Event ignored in Mute state).

#### Phase C: CLI Test
1.  Type `status`.
2.  **Observation:** See system stats.
3.  Type `vol 80`.
4.  **Observation:** Volume changes.

### 3. Verification
If CLI is unresponsive, check UART baud rate (115200) and if `vTaskCLI` has enough stack (printf uses stack!).

---

## 🧪 Additional / Advanced Labs

### Lab 2: Persistent Settings
- **Goal:** Save Volume/Mute state.
- **Task:**
    1.  On change, write to Flash (Sector 11).
    2.  On Boot, read from Flash.
    3.  Use a "Magic Number" to detect if Flash is initialized.

### Lab 3: Watchdog Integration
- **Goal:** System Health.
- **Task:**
    1.  App Task must "Kick" the IWDG every 1s.
    2.  If App Task hangs (e.g., Queue full), system resets.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Queue Full
*   **Cause:** Events arriving faster than App can process.
*   **Solution:** Increase Queue size. Optimize `Process_App_Event` (remove printf).

#### 2. Race Conditions
*   **Cause:** Accessing global `clapCount` from CLI and App Task?
*   **Solution:** `clapCount` is static in `app_main.c`. CLI should request status via Queue, or use a Mutex if sharing variables.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Command Pattern:** The CLI parser can be improved using a table of function pointers `{"cmd", handler_func}` instead of `if-else` chains.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why use a single Event Queue for the App Task?
    *   **A:** Serialization. It ensures the App Task processes one event at a time, avoiding race conditions without needing complex Mutex locking on every variable.
2.  **Q:** What is the priority of the App Task?
    *   **A:** Lower than Audio (Real-time) and Net (Throughput), but higher than CLI/Idle. It's the "Manager".

### Challenge Task
> **Task:** Implement "Macro Actions". Define a sequence of actions (e.g., "Party Mode": Vol 100, Blink LEDs, Send MQTT). Trigger it via CLI or Long Press.

---

## 📚 Further Reading & References
- [Event-Driven Architecture in Embedded Systems](https://www.embedded.com/event-driven-programming-for-embedded-systems/)

---
