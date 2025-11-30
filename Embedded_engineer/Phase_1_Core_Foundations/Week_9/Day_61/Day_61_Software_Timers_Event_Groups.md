# Day 61: Software Timers & Event Groups
## Phase 1: Core Embedded Engineering Foundations | Week 9: RTOS Fundamentals

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
1.  **Contrast** Hardware Timers (TIMx) with FreeRTOS Software Timers.
2.  **Implement** One-Shot and Auto-Reload Software Timers.
3.  **Use** Event Groups to synchronize multiple tasks (Wait for All / Wait for Any).
4.  **Understand** the Timer Service Task (Daemon Task) and its priority.
5.  **Design** a state machine using Event Bits.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   FreeRTOS
*   **Prior Knowledge:**
    *   Day 58 (Tasks)
*   **Datasheets:**
    *   [FreeRTOS Software Timer API](https://www.freertos.org/FreeRTOS-Software-Timer-API-Functions.html)
    *   [FreeRTOS Event Group API](https://www.freertos.org/FreeRTOS-Event-Groups.html)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Software Timers
Hardware timers are limited (e.g., 14 on STM32F4). Software timers are unlimited (limited only by RAM).
*   **Mechanism:** Managed by the **Timer Service Task** (Daemon). It wakes up when the nearest timer expires.
*   **Callback:** When a timer expires, it calls a function.
    *   **Important:** The callback runs in the context of the Timer Task. **Do not block** (delay) in a callback!
*   **Types:**
    *   **One-Shot:** Runs once, then stops. (e.g., "Turn off backlight after 10s").
    *   **Auto-Reload:** Repeats periodically. (e.g., "Blink LED every 500ms").

### 🔹 Part 2: Event Groups
A set of bits (flags). Tasks can wait for specific bits to be set.
*   **Wait for All (AND):** Wait for Bit 0 AND Bit 1. (e.g., "Wait for WiFi Connected AND MQTT Connected").
*   **Wait for Any (OR):** Wait for Bit 0 OR Bit 1. (e.g., "Wait for Stop Button OR Error Flag").
*   **Memory:** Uses less RAM than multiple Binary Semaphores.

---

## 💻 Implementation: Traffic Light (Timers)

> **Instruction:**
> *   Green (5s) -> Yellow (2s) -> Red (5s) -> Green...
> *   Use One-Shot timers to trigger the transitions.

### 👨‍💻 Code Implementation

#### Step 1: Handles
```c
TimerHandle_t hTimerGreen, hTimerYellow, hTimerRed;
```

#### Step 2: Callbacks
```c
void vGreenCallback(TimerHandle_t xTimer) {
    // Green Done -> Turn Yellow
    LED_Green_Off();
    LED_Yellow_On();
    xTimerStart(hTimerYellow, 0);
}

void vYellowCallback(TimerHandle_t xTimer) {
    // Yellow Done -> Turn Red
    LED_Yellow_Off();
    LED_Red_On();
    xTimerStart(hTimerRed, 0);
}

void vRedCallback(TimerHandle_t xTimer) {
    // Red Done -> Turn Green
    LED_Red_Off();
    LED_Green_On();
    xTimerStart(hTimerGreen, 0);
}
```

#### Step 3: Setup
```c
void Traffic_Light_Init(void) {
    hTimerGreen = xTimerCreate("Green", pdMS_TO_TICKS(5000), pdFALSE, NULL, vGreenCallback);
    hTimerYellow = xTimerCreate("Yel", pdMS_TO_TICKS(2000), pdFALSE, NULL, vYellowCallback);
    hTimerRed = xTimerCreate("Red", pdMS_TO_TICKS(5000), pdFALSE, NULL, vRedCallback);
    
    // Start Sequence
    LED_Green_On();
    xTimerStart(hTimerGreen, 0);
}
```

---

## 💻 Implementation: System Check (Event Groups)

> **Instruction:**
> *   Task 1: Checks SD Card (takes 1s). Sets Bit 0.
> *   Task 2: Checks Network (takes 2s). Sets Bit 1.
> *   Task 3: Checks Sensors (takes 500ms). Sets Bit 2.
> *   Task Main: Waits for ALL bits (0, 1, 2) before starting the application.

### 👨‍💻 Code Implementation

#### Step 1: Definitions
```c
EventGroupHandle_t hSystemEvents;
#define BIT_SD      (1 << 0)
#define BIT_NET     (1 << 1)
#define BIT_SENS    (1 << 2)
#define ALL_BITS    (BIT_SD | BIT_NET | BIT_SENS)
```

#### Step 2: Check Tasks
```c
void vTaskSD(void *p) {
    // ... Check SD ...
    vTaskDelay(1000);
    xEventGroupSetBits(hSystemEvents, BIT_SD);
    vTaskDelete(NULL);
}
// ... Similar for Net and Sens ...
```

#### Step 3: Main Task
```c
void vTaskMain(void *p) {
    printf("Booting...\n");
    
    // Wait for all bits. Clear them on exit. Wait forever.
    EventBits_t bits = xEventGroupWaitBits(
        hSystemEvents,
        ALL_BITS,
        pdTRUE,  // Clear on exit
        pdTRUE,  // Wait for ALL (AND)
        portMAX_DELAY
    );
    
    if ((bits & ALL_BITS) == ALL_BITS) {
        printf("All Systems Go!\n");
        // Start App
    }
}
```

---

## 🔬 Lab Exercise: Lab 61.1 - Debounce Timer

### 1. Lab Objectives
- Use a Software Timer to debounce a button.

### 2. Step-by-Step Guide

#### Phase A: Logic
1.  ISR: Button Press. `xTimerResetFromISR(hDebounceTimer)`.
2.  If button bounces (multiple interrupts), the timer keeps getting reset.
3.  Timer expires only when button is stable for X ms (e.g., 50ms).
4.  Callback: `vButtonAction`.

#### Phase B: Implementation
```c
void EXTI0_IRQHandler(void) {
    // ... Clear Flag ...
    xTimerResetFromISR(hDebounceTimer, &xHigherPriorityTaskWoken);
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

void vDebounceCallback(TimerHandle_t xTimer) {
    // Check if button is still pressed (optional)
    if (Button_Read()) {
        printf("Button Pressed (Debounced)\n");
    }
}
```

### 3. Verification
Press button. Even with noisy contacts, you get only one print.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Watchdog Timer (Software)
- **Goal:** Monitor tasks.
- **Task:**
    1.  Create a One-Shot Timer (5s).
    2.  Task A must call `xTimerReset` every loop.
    3.  If Task A hangs, Timer expires -> Callback -> Reset System.

### Lab 3: Event Group Sync
- **Goal:** Rendezvous.
- **Task:**
    1.  3 Tasks perform different jobs.
    2.  They must all reach a "Sync Point" before any proceeds.
    3.  Use `xEventGroupSync`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Timer Callback Hangs
*   **Cause:** Calling `vTaskDelay` or blocking API in callback.
*   **Result:** The Timer Service Task blocks. No other timers run.
*   **Solution:** Keep callbacks short. Send a message to a Queue if work is needed.

#### 2. Event Group from ISR
*   **Cause:** Using `xEventGroupSetBits` instead of `xEventGroupSetBitsFromISR`.
*   **Note:** `xEventGroupSetBitsFromISR` actually sends a message to the Timer Task to perform the set operation (deferred). So the Timer Task priority matters!

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Timer Task Priority:** `configTIMER_TASK_PRIORITY`. Should be high enough to process commands quickly, but maybe lower than critical real-time tasks. Default is usually high (max - 1).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the resolution of a Software Timer?
    *   **A:** The RTOS Tick (usually 1ms). You cannot have a 10µs software timer.
2.  **Q:** How many bits are in an Event Group?
    *   **A:** 24 bits (on 32-bit systems with `configUSE_16_BIT_TICKS` set to 0). The top 8 bits are reserved by kernel.

### Challenge Task
> **Task:** Implement a "Morse Code Blinker". Create a Queue of durations (dots/dashes). A Timer Callback reads the Queue. If "Dot", turn LED on, change Timer Period to DotTime. Next callback, turn LED off, change Period to GapTime.

---

## 📚 Further Reading & References
- [FreeRTOS Software Timers](https://www.freertos.org/RTOS-software-timer.html)

---
