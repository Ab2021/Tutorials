# Day 64: Task Notifications & Low Power Tickless Idle
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
1.  **Explain** the advantages of Direct Task Notifications over Semaphores (Speed, RAM).
2.  **Implement** signaling using `xTaskNotifyGive` and `ulTaskNotifyTake`.
3.  **Configure** FreeRTOS for **Tickless Idle** mode to save power during idle periods.
4.  **Analyze** the trade-offs of Tickless Idle (Wakeup latency vs Power).
5.  **Debug** issues related to sleep mode interfering with debuggers.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Multimeter (Current Measurement).
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   FreeRTOS
*   **Prior Knowledge:**
    *   Day 60 (Semaphores)
    *   Day 50 (Low Power)
*   **Datasheets:**
    *   [FreeRTOS Task Notifications](https://www.freertos.org/RTOS-task-notifications.html)
    *   [FreeRTOS Low Power Support](https://www.freertos.org/low-power-tickless-rtos.html)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Task Notifications
A "Lightweight Semaphore" built into the Task Control Block (TCB).
*   **Faster:** No need to allocate a separate Semaphore object. No queue overhead.
*   **Less RAM:** Saves ~80 bytes per semaphore.
*   **Limitation:** Can only send *to* a specific task. Cannot be used for multiple tasks waiting on one event (Event Group is better there).

### 🔹 Part 2: Tickless Idle
In standard RTOS, the SysTick fires every 1ms. This wakes the CPU from Sleep every 1ms, preventing deep power saving.
*   **Tickless Mode:** If the Idle Task sees that the next task is due in 100ms, it:
    1.  Stops SysTick.
    2.  Configures a Low Power Timer (LPTIM or RTC) to wake up in 100ms.
    3.  Enters Stop Mode.
    4.  On wakeup, corrects the Tick Count.

---

## 💻 Implementation: Notification Signaling

> **Instruction:** Replace the Binary Semaphore from Day 60 (Button ISR) with a Task Notification.

### 👨‍💻 Code Implementation

#### Step 1: ISR
```c
// Handle to the task we want to wake
TaskHandle_t hHandlerTask;

void EXTI0_IRQHandler(void) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    
    if (EXTI->PR & 1) {
        EXTI->PR |= 1;
        
        // Send Notification
        vTaskNotifyGiveFromISR(hHandlerTask, &xHigherPriorityTaskWoken);
    }
    
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}
```

#### Step 2: Task
```c
void vTaskHandler(void *p) {
    while(1) {
        // Wait for Notification (Clear count on exit)
        // Returns current notification count
        uint32_t count = ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        
        if (count > 0) {
            printf("Button Pressed! Count: %lu\n", count);
        }
    }
}
```

---

## 💻 Implementation: Tickless Idle

> **Instruction:** Configure `FreeRTOSConfig.h` and implement the suppression function.

### 👨‍💻 Code Implementation

#### Step 1: Configuration
In `FreeRTOSConfig.h`:
```c
#define configUSE_TICKLESS_IDLE  1
// Expected idle time before sleeping (e.g., 2 ticks)
#define configEXPECTED_IDLE_TIME_BEFORE_SLEEP 2 
```

#### Step 2: The Macro (Default)
FreeRTOS provides a default `vPortSuppressTicksAndSleep` for Cortex-M. It uses the SysTick.
*   **Problem:** SysTick stops in Stop Mode (usually).
*   **Solution:** We need to implement a custom one using RTC or LPTIM if we want Stop Mode. For Sleep Mode (WFI), the default is fine.

#### Step 3: Custom Suppression (Conceptual)
```c
void vPortSuppressTicksAndSleep(TickType_t xExpectedIdleTime) {
    // 1. Calculate wakeup time
    // 2. Configure RTC Alarm
    // 3. Stop SysTick
    // 4. Enter Stop Mode
    __WFI();
    // 5. Re-enable SysTick
    // 6. Correct Tick Count (Add slept time)
}
```

---

## 🔬 Lab Exercise: Lab 64.1 - Power Comparison

### 1. Lab Objectives
- Measure current with and without Tickless Idle.

### 2. Step-by-Step Guide

#### Phase A: Standard RTOS
1.  Task: `vTaskDelay(1000)`.
2.  Idle Task runs `__WFI()`.
3.  **Observation:** SysTick wakes CPU every 1ms. Current ~15mA.

#### Phase B: Tickless Idle
1.  Enable `configUSE_TICKLESS_IDLE`.
2.  Task: `vTaskDelay(1000)`.
3.  **Observation:** CPU sleeps for 1000ms continuously. Current drops slightly (in Sleep mode) or significantly (if using Stop mode custom hook).

### 3. Verification
Use an oscilloscope on a GPIO toggled in the Tick Hook.
*   Standard: Toggles every 1ms.
*   Tickless: Toggles only when active.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Notification as Mailbox
- **Goal:** Send a 32-bit value.
- **Task:**
    1.  `xTaskNotify(hTask, value, eSetValueWithOverwrite)`.
    2.  Receiver: `xTaskNotifyWait(...)`.
    3.  Pass a sensor reading directly without a Queue.

### Lab 3: Abort Delay
- **Goal:** Wake a task sleeping in `vTaskDelay`.
- **Task:**
    1.  Task A: `vTaskDelay(10000)`.
    2.  Task B: `xTaskAbortDelay(hTaskA)`.
    3.  Task A wakes up immediately.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Time Drift
*   **Cause:** Tickless Idle implementation inaccurate.
*   **Result:** `vTaskDelay(1000)` actually takes 1050ms.
*   **Solution:** Use a high-precision timer (LPTIM) and compensate for wakeup latency.

#### 2. Debugger Loss
*   **Cause:** Entering Stop Mode kills the debug connection.
*   **Solution:** Use `DBGMCU_CR` to keep clocks on during debug, or disable Tickless Idle during development.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Use Notifications First:** Always prefer Task Notifications over Binary Semaphores for simple signaling. They are faster and use less RAM.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can I send a notification from an ISR?
    *   **A:** Yes, `vTaskNotifyGiveFromISR`.
2.  **Q:** What happens if I give a notification multiple times before it's taken?
    *   **A:** It counts up (Counting Semaphore behavior). `ulTaskNotifyTake(pdTRUE)` clears it to 0. `ulTaskNotifyTake(pdFALSE)` decrements it.

### Challenge Task
> **Task:** Implement "Smart Sleep". If the expected idle time is < 5ms, use Sleep Mode. If > 100ms, use Stop Mode (Custom Tickless).

---

## 📚 Further Reading & References
- [Low Power Support for Cortex-M](https://www.freertos.org/low-power-ARM-Cortex-RTOS.html)

---
