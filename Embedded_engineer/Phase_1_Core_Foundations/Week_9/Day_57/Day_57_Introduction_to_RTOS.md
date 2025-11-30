# Day 57: Introduction to RTOS & FreeRTOS
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
1.  **Contrast** Super-Loop architecture with Real-Time Operating Systems (RTOS).
2.  **Explain** the concept of Preemptive Multitasking and Context Switching.
3.  **Configure** FreeRTOS for the STM32F4 (Heap, Tick Rate, Priorities).
4.  **Create** and **Start** multiple FreeRTOS Tasks (`xTaskCreate`).
5.  **Debug** Stack Overflow issues in an RTOS environment.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   [FreeRTOS Kernel Source](https://www.freertos.org/a00104.html) (v10.x or later).
*   **Prior Knowledge:**
    *   Day 12 (Exception Handling - PendSV/SysTick)
    *   Day 9 (Stack Pointer MSP/PSP)
*   **Datasheets:**
    *   [FreeRTOS Reference Manual](https://www.freertos.org/Documentation/RTOS_book.html)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why RTOS?
*   **Super Loop:** `while(1) { TaskA(); TaskB(); }`. Simple, but if TaskA blocks (delay), TaskB waits. Determinism is hard.
*   **RTOS:** A Scheduler runs in the background (SysTick). It pauses TaskA and runs TaskB based on **Priority** and **Time Slicing**.
*   **Determinism:** High priority tasks *always* run when ready.

### 🔹 Part 2: Context Switching
How does the CPU switch from Task A to Task B?
1.  **SysTick Interrupt:** Fires every 1ms.
2.  **Save Context:** Push R0-R15, PSR of Task A to its Stack (PSP).
3.  **Select Next Task:** Scheduler picks Task B.
4.  **Restore Context:** Pop R0-R15, PSR from Task B's Stack.
5.  **Return:** CPU resumes Task B exactly where it left off.
*   **PendSV:** The actual switch happens in the PendSV (Pendable Service Call) handler to avoid delaying higher priority interrupts.

### 🔹 Part 3: FreeRTOS Config
`FreeRTOSConfig.h` controls the kernel.
*   `configUSE_PREEMPTION`: 1 (Enable preemption).
*   `configTICK_RATE_HZ`: 1000 (1ms tick).
*   `configTOTAL_HEAP_SIZE`: 10240 (10KB Heap for tasks).
*   `configMAX_PRIORITIES`: 5.

---

## 💻 Implementation: First RTOS Project

> **Instruction:** Create two tasks. Task 1 blinks LED1 (500ms). Task 2 blinks LED2 (200ms).

### 🛠️ Hardware/System Configuration
*   **LED1:** PD12 (Green).
*   **LED2:** PD13 (Orange).

### 👨‍💻 Code Implementation

#### Step 1: Includes & Handles
```c
#include "stm32f4xx.h"
#include "FreeRTOS.h"
#include "task.h"

TaskHandle_t hTask1 = NULL;
TaskHandle_t hTask2 = NULL;
```

#### Step 2: Task Functions
**Rule:** Tasks must never return. They must be infinite loops.
```c
void vTask1_Handler(void *params) {
    while(1) {
        GPIOD->ODR ^= (1 << 12); // Toggle Green
        vTaskDelay(pdMS_TO_TICKS(500)); // Blocking Delay
    }
}

void vTask2_Handler(void *params) {
    while(1) {
        GPIOD->ODR ^= (1 << 13); // Toggle Orange
        vTaskDelay(pdMS_TO_TICKS(200));
    }
}
```

#### Step 3: Main
```c
int main(void) {
    HAL_Init();
    SystemClock_Config();
    
    // Init GPIO
    RCC->AHB1ENR |= (1 << 3);
    GPIOD->MODER |= (1 << 24) | (1 << 26);
    
    // Create Tasks
    // Stack Depth is in WORDS (4 bytes), not Bytes!
    xTaskCreate(vTask1_Handler, "Task1", 128, NULL, 1, &hTask1);
    xTaskCreate(vTask2_Handler, "Task2", 128, NULL, 1, &hTask2);
    
    // Start Scheduler
    vTaskStartScheduler();
    
    // Should never reach here
    while(1);
}
```

---

## 🔬 Lab Exercise: Lab 57.1 - Preemption Test

### 1. Lab Objectives
- Verify that a higher priority task interrupts a lower priority one.

### 2. Step-by-Step Guide

#### Phase A: Setup
1.  Task 1 (Low Priority): Turns LED On, Busy Wait (Loop) for 1s, Turns LED Off.
2.  Task 2 (High Priority): Toggles another LED every 100ms.

#### Phase B: Observation
1.  If Preemption is ON: Task 2 will blink *while* Task 1 is busy waiting.
2.  If Preemption is OFF (`configUSE_PREEMPTION 0`): Task 2 will wait until Task 1 yields (calls `vTaskDelay`).

### 3. Verification
With Preemption (Default), the system feels "multitasking".

---

## 🧪 Additional / Advanced Labs

### Lab 2: Stack Overflow
- **Goal:** Crash the system.
- **Task:**
    1.  Create a task with small stack (e.g., 64 words).
    2.  Allocate a huge array `int arr[100]` inside the task function.
    3.  **Result:** HardFault or Stack Overflow Hook.
    4.  **Fix:** Enable `configCHECK_FOR_STACK_OVERFLOW 2`. Implement `vApplicationStackOverflowHook`.

### Lab 3: Idle Task
- **Goal:** Measure CPU Load.
- **Task:**
    1.  Implement `vApplicationIdleHook`.
    2.  Toggle a GPIO pin in the hook.
    3.  Measure duty cycle with Logic Analyzer. High duty cycle = Low CPU Load.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. HardFault on StartScheduler
*   **Cause:** Interrupts not configured correctly. FreeRTOS needs SysTick, PendSV, and SVC.
*   **Solution:** Ensure `xPortPendSVHandler`, `xPortSysTickHandler`, `vPortSVCHandler` are mapped to the vector table (usually in `stm32f4xx_it.c` or via `#define` in `FreeRTOSConfig.h`).

#### 2. Tasks don't run
*   **Cause:** Heap too small. `xTaskCreate` returns `pdFAIL`.
*   **Solution:** Check return value. Increase `configTOTAL_HEAP_SIZE`.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Static Allocation:** Use `xTaskCreateStatic` to avoid Heap fragmentation and ensure deterministic memory usage. Requires `configSUPPORT_STATIC_ALLOCATION 1`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `HAL_Delay` and `vTaskDelay`?
    *   **A:** `HAL_Delay` blocks the CPU (busy wait). `vTaskDelay` blocks the *Task* (yields CPU to others). Always use `vTaskDelay` in tasks.
2.  **Q:** What is the Idle Task?
    *   **A:** A task created automatically by the scheduler with Priority 0. It runs when no other task is ready. Used for cleanup and power saving.

### Challenge Task
> **Task:** Implement "Task Statistics". Use `vTaskGetRunTimeStats` to print the CPU usage of each task to UART every 5 seconds.

---

## 📚 Further Reading & References
- [Mastering the FreeRTOS Real Time Kernel (Official Book)](https://www.freertos.org/Documentation/RTOS_book.html)

---
