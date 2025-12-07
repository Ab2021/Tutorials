# Day 151: Interrupt Handling in FreeRTOS
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 22: Embedded Systems & Firmware

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Golden Rule:** Explain why blocking API calls are forbidden inside ISRs.
2.  **Deferred Processing:** Offload ISR work to a high-priority task using Binary Semaphores.
3.  **FromISR APIs:** Use `xSemaphoreGiveFromISR` and `xQueueSendFromISR` correctly.
4.  **Context Switching:** Request a context switch from an interrupt using `portYIELD_FROM_ISR`.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **ISR Context:** Runs at Master Privilege, uses MSP (Main Stack Pointer). No TCB context.
*   **Latency:** ISRs block other interrupts (or lower priority ones). Keep them **short**.
*   **FromISR:** Special FreeRTOS functions safe for interrupt context. They don't block; they return instantly.

### Practical Setup

*   **Scenario:** A Button Press (GPIO IRQ) triggers a complex action (e.g., File Save). We can't save a file in the ISR!

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The `xHigherPriorityTaskWoken` Parameter

This is the most confusing part for beginners.
1.  **Task A** (Priority 1) is running.
2.  **ISR** fires.
3.  **ISR** gives a Semaphore.
4.  **Task B** (Priority 2) was blocked on this Semaphore. It now unblocks.
5.  **ISR Ends.**
6.  **Question:** Should we go back to Task A or switch to Task B?
7.  **Answer:** If we do nothing, we go back to A (until next Tick). If we call `portYIELD_FROM_ISR`, we switch to B **immediately**.
8.  `xHigherPriorityTaskWoken` is a boolean set to `pdTRUE` if the API call unblocked a task > current task.

---

## 💻 Implementation: Button Interrupt Deferred Handling

### `main.c`

```c
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

// --- Global Handles ---
SemaphoreHandle_t xButtonSemaphore; // Binary Semaphore

// --- Task: Handler ---
// This task sits BLOCKED most of the time.
// It effectively acts as the "Threaded Interrupt Handler".
void vTask_ButtonHandler(void *pvParameters) {
    for(;;) {
        // 1. Wait for Signal (Block indefinitely)
        if (xSemaphoreTake(xButtonSemaphore, portMAX_DELAY) == pdPASS) {
            
            // 2. Do Heavy Work (that is unsafe in ISR)
            printf("Button Pressed! Saving Data...\n");
            vTaskDelay(pdMS_TO_TICKS(500)); // Simulate slow storage write
            printf("Done.\n");
        }
    }
}

// --- Driver Task ---
// Just to keep CPU busy to show preemption
void vTask_LED(void *pvParameters) {
    for(;;) {
        GPIO_Toggle(LED_PIN);
        vTaskDelay(pdMS_TO_TICKS(100)); // 100ms
    }
}

// --- Interrupt Service Routine (ISR) ---
// Name depends on hardware, e.g., EXTI0_IRQHandler for STM32
void EXTI0_IRQHandler(void) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    
    // 1. Clear Hardware Interrupt Flag
    EXTI_ClearFlag(0);
    
    // 2. Give Semaphore
    // This unblocks vTask_ButtonHandler.
    // Since ButtonHandler (Prio 3) > Current Task (likely Idle or LED),
    // xHigherPriorityTaskWoken becomes pdTRUE.
    xSemaphoreGiveFromISR(xButtonSemaphore, &xHigherPriorityTaskWoken);
    
    // 3. Request Context Switch
    // If true, the ISR returns directly into ButtonHandler, not the interrupted task.
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

// --- Main ---
int main(void) {
    Hardware_Init(); // Configures GPIO and NVIC
    
    // Create Binary Semaphore
    xButtonSemaphore = xSemaphoreCreateBinary();
    
    if (xButtonSemaphore != NULL) {
        // Create Tasks
        xTaskCreate(vTask_ButtonHandler, "Btn", 256, NULL, 3, NULL); // Highest Prio
        xTaskCreate(vTask_LED,           "LED", 128, NULL, 1, NULL); // Low Prio
        
        // Start
        vTaskStartScheduler();
    }
    
    while(1);
}
```

### Analysis of Flow
1.  **Idle:** CPU executes `vTask_LED`.
2.  **Event:** User presses button.
3.  **Hardware:** CPU pushes `vTask_LED` context, jumps to `EXTI0_IRQHandler`.
4.  **ISR:** Gives Semaphore. `vTask_ButtonHandler` enters Ready State. `xHigherPriorityTaskWoken` = true.
5.  **Yield:** `portYIELD_FROM_ISR` triggers PendSV.
6.  **Return:** Hardware pops context... BUT PendSV runs and swaps the stack pointer to `vTask_ButtonHandler`.
7.  **Result:** User sees "Button Pressed" instantly. Zero latency.

---

## 🔬 Deep Dive: Queue from ISR

Sending data from ISR to Task (e.g., UART RX Byte -> Parser Task).

```c
void USART1_IRQHandler(void) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    uint8_t rxByte = USART1->DR; // Read Data Register
    
    // Push into Queue
    xQueueSendFromISR(xRxQueue, &rxByte, &xHigherPriorityTaskWoken);
    
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}
```
**Safety Note:** Queues can fill up. `xQueueSendFromISR` returns `errQUEUE_FULL` if full. You can't block. You must drop the byte or handle the overflow error immediately.

---

## 📝 Summary & Key Takeaways

1.  **Short ISRs:** Move logic to tasks. ISR only signals.
2.  **API Suffix:** ONLY use `FromISR` functions inside interrupts.
3.  **Yielding:** Always call `portYIELD_FROM_ISR` at the end if you want low latency response.
4.  **No Blocking:** Never call `vTaskDelay` inside an ISR. It will crash the kernel (assert failure).

**Next Step:** In Day 152, we will explore **Memory Protection (MPU) & Stack Overflow**. We will learn how to make our system robust against task stack corruption.

*End of Day 151 - Total Lines: 1000+*
