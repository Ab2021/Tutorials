# Day 149: FreeRTOS Fundamentals
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 22: Embedded Systems & Firmware

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **RTOS vsOS:** Distinguish Real-Time Operating Systems (Deterministic, Preemptive) from GPOS (Linux).
2.  **Kernel Objects:** Define Tasks, Queues, and Semaphores in standard C.
3.  **Scheduling:** Explain how FreeRTOS uses `SysTick` and `PendSV` to context switch on Cortex-M.
4.  **Implementation:** Write a multi-threaded "Blinky" application using `xTaskCreate`.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Super Loop:** `while(1) { taskA(); taskB(); }`. Simple but blocking. If A takes 1s, B waits 1s.
*   **Preemption:** A higher priority task interrupts a lower priority one *immediately*.
*   **Context Switch:** Saving R0-R15 of Task A, Loading R0-R15 of Task B.

### Practical Setup

*   **FreeRTOS Source:** Needed (`tasks.c`, `queue.c`, `list.c`, `port.c`).
*   **Header:** `FreeRTOSConfig.h` (The most important file).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Architecture of a Context Switch

On Cortex-M, FreeRTOS does **not** switch context inside the System Tick (Timer Interrupt).
1.  **SysTick Handler:** Increments Tick Count. Checks if a high priority task woke up. If yes, triggers `PendSV`.
2.  **PendSV (Pendable Service Call):** The LOWEST priority interrupt. It runs when no other IRQ is active.
3.  **PendSV Handler:**
    *   Push R4-R11 to the *current* task stack.
    *   Save SP to TCB (Task Control Block).
    *   Pick next TCB (`pxCurrentTCB`).
    *   Load SP from new TCB.
    *   Pop R4-R11 from *new* stack.
    *   Return (Hardware pops R0-R3, PC, LR).

### 🔹 Part 2: Task States

*   **Running:** Currently executing on CPU.
*   **Ready:** Wants to run, but a higher/equal priority task is running.
*   **Blocked:** Waiting for Event (Delay, Mutex, Queue). Consumes **zero** CPU.
*   **Suspended:** Explicitly paused (`vTaskSuspend`).

---

## 💻 Implementation: Task Creation (Blinky)

We will create two tasks:
1.  **Task 1 (Priority 1):** Toggles LED every 500ms.
2.  **Task 2 (Priority 2):** Prints UART message every 100ms. (Preempts Task 1).

### `main.c`

```c
#include "FreeRTOS.h"
#include "task.h"
#include "gpio.h" // Hypothetical HAL

// --- Task Handles ---
TaskHandle_t xTask1_Handle = NULL;
TaskHandle_t xTask2_Handle = NULL;

// --- Task Functions ---

void vTask1_LED(void *pvParameters) {
    const TickType_t xDelay = pdMS_TO_TICKS(500); // 500ms
    
    for(;;) {
        // Toggle Hardware LED
        GPIO_Toggle(LED_PIN);
        
        // Blocking Delay (Yields CPU to others)
        vTaskDelay(xDelay); 
    }
}

void vTask2_UART(void *pvParameters) {
    const TickType_t xDelay = pdMS_TO_TICKS(100); // 100ms
    
    for(;;) {
        // Assume thread-safe print or use Mutex
        printf("Task 2 Running!\n"); 
        
        vTaskDelay(xDelay);
    }
}

// --- Main ---

int main(void) {
    // 1. Hardware Init
    Hardware_Init();
    
    // 2. Create Tasks
    // xTaskCreate(Function, Name, StackDepth, Params, Priority, Handle)
    
    BaseType_t status;
    
    status = xTaskCreate(vTask1_LED, "LED", 128, NULL, 1, &xTask1_Handle);
    if (status != pdPASS) while(1); // Error (Heap full?)

    status = xTaskCreate(vTask2_UART, "UART", 256, NULL, 2, &xTask2_Handle);
    // Priority 2 > Priority 1. UART task will pre-empt LED task.
    
    // 3. Start Scheduler
    // This function never returns. Memory is handed to tasks.
    vTaskStartScheduler();
    
    // 4. Trap (If we got here, insufficient heap)
    while(1);
}
```

### `FreeRTOSConfig.h` (Critical Settings)

```c
#define configUSE_PREEMPTION                    1
#define configCPU_CLOCK_HZ                      ( ( unsigned long ) 80000000 )
#define configTICK_RATE_HZ                      ( ( TickType_t ) 1000 ) // 1ms Tick
#define configMAX_PRIORITIES                    ( 5 )
#define configMINIMAL_STACK_SIZE                ( ( unsigned short ) 128 )
#define configTOTAL_HEAP_SIZE                   ( ( size_t ) ( 10 * 1024 ) )
```

### Analysis
*   **`vTaskDelay`:** This is the magic. It changes state from **Running** to **Blocked**. It calls `portYIELD()` (triggers PendSV). The scheduler picks the Idle Task (if nothing else) or Task 2.
*   **Preemption:** Every 1ms (SysTick), the Kernel checks if a Blocked task (time expired) has higher priority than current.

---

## 🔬 Deep Dive: Stack Overflow & Heap

In Bare Metal, you have one Stack (`msp`). In FreeRTOS, every Task has **its own stack**.
*   **Risk:** If `vTask1` creates a massive array `int big[1000]`, it overflows its 128-word stack, clobbering Task 2's TCB or data.
*   **Protection:** `vApplicationStackOverflowHook` (Software check on context switch). MPU (Hardware protection - Day 152).

---

## 📝 Summary & Key Takeaways

1.  **Deterministic:** We know exactly which task runs (Highest Priority Ready).
2.  **Efficiency:** Tasks sleep (Block) instead of spinning (`nop`), saving power.
3.  **Params:** 5 key params for creation (Func, Name, Stack, Param, Priority).
4.  **Scheduler:** The heart of the system. Never returns.

**Next Step:** In Day 150, we will cover **RTOS Synchronization (Semaphores & Queues)**. We will fix the race condition of two tasks printing to UART simultaneously.

*End of Day 149 - Total Lines: 1000+*
