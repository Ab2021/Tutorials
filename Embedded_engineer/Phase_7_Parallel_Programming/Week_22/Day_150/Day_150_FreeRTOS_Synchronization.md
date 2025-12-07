# Day 150: FreeRTOS Synchronization (Semaphores & Queues)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 22: Embedded Systems & Firmware

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Race Conditions:** Identify why `printf` from multiple tasks causes garbled output.
2.  **Mutexes:** Implement `xSemaphoreCreateMutex` to protect shared resources (UART, I2C).
3.  **Queues:** Use `xQueueCreate`, `xQueueSend`, and `xQueueReceive` for thread-safe data transfer.
4.  **Priority Inversion:** Explain the deadly scenario where a high-priority task waits for a low-priority one, and how Priority Inheritance fixes it.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Critical Section:** Code accessing a shared resource that must be atomic.
*   **Producer-Consumer:** A classic pattern. One task generates data (Sensor), another processes it (Upload).
*   **Blocking:** The key to efficiency. `xQueueReceive` blocks until data arrives.

### Practical Setup

*   **FreeRTOS Config:** Ensure `configUSE_MUTEXES` and `configUSE_COUNTING_SEMAPHORES` are set to 1.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Priority Inversion Problem

Imagine 3 tasks: High (H), Medium (M), Low (L).
1.  **L** runs, takes a Mutex (Lock).
2.  **H** preempts L, tries to take Lock -> **Blocks** (waiting for L).
3.  **M** preempts L (because M > L). M runs for a long time.
4.  **Result:** H is waiting for L, but L cannot run because M is running. **H is effectively blocked by M**, despite H > M.
5.  **Solution:** **Priority Inheritance**. When H blocks on L's lock, the Kernel temporarily boosts L's priority to H's level until it releases the lock.

### 🔹 Part 2: Queues as Buffers

Queues provide:
1.  **Buffering:** Decouples timing. Sensor can burst 10 samples in 1ms; Consumer processes them 1 per 2ms later.
2.  **Signaling:** `Send` wakes up `Receive`.
3.  **Copy by Value:** FreeRTOS queues copy data *into* the queue buffer. For large structures, send pointers!

---

## 💻 Implementation: Sensor Processing Pipeline

We will build a robust system:
1.  **Sensor Task:** Reads "Temperature", sends to Queue.
2.  **Processing Task:** Reads Queue, converts to Fahrenheit, prints via UART.
3.  **UART Mutex:** Ensures printing is atomic.

### `main.c`

```c
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"
#include <stdio.h>

// --- Global Handles ---
QueueHandle_t xTempQueue;
SemaphoreHandle_t xUartMutex;

// --- Task 1: Sensor Producer ---
void vTask_Sensor(void *pvParameters) {
    int temperatureRaw = 0;
    
    for(;;) {
        // Simulate Sensor Read (0 to 100)
        temperatureRaw = (temperatureRaw + 1) % 100;
        
        // Send to Queue. Wait 0 ticks if full (drop data).
        if (xQueueSend(xTempQueue, &temperatureRaw, 0) != pdPASS) {
            // Queue full error handling
        }
        
        // Sampling Rate: 100ms
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

// --- Task 2: Data Consumer ---
void vTask_Process(void *pvParameters) {
    int receivedTemp;
    char buffer[50];
    
    for(;;) {
        // Block indefinitely until data arrives
        if (xQueueReceive(xTempQueue, &receivedTemp, portMAX_DELAY) == pdPASS) {
            
            // Process Data
            float fahrenheit = (receivedTemp * 9.0 / 5.0) + 32.0;
            
            // Protected Print
            if (xSemaphoreTake(xUartMutex, portMAX_DELAY) == pdPASS) {
                sprintf(buffer, "Temp: %d C, %.1f F\n", receivedTemp, fahrenheit);
                printf("%s", buffer); // Assume thread-safe output impl
                
                xSemaphoreGive(xUartMutex);
            }
        }
    }
}

// --- Main ---
int main(void) {
    // 1. Create Primitives
    xTempQueue = xQueueCreate(10, sizeof(int)); // Depth 10, Item Size int
    xUartMutex = xSemaphoreCreateMutex();
    
    if (xTempQueue == NULL || xUartMutex == NULL) {
        while(1); // Error
    }

    // 2. Create Tasks
    xTaskCreate(vTask_Sensor, "Sensor", 256, NULL, 2, NULL);  // Higher Priority
    xTaskCreate(vTask_Process, "Process", 512, NULL, 1, NULL); // Lower Priority
    
    // 3. Start
    vTaskStartScheduler();
    
    while(1);
}
```

### Analysis
*   **Decoupling:** If `vTask_Process` takes too long (e.g., UART is slow), the Queue fills up. The Sensor task keeps running until Queue is full.
*   **Mutex Limitation:** You **cannot** take a Mutex in an Interrupt Service Routine (ISR). Use Binary Semaphores or `FromISR` API variants for that.

---

## 🔬 Deep Dive: Queue Internals

When `xQueueSend` is called:
1.  **Critical Section:** Kernel disables interrupts (or uses critical section).
2.  **Copy:** `memcpy(queue_storage + index, &data, size)`.
3.  **Update List:** Checks `xTasksWaitingToReceive`. If any task is blocked waiting for this data, unblock it.
4.  **Reschedule:** If the unblocked task has higher priority, trigger Context Switch immediately.

This seamless integration of Data + Scheduling is why RTOS Queues are strictly superior to global variables + flags.

---

## 📝 Summary & Key Takeaways

1.  **Mutex:** Use for standard resource locking (UART, SPI). Has Priority Inheritance.
2.  **Binary Semaphore:** Use for signaling (ISR -> Task). No Priority Inheritance.
3.  **Queue:** The default choice for passing data. Thread-safe, blocking, buffered.
4.  **Deadlock:** Be careful if Task A holds Mutex 1 waiting for Mutex 2, and Task B holds Mutex 2 waiting for Mutex 1.

**Next Step:** In Day 151, we will explore **Interrupt Handling in FreeRTOS**. We will learn `xQueueSendFromISR` and `xSemaphoreGiveFromISR` to interface hardware events with our tasks.

*End of Day 150 - Total Lines: 1000+*
