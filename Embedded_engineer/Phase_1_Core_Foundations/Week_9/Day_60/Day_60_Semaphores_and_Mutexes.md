# Day 60: Semaphores & Mutexes
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
1.  **Differentiate** between Binary Semaphores, Counting Semaphores, and Mutexes.
2.  **Implement** Task Synchronization using Binary Semaphores.
3.  **Protect** shared resources (e.g., I2C bus) using Mutexes.
4.  **Explain** Priority Inversion and how Priority Inheritance solves it.
5.  **Debug** Deadlocks in a multi-tasking environment.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   FreeRTOS
*   **Prior Knowledge:**
    *   Day 58 (Scheduling)
*   **Datasheets:**
    *   [FreeRTOS Semaphore/Mutex API](https://www.freertos.org/a00113.html)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Synchronization vs Mutual Exclusion
*   **Synchronization (Signaling):** Task A waits for Task B (or ISR) to say "Go".
    *   Tool: **Binary Semaphore**.
    *   Analogy: A relay race baton pass.
*   **Mutual Exclusion (Locking):** Task A and Task B both want the Printer. Only one can have it.
    *   Tool: **Mutex** (Mutual Exclusion Semaphore).
    *   Analogy: The key to the restroom.

### 🔹 Part 2: The Priority Inversion Problem
Scenario:
1.  **Low Priority (L)** takes Mutex.
2.  **High Priority (H)** preemption L, tries to take Mutex -> Blocks.
3.  **Medium Priority (M)** preempts L (since M > L).
4.  **Result:** H is waiting for L, but L cannot run because M is hogging the CPU. H is effectively blocked by M (Inversion!).

**Solution: Priority Inheritance**
*   When H blocks on a Mutex held by L, the kernel temporarily boosts L's priority to H.
*   L runs, finishes, releases Mutex.
*   L's priority drops back. H takes Mutex.
*   **Note:** Only Mutexes support this. Binary Semaphores do not.

---

## 💻 Implementation: I2C Mutex

> **Instruction:** Two tasks want to use the I2C bus. We must ensure they don't interleave transactions.

### 👨‍💻 Code Implementation

#### Step 1: Handle
```c
SemaphoreHandle_t hI2CMutex;
```

#### Step 2: Init
```c
void I2C_Mutex_Init(void) {
    hI2CMutex = xSemaphoreCreateMutex();
    if (hI2CMutex == NULL) Error_Handler();
}
```

#### Step 3: Thread-Safe Driver
```c
void I2C_Write_Safe(uint8_t addr, uint8_t *data, uint8_t len) {
    // 1. Take Mutex (Wait forever)
    if (xSemaphoreTake(hI2CMutex, portMAX_DELAY) == pdTRUE) {
        
        // 2. Critical Section (Hardware Access)
        I2C_Start();
        I2C_SendAddr(addr);
        for(int i=0; i<len; i++) I2C_SendByte(data[i]);
        I2C_Stop();
        
        // 3. Give Mutex
        xSemaphoreGive(hI2CMutex);
    }
}
```

#### Step 4: Tasks
```c
void vTaskTemp(void *p) {
    while(1) {
        I2C_Write_Safe(0x48, &cmd, 1); // Read Temp
        vTaskDelay(100);
    }
}

void vTaskEEPROM(void *p) {
    while(1) {
        I2C_Write_Safe(0x50, &data, 1); // Write Log
        vTaskDelay(500);
    }
}
```

---

## 🔬 Lab Exercise: Lab 60.1 - ISR Signaling

### 1. Lab Objectives
- Use a Binary Semaphore to signal a task from an ISR (Button Press).

### 2. Step-by-Step Guide

#### Phase A: Setup
1.  Create `hButtonSem = xSemaphoreCreateBinary()`.
2.  Create `vTaskHandler` that blocks on `xSemaphoreTake(hButtonSem)`.

#### Phase B: ISR
```c
void EXTI0_IRQHandler(void) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    
    if (EXTI->PR & 1) {
        EXTI->PR |= 1;
        
        // Give Semaphore
        xSemaphoreGiveFromISR(hButtonSem, &xHigherPriorityTaskWoken);
    }
    
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}
```

#### Phase C: Task
```c
void vTaskHandler(void *p) {
    while(1) {
        if (xSemaphoreTake(hButtonSem, portMAX_DELAY) == pdTRUE) {
            printf("Button Pressed!\n");
            // Do heavy processing here...
        }
    }
}
```

### 3. Verification
Press button. Task prints message. This keeps the ISR short.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Counting Semaphore
- **Goal:** Manage a pool of 3 buffers.
- **Task:**
    1.  `xSemaphoreCreateCounting(3, 3)`.
    2.  5 Tasks try to `Take`.
    3.  Only 3 succeed. 2 block.
    4.  When one `Gives`, one blocked task wakes up.

### Lab 3: Deadlock Demo
- **Goal:** Create a Deadlock (The "Dining Philosophers" problem).
- **Task:**
    1.  Task A takes Mutex 1.
    2.  Task B takes Mutex 2.
    3.  Task A tries to take Mutex 2 (Blocks).
    4.  Task B tries to take Mutex 1 (Blocks).
    5.  **Result:** System hangs.
    6.  **Fix:** Always take mutexes in the same order (Hierarchy). Or use timeouts.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Recursive Deadlock
*   **Cause:** A task takes a Mutex, calls a function, which tries to take the *same* Mutex.
*   **Result:** Deadlock (Standard Mutex is not recursive).
*   **Solution:** Use `xSemaphoreCreateRecursiveMutex()`. It allows the owner to take it multiple times (must give equal times).

#### 2. Priority Inversion (Hidden)
*   **Cause:** Using Binary Semaphore for Mutual Exclusion.
*   **Solution:** Always use Mutex for locking. Only use Binary Semaphore for signaling.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Timeouts:** Avoid `portMAX_DELAY` in critical systems. If the I2C bus hangs, the task hangs. Use a timeout (e.g., 100ms) and handle the error (reset the bus).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can an ISR take a Mutex?
    *   **A:** No! Taking a Mutex might block. ISRs cannot block. Also, who would hold the lock? The ISR is not a task owner.
2.  **Q:** What is the initial count of a Binary Semaphore created with `xSemaphoreCreateBinary`?
    *   **A:** 0 (Empty). You must `Give` it first or use it for signaling (where ISR gives). Note: `xSemaphoreCreateMutex` starts at 1 (Full).

### Challenge Task
> **Task:** Implement a "Gatekeeper Task". Instead of a Mutex, create a dedicated task that owns the I2C bus. Other tasks send "Job Requests" (structs) to a Queue. The Gatekeeper processes them one by one. This avoids locking issues entirely.

---

## 📚 Further Reading & References
- [Priority Inversion on Mars Pathfinder](https://www.cs.cornell.edu/courses/cs614/2019sp/papers/pathfinder.html)

---
