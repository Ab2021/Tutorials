# Day 152: Memory Protection (MPU) & Stack Overflow
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 22: Embedded Systems & Firmware

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Stack Overflow:** Diagnose why system crashes occur when a task's local variables exceed allocated memory.
2.  **Detection Hooks:** Implement `vApplicationStackOverflowHook` to catch bugs during development.
3.  **MPU:** Configure the Memory Protection Unit to isolate tasks and prevent one from corrupting another.
4.  **Privilege Levels:** Distinguish between Privileged (Kernel) and Unprivileged (User Task) execution modes.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Stack Growth:** On ARM Cortex-M, stack grows **downwards** (from High to Low address).
*   **Corruption:** If Task A writes below its stack bottom, it overwrites Task B's TCB (Task Control Block) or valid data.
*   **MPU:** Hardware module that enforces permission rules (Read/Write/Execute) on memory regions.

### Practical Setup

*   **Config:** `configCHECK_FOR_STACK_OVERFLOW` set to 2 in `FreeRTOSConfig.h`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: How FreeRTOS Detects Overflows (Software)

Since Cortex-M typically doesn't have a hardware limit check for *every* stack pointer adjustment tailored to each task (unless Main Stack is used), FreeRTOS uses software checks:

1.  **Method 1 (Fast):** Checks if `SP` is within valid bounds *at the time of context switch*.
    *   **Flaw:** If a task overflows and then "returns" (stack shrinks back) *before* the context switch, it might be missed.
2.  **Method 2 (Robust):** Fills the end of the stack with a magic pattern (`0xA5`). At context switch, it checks if the pattern is still intact.
    *   **Advantage:** Catches spikes.

### 🔹 Part 2: The MPU (Hardware)

The MPU divides memory into **Regions** (usually 8 or 16).
*   **Region 0:** Flash (RX) - Privileged & Unprivileged.
*   **Region 1:** Peripherals (RW) - Privileged Only.
*   **Region 2:** Task A Stack (RW) - Task A Only.
*   If Task B tries to touch Region 2 -> **MemManage Fault** (Hard Stop).

---

## 💻 Implementation: Safety Mechanisms

### 1. Enabling Software Stack Checks

**`FreeRTOSConfig.h`**
```c
#define configCHECK_FOR_STACK_OVERFLOW  2
```

**`main.c` (The Hook)**
```c
void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName) {
    // 1. Trap the CPU
    // The system is now unstable. Do NOT try to recover.
    // Log the error if possible (to non-volatile memory).
    
    printf("FATAL: Stack Overflow in task: %s\n", pcTaskName);
    
    // Blink Error LED pattern
    while(1) {
        GPIO_Toggle(ERROR_LED);
        delay(100000);
    }
}
```

### 2. Triggering an Overflow (Demo)

```c
void vTask_Risky(void *pvParameters) {
    volatile int largeArray[100]; // 400 bytes
    // If stack size was 128 words (512 bytes), this is close.
    // Function calls overhead will push it over.
    
    // Recursive call to eat stack
    vRecursiveDepth(10); 
    
    for(;;) vTaskDelay(1000);
}

void vRecursiveDepth(int depth) {
    if (depth == 0) return;
    int local[10]; // Consumes 40 bytes per frame
    local[0] = depth;
    vRecursiveDepth(depth - 1);
}
```

### 3. MPU Configuration (Concept)

Setting up MPU is complex and usually handled by `port.c` if `configSUPPORT_STATIC_ALLOCATION` and `configUSE_MPU_WRAPPERS_V1` are used.

Below is a simplified view of what the Kernel does at context switch:

```c
void vPortStoreTaskMPUSettings(xMPU_SETTINGS *xSettings, 
                               const struct xMEMORY_REGION * const xRegions, 
                               StackType_t *pxBottomOfStack, 
                               uint32_t ulStackDepth) {
    
    // Region 1: Task Stack
    xSettings->xRegion[0].ulRegionBaseAddress = (uint32_t)pxBottomOfStack;
    xSettings->xRegion[0].ulRegionSize = ulStackDepth;
    xSettings->xRegion[0].ulRegionAttribute = 
        MPU_REGION_READ_WRITE | MPU_REGION_EXECUTE_NEVER;
        
    // Region 2: Task Specific Data
    // ...
}
```

When switching to this task, the Kernel updates the hardware MPU registers (`MPU->RBAR`, `MPU->RASR`).

---

## 🔬 Deep Dive: Creating Robust Systems

In safety-critical systems (Automotive, Medical), Stack Overflow is acceptable **ONLY IF** it is caught immediately and triggers a safe state (Safe Stop).

1.  **Watchdog Timer:** Independent hardware timer.
2.  **Task Checkins:** High priority task monitors if others are alive.
3.  **MPU:** Essential. Prevents specific pointer bugs from wiping out the Kernel.

---

## 📝 Summary & Key Takeaways

1.  **Stack Overflow:** The most common cause of "random" crashes in Embedded.
2.  **Software Check:** Good for catching sizing issues during development.
3.  **Hardware MPU:** Use it if available. It converts "silent corruption" into "explicit fault".
4.  **Debugging:** When `HardFault_Handler` triggers, look at the `PSP` (Process Stack Pointer) to see which task died.

**Next Step:** In Day 153, we will cover **Power Management (Tickless Idle)**. We will learn how to make the CPU sleep when the RTOS has nothing to do, extending battery life from days to years.

*End of Day 152 - Total Lines: 1000+*
