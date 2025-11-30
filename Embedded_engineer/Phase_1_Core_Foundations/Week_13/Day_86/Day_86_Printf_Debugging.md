# Day 86: Printf Debugging and Logging
## Phase 1: Core Embedded Engineering Foundations | Week 13: Debugging and Testing

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
1.  **Retarget** the standard C library `printf` function to output via UART.
2.  **Implement** a structured Logging System with Log Levels (INFO, WARN, ERROR).
3.  **Design** a Non-Blocking Logger using Circular Buffers and DMA.
4.  **Analyze** the performance impact of logging on real-time systems.
5.  **Use** ANSI Color Codes to enhance log readability.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   USB-UART Bridge (if not using ST-LINK VCP).
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   Serial Terminal (Putty, TeraTerm, or VS Code Serial Monitor).
*   **Prior Knowledge:**
    *   Day 22 (UART Basics)
    *   Day 14 (Ring Buffers)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: How `printf` Works
*   **Application:** Calls `printf("Hello %d", 42);`.
*   **Newlib (C Library):** Formats the string into a buffer. Calls `_write(file, ptr, len)`.
*   **System Call (Stub):** We must implement `_write`.
*   **Driver:** `_write` calls `HAL_UART_Transmit`.

### 🔹 Part 2: Blocking vs Non-Blocking
*   **Blocking:** `printf` waits for UART to finish sending.
    *   *Pros:* Simple. Safe (data always sent).
    *   *Cons:* CPU stalls. 100 chars @ 115200 baud = 8.7ms delay! **Disaster for Audio/Motor Control.**
*   **Non-Blocking (Async):** `printf` writes to a RAM buffer and returns immediately.
    *   *Pros:* Fast (< 10us).
    *   *Cons:* Complex. Buffer can overflow.

### 🔹 Part 3: Structured Logging
Instead of random printfs, use macros:
*   `LOG_INFO("System Boot");` -> `[INFO] [100ms] System Boot`
*   `LOG_ERROR("Malloc Fail");` -> `[ERR ] [105ms] Malloc Fail`

---

## 💻 Implementation: Retargeting Printf

> **Instruction:** Implement `_write` to redirect stdout to UART2.

### 👨‍💻 Code Implementation

#### Step 1: `syscalls.c` (or `retarget.c`)
```c
#include <sys/stat.h>
#include <unistd.h>
#include "stm32f4xx_hal.h"

extern UART_HandleTypeDef huart2;

int _write(int file, char *ptr, int len) {
    // 1 = stdout, 2 = stderr
    if (file == 1 || file == 2) {
        HAL_UART_Transmit(&huart2, (uint8_t*)ptr, len, 1000);
        return len;
    }
    return -1;
}
```

#### Step 2: Test
```c
int main(void) {
    HAL_Init();
    MX_USART2_UART_Init();
    
    printf("Hello World! Counter: %d\n", 123);
    
    // Float support requires linker flag: -u _printf_float
    printf("Pi: %.2f\n", 3.14159);
}
```

---

## 💻 Implementation: Async Logger

> **Instruction:** Create a high-performance logger using a Ring Buffer and DMA.

### 👨‍💻 Code Implementation

#### Step 1: The Ring Buffer
```c
#define LOG_BUF_SIZE 2048
char logBuffer[LOG_BUF_SIZE];
volatile int head = 0;
volatile int tail = 0;
volatile int dmaBusy = 0;

void Log_Write(char *data, int len) {
    for(int i=0; i<len; i++) {
        int next = (head + 1) % LOG_BUF_SIZE;
        if (next != tail) { // If not full
            logBuffer[head] = data[i];
            head = next;
        } else {
            // Overflow! Drop char or block?
            // Dropping is safer for real-time.
        }
    }
    
    // Trigger TX if idle
    Log_Flush();
}
```

#### Step 2: The Flush Logic (DMA)
```c
void Log_Flush(void) {
    if (dmaBusy) return;
    if (head == tail) return; // Empty
    
    // Calculate contiguous block size
    int len;
    if (head > tail) {
        len = head - tail;
    } else {
        len = LOG_BUF_SIZE - tail; // Wrap around later
    }
    
    dmaBusy = 1;
    HAL_UART_Transmit_DMA(&huart2, (uint8_t*)&logBuffer[tail], len);
}

void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart->Instance == USART2) {
        // Advance tail
        int len = huart->TxXferSize; // How many we just sent
        tail = (tail + len) % LOG_BUF_SIZE;
        
        dmaBusy = 0;
        Log_Flush(); // Send more if available
    }
}
```

#### Step 3: The Macros
```c
#define LOG_LEVEL_INFO 1
#define LOG_LEVEL_ERR  2

void Log_Msg(int level, const char *fmt, ...) {
    char buf[128];
    va_list args;
    va_start(args, fmt);
    
    int len = 0;
    
    // Timestamp
    len += snprintf(buf+len, 128-len, "[%lu] ", HAL_GetTick());
    
    // Level
    if (level == LOG_LEVEL_INFO) len += snprintf(buf+len, 128-len, "[INFO] ");
    else len += snprintf(buf+len, 128-len, "[ERR ] ");
    
    // Message
    len += vsnprintf(buf+len, 128-len, fmt, args);
    
    // Newline
    if (len < 127) { buf[len++] = '\n'; buf[len] = 0; }
    
    va_end(args);
    
    Log_Write(buf, len);
}
```

---

## 🔬 Lab Exercise: Lab 86.1 - Logger Stress Test

### 1. Lab Objectives
- Verify Non-Blocking behavior.
- Ensure no data corruption during high load.

### 2. Step-by-Step Guide

#### Phase A: Setup
1.  Main Loop: Toggle LED every 100ms.
2.  Also in Main Loop: `Log_Msg(LOG_LEVEL_INFO, "Stress Test Message %d", i++);`

#### Phase B: Observation
1.  **Blocking Mode:** LED blinks slowly (UART bottleneck).
2.  **Async Mode:** LED blinks perfectly at 100ms. UART output lags behind but catches up.

### 3. Verification
If output is garbled, check Ring Buffer logic (Head/Tail atomicity). Disable interrupts during `head` update if needed, or use atomic instructions.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Color Logs
- **Goal:** Pretty terminal.
- **Task:**
    1.  Define ANSI codes: `#define ANSI_RED "\x1b[31m"`
    2.  Update `Log_Msg` to insert Red for Errors, Green for Info.
    3.  Don't forget reset code `\x1b[0m` at end of line.

### Lab 3: SD Card Logging
- **Goal:** Black Box Recorder.
- **Task:**
    1.  Modify `Log_Flush`.
    2.  Instead of UART, write to a second buffer.
    3.  When second buffer full (512 bytes), write to SD Card (`f_write`).
    4.  **Note:** SD Write is blocking (SPI). Do this in a Low Priority Task!

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Float Printf not working
*   **Cause:** Newlib-nano doesn't include float support by default to save space.
*   **Solution:** Add linker flag `-u _printf_float`.

#### 2. HardFault in `vsnprintf`
*   **Cause:** Stack overflow. `printf` uses a lot of stack.
*   **Solution:** Increase Stack Size (0x400 -> 0x800).

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Compile-Time Removal:**
    ```c
    #if LOG_LEVEL >= LOG_LEVEL_INFO
    #define LOG_INFO(...) Log_Msg(1, __VA_ARGS__)
    #else
    #define LOG_INFO(...) ((void)0)
    #endif
    ```
    This removes the code entirely from the binary if logging is disabled.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is `dmaBusy` flag needed?
    *   **A:** We cannot start a new DMA transfer if one is already active. We must wait for `TxCpltCallback`.
2.  **Q:** What happens if the Ring Buffer fills up?
    *   **A:** In this implementation, we drop data. This is better than crashing or blocking the real-time system.

### Challenge Task
> **Task:** Implement "Deferred Formatting". Instead of `snprintf` (slow) in the caller, copy the raw arguments (format string pointer + values) to the buffer. Format them later in a low-priority task. Extremely fast!

---

## 📚 Further Reading & References
- [Memfault: Interrupt-Safe Logging](https://interrupt.memfault.com/blog/firmware-logging)

---
