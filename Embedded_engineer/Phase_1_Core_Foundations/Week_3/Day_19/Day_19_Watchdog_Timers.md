# Day 19: Watchdog Timers
## Phase 1: Core Embedded Engineering Foundations | Week 3: Timers and GPIO

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
1.  **Differentiate** between the Independent Watchdog (IWDG) and Window Watchdog (WWDG).
2.  **Implement** the IWDG to reset the system in case of a software lockup.
3.  **Configure** the WWDG to detect timing anomalies (early or late execution).
4.  **Debug** watchdog resets by checking the Reset Control & Status Register (RCC_CSR).
5.  **Design** a "Kick Strategy" that ensures the watchdog is only refreshed when the system is healthy.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 13 (Clocks - LSI vs PCLK)
    *   Day 12 (Faults)
*   **Datasheets:**
    *   [STM32F407 Reference Manual (IWDG/WWDG Section)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Watchdog Concept
A Watchdog Timer (WDT) is a hardware counter that counts down. If it reaches zero, it resets the microcontroller. The software must "kick" (reload) the watchdog periodically to prove it is still running correctly.

### 🔹 Part 2: Independent Watchdog (IWDG)
*   **Clock Source:** LSI (Low Speed Internal, ~32 kHz).
*   **Independence:** Runs even if the main clock (HSE/HSI) fails.
*   **Use Case:** Detecting total system freeze or hardware failure.
*   **Key Registers:**
    *   `KR`: Key Register (Write 0xCCCC to start, 0xAAAA to reload, 0x5555 to unlock).
    *   `PR`: Prescaler.
    *   `RLR`: Reload Register.

### 🔹 Part 3: Window Watchdog (WWDG)
*   **Clock Source:** PCLK1 (APB1 Clock).
*   **Window Feature:** You must kick the dog *within a specific time window*.
    *   Too late? Reset (Counter hits 0x3F).
    *   Too early? Reset (Counter > Window Value).
*   **Use Case:** Detecting stuck loops or logic errors where the code runs too fast (skipping steps).

```mermaid
graph LR
    Start[Start Loop] --> Task[Execute Task]
    Task --> Check{Time in Window?}
    Check -->|Yes| Kick[Refresh WWDG]
    Check -->|No (Too Early)| Reset[System Reset]
    Kick --> Start
    Task -->|Timeout (Too Late)| Reset
```

---

## 💻 Implementation: IWDG "Dead Man's Switch"

> **Instruction:** We will configure the IWDG to reset the board if the User Button is held down for more than 2 seconds (simulating a stuck task).

### 🛠️ Hardware/System Configuration
STM32F4 Discovery.

### 👨‍💻 Code Implementation

#### Step 1: Check Reset Cause (`reset_check.c`)
It's crucial to know *why* the system reset.

```c
#include "stm32f4xx.h"
#include <stdio.h>

void Check_Reset_Source(void) {
    if (RCC->CSR & (1 << 29)) { // IWDGRSTF
        printf("Reset Cause: Independent Watchdog!\n");
    } else if (RCC->CSR & (1 << 28)) { // WWDGRSTF
        printf("Reset Cause: Window Watchdog!\n");
    } else if (RCC->CSR & (1 << 26)) { // PINRSTF
        printf("Reset Cause: Reset Button\n");
    } else {
        printf("Reset Cause: Power On\n");
    }
    
    // Clear flags
    RCC->CSR |= (1 << 24); // RMVF
}
```

#### Step 2: IWDG Configuration
*   LSI = 32 kHz.
*   Target Timeout = 2 seconds.
*   Formula: $T = \frac{4 \times 2^{PR} \times RLR}{32000}$
*   Let PR = 4 (Div 64). Clock = 500 Hz (2ms ticks).
*   RLR = 1000. (1000 * 2ms = 2000ms = 2s).

```c
void IWDG_Init(void) {
    // 1. Enable Write Access
    IWDG->KR = 0x5555;
    
    // 2. Set Prescaler (Div 64) -> Bit 2 = 1
    IWDG->PR = 0x04;
    
    // 3. Set Reload Value
    IWDG->RLR = 1000;
    
    // 4. Start IWDG
    IWDG->KR = 0xCCCC;
}

void IWDG_Refresh(void) {
    IWDG->KR = 0xAAAA;
}
```

#### Step 3: Main Loop
```c
int main(void) {
    // Init UART...
    Check_Reset_Source();
    
    // Init Button (PA0)
    RCC->AHB1ENR |= (1 << 0);
    GPIOA->MODER &= ~0x3;
    
    IWDG_Init();
    
    printf("System Running...\n");

    while(1) {
        // Simulate Task
        for(int i=0; i<100000; i++); 
        
        // Check if Button is pressed (Simulate Stuck)
        if (GPIOA->IDR & (1 << 0)) {
            printf("Stuck! Not kicking the dog...\n");
            // Don't refresh
            while(GPIOA->IDR & (1 << 0));
        } else {
            IWDG_Refresh();
        }
    }
}
```

---

## 🔬 Lab Exercise: Lab 19.1 - The Window Challenge

### 1. Lab Objectives
- Configure WWDG.
- Demonstrate that refreshing "too early" causes a reset.

### 2. Step-by-Step Guide

#### Phase A: Setup
*   PCLK1 = 42 MHz.
*   WWDG Prescaler = 8.
*   Counter Clock = 42MHz / 4096 / 8 = 1281 Hz (780us ticks).
*   Window = 80 (0x50).
*   Counter = 127 (0x7F).
*   Refresh allowed only when Counter < 80.

#### Phase B: Coding
```c
void WWDG_Init(void) {
    RCC->APB1ENR |= (1 << 11); // WWDG Clock
    
    // Set Prescaler (Div 8) -> WDGTB = 11
    // Set Window = 0x50
    WWDG->CFR = (3 << 7) | 0x50;
    
    // Enable and Set Counter = 0x7F
    WWDG->CR = (1 << 7) | 0x7F;
}

void WWDG_Refresh(void) {
    // Update Counter to 0x7F
    WWDG->CR = 0x7F;
}
```

#### Phase C: Test
1.  Loop with `Delay_ms(1)`. Call `WWDG_Refresh()`.
    *   Result: Reset! (1ms is too fast, counter is still > 80).
2.  Loop with `Delay_ms(50)`. Call `WWDG_Refresh()`.
    *   Result: Success.

### 3. Verification
Check `RCC->CSR` to confirm `WWDGRSTF`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Early Warning Interrupt (EWI)
- **Goal:** Save system state before WWDG reset.
- **Task:**
    1.  Enable EWI in WWDG configuration.
    2.  In `WWDG_IRQHandler`, write a "Panic Log" to Backup SRAM (BKPSRAM).
    3.  Wait for reset.

### Lab 3: Task Monitoring
- **Goal:** Ensure multiple tasks are running.
- **Task:**
    1.  Create `uint32_t task_flags = 0;`.
    2.  Task A sets bit 0. Task B sets bit 1.
    3.  In the main loop, only call `IWDG_Refresh()` if `task_flags == 0x3`.
    4.  Clear `task_flags` after refresh.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Debugger Disconnects
*   **Symptom:** You hit a breakpoint, and the board resets.
*   **Cause:** The Watchdog keeps counting while the CPU is halted.
*   **Solution:** Configure `DBGMCU` to freeze watchdogs during debug.
    *   `DBGMCU->APB1FZ |= DBGMCU_APB1_FZ_DBG_IWDG_STOP;`

#### 2. Reset Loop
*   **Symptom:** Board resets immediately after startup.
*   **Cause:** Initialization takes too long (longer than watchdog timeout).
*   **Solution:** Kick the dog *during* initialization, or enable it later.

---

## ⚡ Optimization & Best Practices

### Safety Critical Systems
- **External Watchdog:** For IEC 61508 / ISO 26262 compliance, an internal watchdog is often not enough. Use an external hardware watchdog IC that monitors a GPIO toggle.

### Code Quality
- **Window:** Use WWDG for critical control loops (e.g., motor control) where timing accuracy is safety-critical. Use IWDG for general system health.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can you stop the IWDG once it is started?
    *   **A:** No. Once enabled (by hardware option byte or software), it runs until the next reset.
2.  **Q:** What is the "Key" value to reload the IWDG?
    *   **A:** `0xAAAA`.

### Challenge Task
> **Task:** Implement a "Dual Watchdog" system. Use IWDG for global timeout (1s) and WWDG for tight loop monitoring (10ms window).

---

## 📚 Further Reading & References
- [STM32 Watchdog Application Note (AN3268)](https://www.st.com/resource/en/application_note/dm00025345-stm32-cross-series-timer-overview-stmicroelectronics.pdf)

---
