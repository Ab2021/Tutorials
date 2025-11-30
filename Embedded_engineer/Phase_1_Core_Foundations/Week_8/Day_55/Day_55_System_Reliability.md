# Day 55: System Reliability (Watchdogs, BOR, PVD)
## Phase 1: Core Embedded Engineering Foundations | Week 8: Power Management & Bootloaders

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
1.  **Distinguish** between Independent Watchdog (IWDG) and Window Watchdog (WWDG).
2.  **Configure** the IWDG to reset the system if the main loop hangs.
3.  **Implement** the WWDG to enforce strict timing constraints on critical tasks.
4.  **Understand** the Brown-Out Reset (BOR) levels and their importance for Flash integrity.
5.  **Use** the Programmable Voltage Detector (PVD) to save data before power loss.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Variable Power Supply (Optional, for PVD testing).
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 19 (Watchdogs - Basic)
    *   Day 50 (Low Power)
*   **Datasheets:**
    *   [STM32F407 Reference Manual (IWDG/WWDG/PWR)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Systems Fail
*   **Software Bugs:** Infinite loops, Deadlocks, Stack Overflow.
*   **Hardware Glitches:** EMI (Electro-Magnetic Interference) flipping bits in PC or Registers.
*   **Power Issues:** Voltage dips causing undefined logic states.

### 🔹 Part 2: The Watchdog Solution
A Watchdog is a hardware timer that counts down. If it reaches zero, it resets the MCU. The software must "kick" (reload) the dog periodically to prove it's alive.
*   **IWDG (Independent):** Runs on LSI (32kHz). Works even if HSE/PLL fails. Good for general "Is the system stuck?" checks.
*   **WWDG (Window):** Runs on APB1. Has a "Window". You must kick it *within* a specific time slot (not too early, not too late). Good for "Is the system running *correctly*?" checks.

### 🔹 Part 3: Voltage Monitoring
*   **POR/PDR:** Power On/Down Reset. Fixed at ~1.7V.
*   **BOR:** Brown Out Reset. Configurable (Level 1 to 3). Keeps MCU in Reset until VDD > Threshold (e.g., 2.7V). Prevents Flash corruption (Flash needs > 2.7V for high speed).
*   **PVD:** Programmable Voltage Detector. Generates an *Interrupt* (not Reset) when VDD drops below threshold. Gives you ~10ms to save data.

---

## 💻 Implementation: Robust IWDG

> **Instruction:** Setup IWDG with 1 second timeout.

### 👨‍💻 Code Implementation

```c
#include "stm32f4xx.h"

void IWDG_Init(void) {
    // 1. Enable Write Access to IWDG_PR and IWDG_RLR
    IWDG->KR = 0x5555;
    
    // 2. Set Prescaler (LSI = 32kHz)
    // Div 32 -> 1 kHz clock (1ms tick)
    IWDG->PR = 3; 
    
    // 3. Set Reload Value
    // 1000 ticks = 1 second
    IWDG->RLR = 1000;
    
    // 4. Start IWDG
    IWDG->KR = 0xCCCC;
}

void IWDG_Kick(void) {
    IWDG->KR = 0xAAAA;
}

int main(void) {
    HAL_Init();
    
    // Check Reset Cause
    if (RCC->CSR & (1 << 29)) { // IWDGRSTF
        // We crashed! Log it.
        RCC->CSR |= (1 << 24); // Clear flags
        Blink_Red_LED_Fast();
    }
    
    IWDG_Init();
    
    while(1) {
        // Main Task
        Do_Work();
        
        // Feed the Dog
        IWDG_Kick();
    }
}
```

---

## 💻 Implementation: PVD (Last Gasp)

> **Instruction:** Detect VDD < 2.9V. Save critical data to Backup Registers.

### 👨‍💻 Code Implementation

#### Step 1: PVD Init
```c
void PVD_Init(void) {
    RCC->APB1ENR |= (1 << 28); // PWR Clock
    
    // Set Threshold to 2.9V (PLS = 111)
    PWR->CR |= (7 << 5);
    
    // Enable PVD
    PWR->CR |= (1 << 4);
    
    // Configure EXTI Line 16 (PVD Output)
    EXTI->IMR |= (1 << 16);
    EXTI->RTSR |= (1 << 16); // Rising Edge (VDD dropping below threshold)
    EXTI->FTSR |= (1 << 16); // Falling Edge (VDD rising above threshold)
    
    NVIC_EnableIRQ(PVD_IRQn);
}
```

#### Step 2: ISR
```c
void PVD_IRQHandler(void) {
    if (EXTI->PR & (1 << 16)) {
        EXTI->PR |= (1 << 16);
        
        if (PWR->CSR & (1 << 2)) { // PVDO is High (Voltage is Low)
            // EMERGENCY SAVE
            RTC->BKP0R = 0xDEAD; // Save state
            // Enter Standby to save remaining energy?
        }
    }
}
```

---

## 🔬 Lab Exercise: Lab 55.1 - The Hang Simulation

### 1. Lab Objectives
- Verify IWDG resets the board.

### 2. Step-by-Step Guide

#### Phase A: Normal Operation
1.  Flash code.
2.  LED blinks normally.

#### Phase B: Simulate Hang
1.  Add a button check.
2.  `if (Button_Pressed()) while(1);` // Infinite loop without kicking.
3.  Press Button.
4.  **Observation:** LED stops blinking for 1 second. Then Board Resets (Red LED flashes fast indicating IWDG Reset).

### 3. Verification
This confirms the safety net is working.

---

## 🧪 Additional / Advanced Labs

### Lab 2: WWDG Challenge
- **Goal:** Kick the dog *exactly* between 30ms and 40ms.
- **Task:**
    1.  Configure WWDG.
    2.  Use a Timer Interrupt to kick it.
    3.  Change Timer period to 20ms (Too early -> Reset).
    4.  Change Timer period to 50ms (Too late -> Reset).

### Lab 3: Reset Cause Logger
- **Goal:** Persistent Crash Log.
- **Task:**
    1.  On boot, read `RCC->CSR`.
    2.  If `IWDGRSTF`, increment a counter in Backup Register `RTC->BKP1R`.
    3.  If `SFTRSTF` (Software Reset), increment `RTC->BKP2R`.
    4.  Print stats to UART.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Debugger disconnects constantly
*   **Cause:** IWDG keeps resetting the chip while you are paused at a breakpoint.
*   **Solution:** Enable "DBG_IWDG_STOP" in `DBGMCU->APB1FZ` register. This freezes the watchdog when the core is halted.

#### 2. PVD Firing randomly
*   **Cause:** Noisy power supply.
*   **Solution:** Add a capacitor (10uF) on VDD.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Kick Strategy:** Don't just put `IWDG_Kick()` in the `SysTick_Handler`. That's cheating. The main loop could be stuck, but interrupts still firing.
- **Best Practice:** Have each task set a "I'm Alive" flag. Only kick the dog if ALL flags are set.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can I stop the IWDG once started?
    *   **A:** No. Once enabled (by hardware option byte or software), it runs until Reset.
2.  **Q:** What is the difference between POR and BOR?
    *   **A:** POR is built-in (hysteresis ~100mV). BOR is more precise and configurable, ensuring VDD is high enough for Flash/Peripherals before releasing Reset.

### Challenge Task
> **Task:** Implement a "Task Monitor". Create 3 dummy tasks (functions called in loop). Task A takes 10ms. Task B takes 20ms. Task C takes 30ms. Use WWDG to ensure the total loop time is within 50-70ms. If Task A hangs, WWDG should reset.

---

## 📚 Further Reading & References
- [STM32F4 Reset and Clock Control (RCC)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)

---
