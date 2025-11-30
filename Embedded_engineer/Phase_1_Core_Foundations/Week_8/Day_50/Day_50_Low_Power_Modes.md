# Day 50: Low Power Modes (Sleep, Stop, Standby)
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
1.  **Analyze** the sources of power consumption in CMOS circuits (Static vs Dynamic).
2.  **Differentiate** between STM32 Low Power Modes: Sleep, Stop, and Standby.
3.  **Implement** Wakeup strategies using Interrupts (WFI) and Events (WFE).
4.  **Configure** the Power Controller (PWR) to enter Stop Mode and reduce current to µA range.
5.  **Recover** from Standby Mode (which acts like a System Reset).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Multimeter (with µA/mA range) for current measurement.
    *   Jumper wires to measure IDD (Current).
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 11 (Interrupts/NVIC)
    *   Day 8 (Cortex-M Architecture)
*   **Datasheets:**
    *   [STM32F407 Reference Manual (PWR Section)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)
    *   [AN4365: Using STM32F4 Low Power Modes](https://www.st.com/resource/en/application_note/dm00104048-using-stm32f4-mcu-power-modes-stmicroelectronics.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Power Consumption Physics
Power in a microcontroller comes from two main sources:
1.  **Dynamic Power ($P_{dyn}$):** Switching transistors.
    *   $P_{dyn} = C \times V^2 \times f$
    *   Depends on Clock Frequency ($f$) and Voltage ($V$).
    *   **Optimization:** Lower the clock, lower the voltage, disable unused clocks.
2.  **Static Power ($P_{stat}$):** Leakage current.
    *   Depends on Temperature and Process technology.
    *   **Optimization:** Power gating (turning off power domains).

### 🔹 Part 2: STM32 Power Modes
The STM32F4 offers three main low-power modes to balance startup time vs power saving.

| Mode | Regulator | Clocks | SRAM | Wakeup Sources | Wakeup Time | Current (Typ) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Run** | On | On | Retained | N/A | N/A | ~50 mA |
| **Sleep** | On | CPU Off, Periph On | Retained | Any Interrupt | Immediate | ~15 mA |
| **Stop** | On/Low Power | Oscillators Off | Retained | EXTI Line | ~15 µs | ~100 µA |
| **Standby** | Off | Off | **Lost** | WKUP Pin, RTC, Reset | ~300 µs (Reset) | ~2 µA |

#### 2.1 Sleep Mode
*   **Concept:** "Nap". CPU stops executing instructions. Peripherals (UART, SPI, Timers) keep running.
*   **Entry:** `__WFI()` (Wait For Interrupt) or `__WFE()` (Wait For Event).
*   **Exit:** Any Interrupt (e.g., Systick, UART Rx).

#### 2.2 Stop Mode
*   **Concept:** "Deep Sleep". 1.2V domain remains powered but clocks (HSI/HSE/PLL) are stopped.
*   **Entry:** Set `PDDS=0` (Power Down Deep Sleep) and `LPSDSR` in PWR_CR. Set `SLEEPDEEP` bit in SCB. Call `__WFI()`.
*   **Exit:** Any **EXTI** Line. (GPIO, RTC Alarm, USB Wakeup).
*   **Note:** Upon wakeup, the system runs on HSI (16 MHz). You must re-enable PLL!

#### 2.3 Standby Mode
*   **Concept:** "Hibernate". 1.2V domain is powered off. Only Backup Domain (RTC + Backup SRAM) works.
*   **Entry:** Set `PDDS=1`. Set `SLEEPDEEP`. Call `__WFI()`.
*   **Exit:** WKUP Pin (PA0), RTC Alarm, IWDG Reset, NRST pin.
*   **Note:** Wakeup acts like a **System Reset**. Variables in SRAM are lost.

---

## 💻 Implementation: Sleep Mode (Blinky)

> **Instruction:** Blink LED, then Sleep. Wake up on Systick (every 1ms) or Button.

### 👨‍💻 Code Implementation

#### Step 1: Basic Sleep
```c
#include "stm32f4xx.h"

void Sleep_Test(void) {
    // 1. Enable LED
    RCC->AHB1ENR |= (1 << 3); // GPIOD
    GPIOD->MODER |= (1 << 24); // PD12 Output

    while(1) {
        GPIOD->ODR ^= (1 << 12); // Toggle
        Delay_ms(100);
        
        // Enter Sleep Mode
        // Request Wait For Interrupt
        // CPU stops here until an interrupt occurs (e.g., SysTick)
        __WFI(); 
        
        // After wakeup, execution continues here
    }
}
```
*Critique:* Since SysTick fires every 1ms, the CPU wakes up every 1ms. This isn't very efficient. To save more power, we should disable SysTick before sleeping if we want to sleep longer.

---

## 💻 Implementation: Stop Mode (Button Wakeup)

> **Instruction:** Configure PA0 (Button) as EXTI. Enter Stop Mode. Wake up on button press.

### 🛠️ Hardware/System Configuration
*   **PA0:** Input (Button).
*   **EXTI0:** Rising Edge.
*   **PD12:** LED (Status).

### 👨‍💻 Code Implementation

#### Step 1: EXTI Config
```c
void EXTI0_Init(void) {
    RCC->AHB1ENR |= (1 << 0); // GPIOA
    RCC->APB2ENR |= (1 << 14); // SYSCFG
    
    SYSCFG->EXTICR[0] &= ~0xF; // PA0
    EXTI->IMR |= (1 << 0);     // Unmask Line 0
    EXTI->RTSR |= (1 << 0);    // Rising Edge
    
    NVIC_EnableIRQ(EXTI0_IRQn);
}

void EXTI0_IRQHandler(void) {
    if (EXTI->PR & (1 << 0)) {
        EXTI->PR |= (1 << 0); // Clear Pending
        // Wakeup happens automatically
    }
}
```

#### Step 2: Enter Stop Mode
```c
void Enter_StopMode(void) {
    // 1. Enable PWR Clock
    RCC->APB1ENR |= (1 << 28);
    
    // 2. Clear PDDS (Power Down Deep Sleep) -> Stop Mode
    PWR->CR &= ~(1 << 1);
    
    // 3. Set Low Power Deep Sleep (LPSDSR) for Voltage Regulator (Optional, saves more)
    PWR->CR |= (1 << 0);
    
    // 4. Set SLEEPDEEP bit in Cortex-M System Control Block
    SCB->SCR |= (1 << 2);
    
    // 5. Request Wait For Interrupt
    __WFI();
    
    // --- WAKEUP POINT ---
    
    // 6. Clear SLEEPDEEP bit
    SCB->SCR &= ~(1 << 2);
    
    // 7. Re-enable PLL (Important! We woke up on HSI)
    SystemClock_Config(); 
}
```

#### Step 3: Main Loop
```c
int main(void) {
    HAL_Init();
    SystemClock_Config(); // 168 MHz
    EXTI0_Init();
    
    while(1) {
        // Active: Flash LED fast
        for(int i=0; i<10; i++) {
            GPIOD->ODR ^= (1 << 12);
            Delay_ms(50);
        }
        
        // Go to Stop Mode
        GPIOD->ODR &= ~(1 << 12); // LED Off
        Enter_StopMode();
        
        // We are back!
    }
}
```

---

## 💻 Implementation: Standby Mode

> **Instruction:** Enter Standby. Wake up via WKUP pin (PA0).

### 👨‍💻 Code Implementation

```c
void Enter_StandbyMode(void) {
    RCC->APB1ENR |= (1 << 28); // PWR Clock
    
    // 1. Enable Wakeup Pin (EWUP)
    PWR->CSR |= (1 << 8);
    
    // 2. Set PDDS (Power Down Deep Sleep) -> Standby Mode
    PWR->CR |= (1 << 1);
    
    // 3. Clear WUF (Wakeup Flag)
    PWR->CR |= (1 << 2);
    
    // 4. Set SLEEPDEEP
    SCB->SCR |= (1 << 2);
    
    // 5. Enter
    __WFI();
}

int main(void) {
    // Check if we woke up from Standby
    if (PWR->CSR & (1 << 0)) { // WUF set
        PWR->CR |= (1 << 2); // Clear WUF
        // Blink Green LED (Wakeup)
    } else {
        // Blink Red LED (Fresh Reset)
    }
    
    Delay_ms(2000);
    Enter_StandbyMode();
    
    while(1); // Should not reach here
}
```

---

## 🔬 Lab Exercise: Lab 50.1 - Current Measurement

### 1. Lab Objectives
- Measure the IDD of the STM32 in Run, Sleep, Stop, and Standby.

### 2. Step-by-Step Guide

#### Phase A: Setup
1.  Remove the **IDD Jumper** (JP1 on Discovery Board).
2.  Connect Multimeter (Ampermeter mode) across the pins.

#### Phase B: Measure
1.  **Run (168 MHz):** Expect ~60-80 mA.
2.  **Sleep:** Expect ~20-30 mA.
3.  **Stop:** Expect ~1-2 mA (on Discovery board, other chips consume power too). On a bare chip, it's < 500 µA.
4.  **Standby:** Expect ~20 µA (mostly board leakage).

### 3. Verification
Verify that pressing the button wakes the board and current jumps back to Run levels.

---

## 🧪 Additional / Advanced Labs

### Lab 2: RTC Alarm Wakeup
- **Goal:** Wake up from Standby every 10 seconds.
- **Task:**
    1.  Configure RTC Alarm A.
    2.  Enable RTC Alarm Interrupt in EXTI (Line 17).
    3.  Enter Standby.
    4.  Wait.

### Lab 3: PVD (Programmable Voltage Detector)
- **Goal:** Detect low battery.
- **Task:**
    1.  Configure PVD Threshold (e.g., 2.5V).
    2.  Enable PVD Interrupt (EXTI 16).
    3.  Lower VDD (if using variable supply).
    4.  ISR should fire -> Save critical data to Backup Registers -> Enter Standby.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Cannot Connect Debugger (ST-Link)
*   **Cause:** If the code enters Stop/Standby immediately after reset, the debugger can't attach.
*   **Solution:** Hold the Reset button. Start the debugger connection. Release Reset. Or add a `Delay_ms(2000)` at the start of `main`.

#### 2. Immediate Wakeup
*   **Cause:** Pending interrupt not cleared (e.g., Systick, EXTI PR).
*   **Solution:** Ensure `EXTI->PR` is cleared. Suspend Systick (`SysTick->CTRL = 0`) before sleeping.

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **Flash Stop Mode:** In Stop mode, Flash can be powered down (`FPDS` bit). This adds wakeup latency but saves ~100 µA.

### Code Quality
- **Low Power Manager:** Create a central `Power_Manager.c` that handles the complex sequence of saving context, configuring clocks, and entering modes.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Does DMA work in Stop Mode?
    *   **A:** No. HSI/HSE are off. No clocks = No DMA.
2.  **Q:** How do you retain data in Standby?
    *   **A:** Use the 4KB Backup SRAM (if enabled/battery backed) or the 20 Backup Registers (80 bytes). Regular SRAM is lost.

### Challenge Task
> **Task:** Implement "Auto-Sleep". If no button is pressed for 5 seconds, enter Stop Mode. Pressing button wakes up and resets the 5s timer.

---

## 📚 Further Reading & References
- [STM32F4 Power Control (RM0090)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)

---
