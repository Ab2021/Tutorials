# Day 16: Timer Fundamentals
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
1.  **Explain** the architecture of STM32 General Purpose Timers (CNT, PSC, ARR).
2.  **Calculate** precise timing intervals based on the APB clock frequency.
3.  **Configure** a timer to generate periodic interrupts.
4.  **Implement** a blocking delay function (`delay_us`) using a hardware timer.
5.  **Debug** timer issues related to clock tree configuration and prescaler values.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 11 (Interrupts)
    *   Day 13 (Clocks/APB Buses)
*   **Datasheets:**
    *   [STM32F407 Reference Manual (Timer Section)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Timer Architecture

#### 1.1 The Core Counter
At the heart of every timer is a counter (`CNT`).
*   **Up-Counting:** 0 -> ARR (Auto-Reload Register) -> 0.
*   **Down-Counting:** ARR -> 0 -> ARR.
*   **Center-Aligned:** 0 -> ARR -> 0.

#### 1.2 Time Base Unit
The speed of the counter is determined by:
1.  **Input Clock ($f_{CLK}$):** Usually the APB bus clock (sometimes x2).
2.  **Prescaler (`PSC`):** Divides the clock.
    *   $f_{CNT} = \frac{f_{CLK}}{PSC + 1}$
3.  **Auto-Reload (`ARR`):** The limit value.
    *   $Period = \frac{ARR + 1}{f_{CNT}}$

**Example:**
*   $f_{CLK} = 16 MHz$
*   We want 1 second period ($1 Hz$).
*   Set $PSC = 15999$. -> $f_{CNT} = 16000000 / 16000 = 1000 Hz$ (1 ms ticks).
*   Set $ARR = 999$. -> $Period = 1000 / 1000 = 1 sec$.

```mermaid
graph LR
    Clock[APB Clock] -->|/ (PSC+1)| Prescaler[PSC]
    Prescaler -->|f_CNT| Counter[CNT Register]
    Counter -->|Compare| Comparator
    ARR[ARR Register] --> Comparator
    Comparator -->|Overflow| UpdateEvent[Update Event (UEV)]
    UpdateEvent --> Interrupt[NVIC IRQ]
```

### 🔹 Part 2: Timer Types in STM32F4
*   **Advanced (TIM1, TIM8):** Dead-time insertion, motor control, break input.
*   **General Purpose (TIM2-TIM5):** 32-bit or 16-bit, Input Capture, Output Compare, PWM.
*   **Basic (TIM6, TIM7):** No I/O pins. Just simple timebase (used for DAC triggering).

### 🔹 Part 3: Shadow Registers
Registers like `ARR` and `PSC` are **buffered**.
*   **Preload Register:** What you write to.
*   **Shadow Register:** What the hardware actually uses.
*   **Update Event (UEV):** The contents of Preload are transferred to Shadow only when an Update Event occurs (Counter overflow or software trigger). This ensures glitch-free changes on the fly.

---

## 💻 Implementation: Microsecond Delay Driver

> **Instruction:** We will create a precise blocking delay function using TIM2 (32-bit timer).

### 🛠️ Hardware/System Configuration
STM32F4 Discovery.

### 👨‍💻 Code Implementation

#### Step 1: Initialization (`timer.c`)

```c
#include "stm32f4xx.h"

void TIM2_Init(void) {
    // 1. Enable TIM2 Clock (APB1)
    RCC->APB1ENR |= (1 << 0);

    // 2. Configure Prescaler
    // Assume APB1 Clock = 16 MHz (Default HSI)
    // We want 1 tick = 1 us.
    // 16 MHz / (15 + 1) = 1 MHz.
    TIM2->PSC = 15;

    // 3. Set Auto-Reload to Max (32-bit)
    TIM2->ARR = 0xFFFFFFFF;

    // 4. Enable Counter
    TIM2->CR1 |= (1 << 0); // CEN
    
    // 5. Force Update to load Prescaler value immediately
    TIM2->EGR |= (1 << 0); // UG
}
```

#### Step 2: Delay Function
```c
void Delay_us(uint32_t us) {
    // 1. Get Start Time
    uint32_t start = TIM2->CNT;
    
    // 2. Wait until delta >= us
    while ((TIM2->CNT - start) < us);
}

void Delay_ms(uint32_t ms) {
    for (uint32_t i = 0; i < ms; i++) {
        Delay_us(1000);
    }
}
```

#### Step 3: Main Loop Test
```c
int main(void) {
    TIM2_Init();
    
    // Init LED (PD12)
    RCC->AHB1ENR |= (1 << 3);
    GPIOD->MODER |= (1 << 24);

    while(1) {
        GPIOD->ODR ^= (1 << 12); // Toggle
        Delay_ms(500); // 0.5s delay
    }
}
```

---

## 🔬 Lab Exercise: Lab 16.1 - The Blinking Interrupt

### 1. Lab Objectives
- Configure TIM3 to generate an interrupt every 1 second.
- Toggle an LED inside the ISR.

### 2. Step-by-Step Guide

#### Phase A: Calculation
*   Clock: 16 MHz.
*   Target: 1 Hz.
*   PSC: 15999 (-> 1 kHz).
*   ARR: 999 (-> 1 Hz).

#### Phase B: Coding
```c
void TIM3_Init(void) {
    RCC->APB1ENR |= (1 << 1); // TIM3
    TIM3->PSC = 15999;
    TIM3->ARR = 999;
    
    // Enable Update Interrupt
    TIM3->DIER |= (1 << 0); // UIE
    
    // Enable NVIC
    NVIC_EnableIRQ(TIM3_IRQn);
    
    // Start Timer
    TIM3->CR1 |= (1 << 0);
}

void TIM3_IRQHandler(void) {
    // Check Status Register
    if (TIM3->SR & (1 << 0)) { // UIF (Update Interrupt Flag)
        TIM3->SR &= ~(1 << 0); // Clear Flag
        
        // Toggle LED
        GPIOD->ODR ^= (1 << 13); // Orange LED
    }
}
```

#### Phase C: Verification
Run the code. The Orange LED should blink at exactly 1 Hz.

### 3. Verification
Measure the period with a stopwatch or oscilloscope.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Measuring Execution Time
- **Goal:** Measure how long a math function takes.
- **Task:**
    1.  Read `TIM2->CNT` (start).
    2.  Run `sin(1.23)` 1000 times.
    3.  Read `TIM2->CNT` (end).
    4.  Print `end - start` (microseconds).

### Lab 3: One-Pulse Mode
- **Goal:** Generate a single pulse of fixed duration when a button is pressed.
- **Task:**
    1.  Configure TIM4 in One-Pulse Mode (`OPM` bit in `CR1`).
    2.  Set `ARR` for pulse width.
    3.  In Button ISR, set `CEN` bit.
    4.  Timer runs once, stops automatically.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Timer Running too Fast/Slow
*   **Cause:** Incorrect assumption about APB Clock frequency.
*   **Detail:** If APB prescaler is != 1, the Timer clock is x2 the APB clock.
    *   Example: System=168MHz. APB1=42MHz (Div 4). TIM2 CLK = 84MHz (x2).
*   **Solution:** Check the Clock Tree configuration in `system_stm32f4xx.c` or Reference Manual.

#### 2. Interrupt Stuck
*   **Cause:** Forgot to clear `UIF` flag in ISR.
*   **Solution:** `TIMx->SR &= ~TIM_SR_UIF;`

#### 3. 16-bit Overflow
*   **Cause:** Using TIM3 (16-bit) for long delays. Max count is 65535.
*   **Solution:** Use TIM2/TIM5 (32-bit) or handle overflow in software.

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **Preload Register:** Always enable `ARPE` (Auto-Reload Preload Enable) if you plan to change the frequency at runtime (e.g., variable frequency tone generation). This prevents the timer from reloading a partial count if the write happens mid-cycle.

### Code Quality
- **Magic Numbers:** Don't use `15999`. Use `#define PRESCALER (SystemCoreClock / 1000 - 1)`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** If `PSC` = 0, what is the counter frequency?
    *   **A:** The same as the input clock ($f_{CLK} / 1$).
2.  **Q:** What is the difference between `TIM_CR1_CEN` and `TIM_EGR_UG`?
    *   **A:** `CEN` starts the counter. `UG` (Update Generation) forces an update event to reload shadow registers from preload registers immediately.

### Challenge Task
> **Task:** Create a "Double Timer" system. TIM2 interrupts every 1ms. It increments a global variable `ms_counter`. TIM3 interrupts every 1s. It checks if `ms_counter` is exactly 1000. If not, it lights a Red LED (Drift detected).

---

## 📚 Further Reading & References
- [STM32 Timer Cookbook (AN4776)](https://www.st.com/resource/en/application_note/dm00236305-general-purpose-timer-cookbook-stmicroelectronics.pdf)

---
