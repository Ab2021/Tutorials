# Day 7: Week 1 Review and Mini-Project
## Phase 1: Core Embedded Engineering Foundations | Week 1: Embedded C Fundamentals

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
1.  **Synthesize** all concepts from Week 1 (Pointers, Bit Ops, Structs, MMIO) into a cohesive project.
2.  **Architect** a modular embedded application using a Layered Architecture approach.
3.  **Debug** complex logical errors using GDB and logical deduction.
4.  **Conduct** a code review based on embedded coding standards (MISRA C principles).
5.  **Deliver** a working "Multi-LED Pattern Controller" with user input.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Days 1-6 (All concepts)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Layered Architecture

#### 1.1 Why Layers?
Spaghetti code (everything in `main.c`) is unmaintainable. We split code into layers:
1.  **BSP (Board Support Package):** Low-level hardware initialization (Clocks, GPIOs).
2.  **Drivers:** Abstract interfaces for peripherals (LED_On, Button_Read).
3.  **Middleware:** Complex logic (State Machines, Protocols).
4.  **Application:** High-level business logic (The "What", not the "How").

```mermaid
graph TD
    App[Application Layer (main.c)] --> Middle[Middleware (Pattern Logic)]
    Middle --> Driver[Driver Layer (LED/Button)]
    Driver --> BSP[BSP / CMSIS (Registers)]
    BSP --> HW[Hardware (STM32F4)]
```

#### 1.2 Modular Design Principles
*   **Encapsulation:** `static` variables and functions to hide internal state.
*   **Abstraction:** `LED_SetColor(RED)` instead of `GPIOD->ODR |= (1<<14)`.
*   **Decoupling:** The Application shouldn't know that the LED is on Port D.

### 🔹 Part 2: Review of Key Concepts

#### 2.1 Pointers & Memory
*   **Volatile:** Essential for registers.
*   **MMIO:** Casting integers to pointers to access hardware.
*   **Structs:** Organizing registers and configuration data.

#### 2.2 Bit Manipulation
*   **Set:** `REG |= MASK`
*   **Clear:** `REG &= ~MASK`
*   **Toggle:** `REG ^= MASK`
*   **Check:** `if (REG & MASK)`

---

## 💻 Implementation: Mini-Project (LED Pattern Controller)

> **Project Goal:** Create a system where the User Button cycles through different LED blinking patterns.
> *   **Pattern 0:** All Off
> *   **Pattern 1:** Rotate Clockwise
> *   **Pattern 2:** Blink All
> *   **Pattern 3:** SOS (Short-Short-Short, Long-Long-Long, Short-Short-Short)

### 🛠️ Hardware/System Configuration
*   **Button:** PA0 (Input, Floating).
*   **LEDs:** PD12, PD13, PD14, PD15.

### 👨‍💻 Code Implementation

#### File Structure
*   `main.c`: Application entry.
*   `bsp.c` / `bsp.h`: Hardware init.
*   `led_driver.c` / `led_driver.h`: LED control.
*   `button_driver.c` / `button_driver.h`: Button reading with debouncing.

#### Step 1: BSP Layer (`bsp.c`)
Handles the raw register setup.

```c
#include "stm32f4xx.h"
#include "bsp.h"

void BSP_Init(void) {
    // 1. Enable Clocks (GPIOA for Button, GPIOD for LEDs)
    RCC->AHB1ENR |= (1 << 0) | (1 << 3);

    // 2. Configure LEDs (PD12-15) as Output
    GPIOD->MODER &= ~(0xFF000000); // Clear bits 24-31
    GPIOD->MODER |=  (0x55000000); // Set to 01 (Output)

    // 3. Configure Button (PA0) as Input (Default is input, but good to be explicit)
    GPIOA->MODER &= ~(0x00000003); // Clear bits 0-1
}

void BSP_Delay(uint32_t count) {
    while(count--) __asm("nop");
}
```

#### Step 2: Driver Layer (`led_driver.c`)
Abstracts the LEDs.

```c
#include "led_driver.h"
#include "stm32f4xx.h"

void LED_Set(uint8_t led_mask) {
    // Clear all LEDs (Bits 12-15)
    GPIOD->BSRR = (0xF000 << 16);
    
    // Set requested LEDs
    // Map mask bit 0 -> PD12, bit 1 -> PD13, etc.
    uint32_t set_bits = 0;
    if (led_mask & 1) set_bits |= (1 << 12);
    if (led_mask & 2) set_bits |= (1 << 13);
    if (led_mask & 4) set_bits |= (1 << 14);
    if (led_mask & 8) set_bits |= (1 << 15);
    
    GPIOD->BSRR = set_bits;
}
```

#### Step 3: Driver Layer (`button_driver.c`)
Handles debouncing.

```c
#include "button_driver.h"
#include "stm32f4xx.h"
#include "bsp.h"

uint8_t Button_IsPressed(void) {
    if (GPIOA->IDR & (1 << 0)) {
        BSP_Delay(50000); // Simple debounce (~50ms)
        if (GPIOA->IDR & (1 << 0)) {
            // Wait for release
            while(GPIOA->IDR & (1 << 0));
            return 1;
        }
    }
    return 0;
}
```

#### Step 4: Application Layer (`main.c`)
Implements the state machine.

```c
#include "bsp.h"
#include "led_driver.h"
#include "button_driver.h"

typedef enum {
    PATTERN_OFF = 0,
    PATTERN_ROTATE,
    PATTERN_BLINK_ALL,
    PATTERN_SOS,
    PATTERN_MAX
} Pattern_t;

int main(void) {
    BSP_Init();
    
    Pattern_t current_pattern = PATTERN_OFF;
    uint8_t step = 0;

    while (1) {
        // Check Button
        if (Button_IsPressed()) {
            current_pattern++;
            if (current_pattern >= PATTERN_MAX) {
                current_pattern = PATTERN_OFF;
            }
            step = 0; // Reset step for new pattern
        }

        // Execute Pattern
        switch (current_pattern) {
            case PATTERN_OFF:
                LED_Set(0);
                break;

            case PATTERN_ROTATE:
                LED_Set(1 << step); // 1, 2, 4, 8
                step++;
                if (step > 3) step = 0;
                BSP_Delay(200000);
                break;

            case PATTERN_BLINK_ALL:
                LED_Set(0xF);
                BSP_Delay(200000);
                LED_Set(0);
                BSP_Delay(200000);
                break;

            case PATTERN_SOS:
                // Complex logic...
                // S (...)
                for(int i=0; i<3; i++) { LED_Set(0xF); BSP_Delay(100000); LED_Set(0); BSP_Delay(100000); }
                BSP_Delay(200000);
                // O (---)
                for(int i=0; i<3; i++) { LED_Set(0xF); BSP_Delay(300000); LED_Set(0); BSP_Delay(100000); }
                BSP_Delay(200000);
                // S (...)
                for(int i=0; i<3; i++) { LED_Set(0xF); BSP_Delay(100000); LED_Set(0); BSP_Delay(100000); }
                BSP_Delay(1000000); // Pause
                break;
        }
    }
}
```

---

## 🔬 Lab Exercise: Lab 7.1 - Code Review & Refactoring

### 1. Lab Objectives
- Analyze the provided code for "Bad Practices".
- Refactor it to meet "Good Practices".

### 2. Step-by-Step Guide

#### Phase A: The "Bad" Code
```c
// main.c
int main() {
    *(int*)0x40023830 |= 8; // Magic numbers!
    *(int*)0x40020C00 = 0x55000000;
    while(1) {
        *(int*)0x40020C14 = 0xF000;
        for(int i=0; i<100000; i++); // Blocking delay
        *(int*)0x40020C14 = 0;
        for(int i=0; i<100000; i++);
    }
}
```

#### Phase B: The Critique
1.  **Magic Numbers:** What is `0x40023830`? What is `8`?
2.  **No Volatile:** The compiler might optimize out the loop.
3.  **No Abstraction:** Hard to read or port.
4.  **Blocking Delay:** CPU burns power doing nothing.

#### Phase C: Refactoring
Rewrite using the `BSP` and `Driver` approach shown above.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Interrupt-Driven Button
- **Goal:** Replace the polling `Button_IsPressed()` with an External Interrupt (EXTI).
- **Task:** Configure NVIC and EXTI Line 0. In the ISR, set a global flag `volatile uint8_t button_pressed = 1;`.

### Lab 3: PWM Brightness (Software)
- **Goal:** Dim the LEDs instead of just On/Off.
- **Task:** Implement Software PWM inside the main loop (fast switching) to control brightness.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Button Bouncing
*   **Symptom:** One press skips multiple patterns.
*   **Cause:** Mechanical contacts vibrate, creating multiple electrical pulses.
*   **Solution:** Hardware capacitor or Software delay (Debouncing).

#### 2. Logic Errors in State Machine
*   **Symptom:** Stuck in one state.
*   **Cause:** Missing `break` in switch-case, or incorrect transition logic.
*   **Solution:** Use a debugger to step through the `switch` statement.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **MISRA C:**
    *   Rule 11.4: A cast should not be performed between a pointer to object type and a different pointer to object type (Strict aliasing).
    *   Rule 16.1: All switch statements should be well-formed (always have a `default` case).

### Project Organization
- Keep `.h` files for public interfaces.
- Keep `.c` files for implementation.
- Use `#ifndef HEADER_H` guards.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we separate BSP from Application?
    *   **A:** To make the application code portable. If we switch chips, we only rewrite the BSP.
2.  **Q:** What is "Debouncing"?
    *   **A:** Filtering out the mechanical noise of a switch to ensure a single clean signal.

### Challenge Task
> **Task:** Add a "Speed Control" feature. Holding the button for > 2 seconds changes the blinking speed of the current pattern.

---

## 📚 Further Reading & References
- [Test Driven Development for Embedded C](https://www.amazon.com/Driven-Development-Embedded-Pragmatic-Programmers/dp/193435662X)
- [Making Embedded Systems (O'Reilly)](https://www.oreilly.com/library/view/making-embedded-systems/9781449302139/)

---
