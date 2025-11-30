# Day 15: GPIO Deep Dive
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
1.  **Analyze** the internal schematic of a GPIO pin (Push-Pull vs Open-Drain, Pull-up/down).
2.  **Configure** GPIO speed settings and understand their impact on EMI and signal integrity.
3.  **Implement** Alternate Functions (AF) to connect pins to internal peripherals (UART, SPI, Timers).
4.  **Design** a software debouncing algorithm for noisy mechanical switches.
5.  **Write** a driver for a 4x4 Keypad Matrix.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Breadboard, Jumper Wires
    *   4x4 Keypad (or simulate with wires)
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 5 (Memory Map)
    *   Day 6 (Structs/Enums)
*   **Datasheets:**
    *   [STM32F407 Reference Manual (GPIO Section)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The GPIO Architecture

#### 1.1 Internal Schematic
Each GPIO pin is complex. It contains:
*   **Output Driver:** Two MOSFETs (P-MOS and N-MOS).
*   **Input Driver:** Schmitt Trigger (to clean up noise).
*   **Resistors:** Internal Pull-Up (~40kΩ) and Pull-Down (~40kΩ).
*   **Protection Diodes:** Clamping diodes to VDD and VSS to protect against ESD (within limits).

#### 1.2 Output Modes
1.  **Push-Pull:** Uses both P-MOS (Source) and N-MOS (Sink).
    *   *Logic 1:* P-MOS ON, N-MOS OFF. Pin driven to VDD.
    *   *Logic 0:* P-MOS OFF, N-MOS ON. Pin driven to GND.
    *   *Use Case:* LEDs, Digital Logic.
2.  **Open-Drain:** Uses only N-MOS.
    *   *Logic 0:* N-MOS ON. Pin driven to GND.
    *   *Logic 1:* N-MOS OFF. Pin floats (High-Z). Requires external Pull-Up resistor.
    *   *Use Case:* I2C Bus, Level Shifting (5V tolerance), Wired-OR logic.

#### 1.3 Speed and Slew Rate
The `OSPEEDR` register controls the rise/fall time of the signal.
*   **Low Speed:** Slow edges. Reduces EMI (Electromagnetic Interference) and ringing.
*   **High Speed:** Fast edges. Necessary for high-frequency signals (SPI, SDIO).
*   **Rule:** Always use the *lowest* speed that works for your application to minimize noise.

### 🔹 Part 2: Alternate Functions (AF)

Pins are multiplexed. PA2 can be GPIO, USART2_TX, TIM2_CH3, or ADC1_IN2.
*   **AFR (Alternate Function Register):** Two 32-bit registers (`AFRL` and `AFRH`) select the function (AF0 to AF15) for each pin.
    *   `AFRL`: Pins 0-7.
    *   `AFRH`: Pins 8-15.
    *   4 bits per pin.

### 🔹 Part 3: Input Handling

#### 3.1 Floating vs. Pull-Up/Down
*   **Floating:** High impedance. Susceptible to noise. Good for external active drivers.
*   **Pull-Up/Down:** Defines the state when no external signal is present. Essential for buttons.

#### 3.2 Schmitt Trigger
Hysteresis prevents the input from toggling rapidly if the voltage hovers around the threshold (e.g., 1.6V).

---

## 💻 Implementation: Keypad Matrix Driver

> **Instruction:** We will implement a driver for a 4x4 Matrix Keypad.
> *   **Rows:** Output (Push-Pull).
> *   **Cols:** Input (Pull-Up).

### 🛠️ Hardware/System Configuration
*   **Rows (R1-R4):** PE2, PE3, PE4, PE5.
*   **Cols (C1-C4):** PE6, PE7, PE8, PE9.

### 👨‍💻 Code Implementation

#### Step 1: Initialization (`keypad.c`)

```c
#include "stm32f4xx.h"

void Keypad_Init(void) {
    // Enable GPIOE Clock
    RCC->AHB1ENR |= (1 << 4);

    // Configure Rows (PE2-PE5) as Output
    GPIOE->MODER &= ~(0x00000FF0); // Clear
    GPIOE->MODER |=  (0x00000550); // Set to 01 (Output)

    // Configure Cols (PE6-PE9) as Input
    GPIOE->MODER &= ~(0x000FF000); // Clear (Input is 00)

    // Enable Pull-Up for Cols
    GPIOE->PUPDR &= ~(0x000FF000); // Clear
    GPIOE->PUPDR |=  (0x00055000); // Set to 01 (Pull-Up)
    
    // Set all Rows High initially (Idle state)
    GPIOE->ODR |= (0xF << 2);
}
```

#### Step 2: Scanning Logic
Algorithm:
1.  Set all Rows HIGH.
2.  Set Row 1 LOW.
3.  Check Cols. If Col X is LOW, then Key(1, X) is pressed.
4.  Set Row 1 HIGH.
5.  Repeat for Row 2, 3, 4.

```c
char Keypad_Scan(void) {
    // Map: 4x4
    char keys[4][4] = {
        {'1', '2', '3', 'A'},
        {'4', '5', '6', 'B'},
        {'7', '8', '9', 'C'},
        {'*', '0', '#', 'D'}
    };

    for (int row = 0; row < 4; row++) {
        // Set current Row LOW, others HIGH
        GPIOE->ODR |= (0xF << 2); // All High
        GPIOE->ODR &= ~(1 << (row + 2)); // Current Low

        // Small delay for signal to settle (capacitance)
        for(volatile int i=0; i<100; i++);

        // Check Cols
        for (int col = 0; col < 4; col++) {
            // Read PE(6+col)
            if (!(GPIOE->IDR & (1 << (col + 6)))) {
                // Key Pressed!
                // Wait for release (simple debounce)
                while(!(GPIOE->IDR & (1 << (col + 6))));
                return keys[row][col];
            }
        }
    }
    return 0; // No key
}
```

#### Step 3: Main Loop
```c
#include <stdio.h>

int main(void) {
    Keypad_Init();
    
    // Init UART for debug output (Assume UART_Init exists)
    
    while(1) {
        char key = Keypad_Scan();
        if (key != 0) {
            printf("Key Pressed: %c\n", key);
        }
    }
}
```

---

## 🔬 Lab Exercise: Lab 15.1 - Open Drain & I2C Simulation

### 1. Lab Objectives
- Understand why Open-Drain is needed for shared buses.
- Simulate a "Wired-AND" logic.

### 2. Step-by-Step Guide

#### Phase A: Setup
1.  Configure PD12 and PD13 as **Open-Drain Output**.
2.  Connect external Pull-Up resistors (4.7k) from pins to 3.3V.
3.  Connect PD12 and PD13 together with a wire.

#### Phase B: Logic Test
1.  Set PD12 High, PD13 High. Measure voltage (Should be 3.3V).
2.  Set PD12 Low, PD13 High. Measure voltage (Should be 0V).
    *   *Explanation:* PD12 pulls the line to Ground. PD13 is floating (High-Z), so it doesn't fight PD12.
3.  Set PD12 High, PD13 Low. Measure voltage (Should be 0V).

#### Phase C: The Conflict (Push-Pull)
**WARNING:** Do NOT do this physically if you value your board.
*   If PD12 was Push-Pull High (3.3V) and PD13 was Push-Pull Low (0V), and you connected them:
    *   Current flows from PD12 -> PD13.
    *   Short circuit! High current -> Heat -> Dead MCU.

### 3. Verification
This lab confirms why I2C uses Open-Drain. Multiple devices can pull the line low without shorting out devices that want it high.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Debouncing Algorithm
- **Goal:** Implement a robust non-blocking debounce.
- **Task:**
    1.  Read button state every 5ms (SysTick).
    2.  Shift state into a variable: `state = (state << 1) | raw_input`.
    3.  If `state == 0xFFFF` (Stable High) or `0x0000` (Stable Low), accept the value.
    4.  Ignore intermediate states (bouncing).

### Lab 3: Measuring Slew Rate
- **Goal:** Observe the effect of `OSPEEDR`.
- **Task:**
    1.  Toggle a pin as fast as possible in a loop.
    2.  Set Speed to Low, Medium, High, Very High.
    3.  Use an Oscilloscope to measure the Rise Time (10% to 90%).
    4.  Observe ringing/overshoot on Very High speed.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Pin Not Toggling
*   **Cause:**
    *   Clock not enabled (`RCC->AHB1ENR`).
    *   Pin stuck in Alternate Function mode.
    *   Pin locked (`LCKR` register used previously).
*   **Solution:** Check registers in Debugger View.

#### 2. Keypad Ghosting
*   **Symptom:** Pressing 3 keys causes a 4th "Phantom" key to appear.
*   **Cause:** Lack of diodes in the matrix. Current flows backwards through other switches.
*   **Solution:** Add diodes in series with each switch (Hardware fix) or software workarounds (limit to 2 keys).

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **BSRR vs ODR:** Always use `BSRR` (Bit Set/Reset Register) to change pin state. It is atomic and prevents Read-Modify-Write issues that can happen with `ODR` if an interrupt occurs mid-operation.

### Code Quality
- **Locking:** For safety-critical signals (e.g., Motor Enable), use the GPIO Lock Mechanism (`LCKR`) to prevent software bugs from changing the pin configuration during runtime.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `GPIO_BSRR` and `GPIO_ODR`?
    *   **A:** `ODR` stores the output data. `BSRR` is a "write-only" port that allows atomic setting/resetting of individual bits in `ODR` without reading the whole register first.
2.  **Q:** Why do we need Alternate Functions?
    *   **A:** Because there are more internal peripherals (Timers, UARTs, SPIs) than physical pins. We must choose which peripheral gets to use the pin.

### Challenge Task
> **Task:** Implement a "GPIO Toggle" using the `LCKR` register. Lock the configuration of PD12. Then try to change it to Input mode in code. Verify that the change is ignored.

---

## 📚 Further Reading & References
- [Understanding GPIO Settings (Push-Pull, Open-Drain)](https://learn.sparkfun.com/tutorials/switch-basics/all)
- [STM32 GPIO Application Note (AN4899)](https://www.st.com/resource/en/application_note/dm00315319-stm32-gpio-configuration-for-hardware-settings-and-low-power-consumption-stmicroelectronics.pdf)

---
