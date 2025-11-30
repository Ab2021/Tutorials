# Day 38: Motor Control (DC/Stepper)
## Phase 1: Core Embedded Engineering Foundations | Week 6: Sensors and Actuators

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
1.  **Explain** the H-Bridge circuit for bidirectional DC motor control.
2.  **Implement** PWM speed control using STM32 Timers.
3.  **Drive** a Stepper Motor (Bipolar/Unipolar) using full-step and half-step sequences.
4.  **Interface** with common motor drivers (L298N, A4988).
5.  **Protect** the MCU from back-EMF spikes.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   L298N H-Bridge Module (or similar)
    *   DC Motor (Small 5V/12V)
    *   Stepper Motor (28BYJ-48 or NEMA 17)
    *   External Power Supply (Do not power motors from USB!)
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 17 (PWM)
    *   Day 15 (GPIO)
*   **Datasheets:**
    *   [L298N Datasheet](https://www.sparkfun.com/datasheets/Robotics/L298_H_Bridge.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: DC Motors & H-Bridges
*   **DC Motor:** Speed $\propto$ Voltage. Direction depends on Polarity.
*   **H-Bridge:** A circuit with 4 switches (Transistors/MOSFETs) arranged in an 'H' shape.
    *   **Forward:** Close Q1 (Top Left) and Q4 (Bottom Right).
    *   **Reverse:** Close Q3 (Top Right) and Q2 (Bottom Left).
    *   **Brake:** Close Q2 and Q4 (Short circuit back-EMF).
    *   **Shoot-Through:** Closing Q1 and Q2 simultaneously shorts Power to Ground -> Magic Smoke.

### 🔹 Part 2: Stepper Motors
*   **Concept:** Moves in discrete steps (e.g., 1.8 deg/step).
*   **Bipolar:** 4 wires (2 coils). Needs H-Bridge to reverse current.
*   **Unipolar:** 5/6 wires (Center tapped coils). Easier to drive (ULN2003).
*   **Sequences:**
    *   **Wave Drive:** Energize 1 coil at a time. Low torque.
    *   **Full Step:** Energize 2 coils. High torque.
    *   **Half Step:** Alternate 1 and 2 coils. Smoother, double resolution.

---

## 💻 Implementation: DC Motor Control (L298N)

> **Instruction:** Control Speed (PWM) and Direction (GPIO).

### 🛠️ Hardware/System Configuration
*   **IN1:** PA0 (GPIO Output).
*   **IN2:** PA1 (GPIO Output).
*   **ENA:** PA2 (TIM2_CH3 PWM).
*   **Power:** 12V to L298N, GND common with STM32.

### 👨‍💻 Code Implementation

#### Step 1: Initialization (`motor.c`)

```c
#include "stm32f4xx.h"

void Motor_Init(void) {
    // 1. GPIO Init (PA0, PA1)
    RCC->AHB1ENR |= (1 << 0);
    GPIOA->MODER |= (1 << 0) | (1 << 2); // Output
    
    // 2. PWM Init (PA2 - TIM2_CH3)
    GPIOA->MODER |= (2 << 4); // AF
    GPIOA->AFR[0] |= (1 << 8); // AF1 (TIM2)
    
    RCC->APB1ENR |= (1 << 0); // TIM2
    TIM2->PSC = 83; // 1 MHz Timer
    TIM2->ARR = 999; // 1 kHz PWM
    
    // PWM Mode 1 on CH3
    TIM2->CCMR2 |= (6 << 4); 
    TIM2->CCER |= (1 << 8); // Enable Output
    TIM2->CR1 |= (1 << 0); // Enable Timer
}
```

#### Step 2: Control Functions
```c
void Motor_SetSpeed(int speed) {
    // Speed: -100 to +100
    
    if (speed > 0) {
        // Forward: IN1=1, IN2=0
        GPIOA->ODR |= (1 << 0);
        GPIOA->ODR &= ~(1 << 1);
        TIM2->CCR3 = speed * 10; // Scale to 0-1000
    } else if (speed < 0) {
        // Reverse: IN1=0, IN2=1
        GPIOA->ODR &= ~(1 << 0);
        GPIOA->ODR |= (1 << 1);
        TIM2->CCR3 = (-speed) * 10;
    } else {
        // Stop: IN1=0, IN2=0 (Coast) or 1,1 (Brake)
        GPIOA->ODR &= ~((1 << 0) | (1 << 1));
        TIM2->CCR3 = 0;
    }
}
```

---

## 💻 Implementation: Stepper Motor (Wave Drive)

> **Instruction:** Drive a Bipolar Stepper (4 wires: A+, A-, B+, B-).

### 🛠️ Hardware/System Configuration
*   **A+:** PC0
*   **A-:** PC1
*   **B+:** PC2
*   **B-:** PC3

### 👨‍💻 Code Implementation

#### Step 1: Step Sequence
```c
// Full Step Sequence (2-Phase On)
const uint8_t step_seq[4] = {
    0b1010, // A+ B+
    0b0110, // A- B+
    0b0101, // A- B-
    0b1001  // A+ B-
};

void Stepper_Init(void) {
    RCC->AHB1ENR |= (1 << 2); // GPIOC
    GPIOC->MODER |= 0x55; // PC0-3 Output
}

void Stepper_Step(int steps, int delay_ms) {
    static int step_idx = 0;
    
    for (int i = 0; i < abs(steps); i++) {
        if (steps > 0) step_idx++;
        else step_idx--;
        
        // Wrap around
        if (step_idx > 3) step_idx = 0;
        if (step_idx < 0) step_idx = 3;
        
        // Output to Pins
        // Clear PC0-3
        GPIOC->ODR &= ~(0xF);
        // Set new pattern
        GPIOC->ODR |= step_seq[step_idx];
        
        Delay_ms(delay_ms);
    }
}
```

---

## 🔬 Lab Exercise: Lab 38.1 - Speed Ramp

### 1. Lab Objectives
- Smoothly accelerate and decelerate the DC motor.
- Avoid current spikes.

### 2. Step-by-Step Guide

#### Phase A: Logic
1.  Start at Speed 0.
2.  Increment Speed by 1 every 10ms until 100.
3.  Hold for 2 seconds.
4.  Decrement Speed by 1 every 10ms until 0.

#### Phase B: Implementation
```c
for (int s = 0; s <= 100; s++) {
    Motor_SetSpeed(s);
    Delay_ms(10);
}
```

### 3. Verification
The motor should whine (PWM freq) and spin up smoothly.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Microstepping (Concept)
- **Goal:** Use PWM to simulate sine waves on Stepper coils.
- **Task:**
    1.  Instead of ON/OFF, use PWM on A+, A-, B+, B-.
    2.  Set A = `sin(theta)`, B = `cos(theta)`.
    3.  Increment theta slowly.
    4.  Result: Super smooth motion, high resolution.

### Lab 3: Current Sensing
- **Goal:** Detect stall.
- **Task:**
    1.  L298N has "CSA/CSB" pins (Current Sense). Connect to ADC.
    2.  If ADC value > Threshold (High Current), Stop Motor.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Motor Hums but doesn't turn
*   **Cause:** PWM duty cycle too low (Stiction).
*   **Cause:** Power supply too weak.
*   **Cause:** Stepper sequence wrong (coils wired incorrectly).

#### 2. MCU Resets when Motor Starts
*   **Cause:** Inductive spike / Brownout.
*   **Solution:** Use separate power supplies. Add Flyback diodes (if not in driver). Add large capacitor (100uF) near motor driver.

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **Acceleration Profiles:** For steppers, use "S-Curve" acceleration instead of linear. This reduces vibration and allows higher top speeds.

### Code Quality
- **Timer Interrupts:** For steppers, don't use `Delay_ms`. Use a Timer Interrupt to generate steps. Change the ARR to change speed. This allows the main loop to do other things.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the purpose of the "Dead Time" in an H-Bridge?
    *   **A:** A small delay when switching direction to ensure Q1 turns off before Q2 turns on, preventing Shoot-Through (Short Circuit).
2.  **Q:** Why do steppers get hot even when stopped?
    *   **A:** Because current is constantly flowing through the coils to maintain "Holding Torque".

### Challenge Task
> **Task:** Implement a "Stepper Motor Controller" with UART. Commands: `MOVE 100`, `SPEED 50`. Use a Timer Interrupt for step generation so the UART remains responsive while moving.

---

## 📚 Further Reading & References
- [Stepping Motors Fundamentals (Microchip AN907)](https://ww1.microchip.com/downloads/en/AppNotes/00907a.pdf)

---
