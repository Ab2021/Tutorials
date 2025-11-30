# Day 51: Power Consumption Analysis & Optimization
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
1.  **Measure** dynamic and static power consumption using Shunt Resistors and Profilers.
2.  **Optimize** GPIO configurations (Analog Mode) to eliminate leakage current.
3.  **Implement** Clock Gating and Voltage Scaling to reduce dynamic power.
4.  **Profile** code energy usage (Energy = Power × Time).
5.  **Achieve** the lowest possible Stop Mode current on the STM32F4.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Multimeter (µA range) or Power Profiler (e.g., Nordic PPK2, STLink-V3PWR).
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   STM32CubeMX (for clock tree visualization, optional).
*   **Prior Knowledge:**
    *   Day 50 (Low Power Modes)
    *   Day 15 (GPIO)
*   **Datasheets:**
    *   [STM32F407 Datasheet (Electrical Characteristics)](https://www.st.com/resource/en/datasheet/stm32f407vg.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Where does the current go?
In a microcontroller, current is consumed by:
1.  **CPU Core:** Processing instructions. $\propto$ Frequency.
2.  **Peripherals:** Timers, ADC, UART. $\propto$ Active state.
3.  **GPIOs:** Driving loads (LEDs) or Floating Inputs (Schmitt Trigger toggling).
4.  **Leakage:** Transistors not fully off.

### 🔹 Part 2: Optimization Strategies
*   **Clock Gating:** Disable `RCC` clocks for unused peripherals.
*   **GPIO Management:**
    *   **Unused Pins:** Configure as **Analog Mode**. This disables the Schmitt Trigger and disconnects the input buffer, preventing noise-induced switching current.
    *   **Floating Inputs:** Avoid them. Use Pull-Up/Down.
*   **Voltage Scaling:** The STM32F4 allows scaling the internal 1.2V regulator.
    *   Scale 1 (High Perf): Up to 168 MHz.
    *   Scale 2: Up to 144 MHz.
    *   Scale 3 (Low Power): Up to 120 MHz. Lower voltage = Lower leakage.
*   **Flash Settings:**
    *   **Instruction Cache (I-Cache):** Reduces Flash access (which is power hungry).
    *   **Prefetch:** Can increase power if branch prediction fails often.

---

## 💻 Implementation: The "Perfect" Low Power Config

> **Instruction:** We will write a function `LowPower_Config()` that prepares the MCU for the absolute minimum power consumption in Stop Mode.

### 👨‍💻 Code Implementation

#### Step 1: GPIO Optimization
```c
#include "stm32f4xx.h"

void GPIO_LowPower_Config(void) {
    // Enable all GPIO Clocks temporarily to configure them
    RCC->AHB1ENR |= 0x1FF; // GPIOA to GPIOI
    
    GPIO_TypeDef* ports[] = {GPIOA, GPIOB, GPIOC, GPIOD, GPIOE, GPIOF, GPIOG, GPIOH, GPIOI};
    
    for (int i = 0; i < 9; i++) {
        // Set MODER to 11 (Analog Mode) for all pins
        ports[i]->MODER = 0xFFFFFFFF;
        
        // Set PUPDR to 00 (No Pull)
        ports[i]->PUPDR = 0x00000000;
        
        // Optimize: If a pin is connected to an external Pull-Up (e.g., I2C), 
        // keeping it Analog is fine (High Z).
        // If connected to a switch that is Open, Analog is fine.
    }
    
    // Disable GPIO Clocks
    RCC->AHB1ENR &= ~0x1FF;
}
```

#### Step 2: Peripheral Clock Gating
```c
void Peripheral_Off(void) {
    // Disable all peripheral clocks
    RCC->AHB1ENR = 0;
    RCC->AHB2ENR = 0;
    RCC->AHB3ENR = 0;
    RCC->APB1ENR = 0;
    RCC->APB2ENR = 0;
    
    // Note: We need PWR clock to enter Stop Mode
    RCC->APB1ENR |= (1 << 28); 
}
```

#### Step 3: Flash Power Down
```c
void Flash_DeepSleep(void) {
    // Enable Flash Power Down in Stop Mode
    PWR->CR |= (1 << 9); // FPDS
}
```

#### Step 4: Main Optimization Routine
```c
int main(void) {
    HAL_Init();
    
    // 1. Run Mode Optimization
    // Reduce Clock if full speed not needed
    // SystemClock_Config_HSI(); // Run at 16 MHz instead of 168 MHz
    
    // 2. Prepare for Stop Mode
    GPIO_LowPower_Config();
    Peripheral_Off();
    Flash_DeepSleep();
    
    // 3. Enter Stop Mode
    // Regulator in Low Power Mode
    PWR->CR |= (1 << 0); // LPDS
    
    // Enter
    SCB->SCR |= (1 << 2);
    __WFI();
    
    while(1);
}
```

---

## 🔬 Lab Exercise: Lab 51.1 - The µA Hunt

### 1. Lab Objectives
- Reduce the Stop Mode current of the Discovery Board from ~15mA (default) to < 1mA.

### 2. Step-by-Step Guide

#### Phase A: Baseline
1.  Flash a blank `main() { while(1); }`.
2.  Measure Current: ~60 mA (Run Mode).

#### Phase B: Stop Mode (Naive)
1.  Call `__WFI()` with `SLEEPDEEP`.
2.  Measure Current: ~15 mA.
3.  **Why so high?** The Discovery Board has external components (ST-Link, Audio DAC, Accelerometer) connected to GPIOs. Floating pins or driven pins causing current leaks.

#### Phase C: Optimization
1.  Call `GPIO_LowPower_Config()`.
2.  **Critical:** The ST-Link is connected to PA13/PA14. If you set them to Analog, you lose Debug access!
    *   **Fix:** Exclude PA13/PA14 from the Analog loop.
3.  **Critical:** The Audio DAC (CS43L22) Reset pin (PD4) must be held Low (Reset) to keep it in low power. If you set PD4 to Analog (Floating), the DAC might wake up and consume current.
    *   **Fix:** Drive PD4 Low (Output).
4.  Measure Current: Should drop to ~1-2 mA.

### 3. Verification
Getting below 1mA on a Dev Board is hard because of the LDO/Regulator efficiency and LEDs (Power LED). On a custom PCB, this code would yield ~15 µA.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Dynamic Voltage Scaling
- **Goal:** Switch between Scale 1 and Scale 3.
- **Task:**
    1.  Read `PWR->CR` bit 14 (VOS).
    2.  Set VOS = 0 (Scale 3).
    3.  Limit Clock to 120 MHz.
    4.  Measure Run Mode current reduction.

### Lab 3: ULPMark (Concept)
- **Goal:** Calculate Energy Efficiency.
- **Task:**
    1.  Perform a task (e.g., Calculate FFT).
    2.  Measure Time ($t$) and Current ($I$).
    3.  Energy $E = V \times I \times t$.
    4.  Compare: Running at 168 MHz (Fast, High Power) vs 16 MHz (Slow, Low Power).
    5.  **Result:** Usually, "Race to Sleep" (Run fast, then sleep) is better than running slow for a long time, because background static power is constant.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "I bricked my board"
*   **Cause:** Setting SWD pins (PA13/PA14) to Analog or Output.
*   **Solution:** Connect `NRST` pin to Ground. Connect ST-Link. Release `NRST` exactly when you click "Connect" in STM32CubeProgrammer (Connect under Reset).

#### 2. Current is still high
*   **Cause:** Floating Input pins.
*   **Solution:** Check every single pin. If it's not connected, make it Analog. If it's connected to a pull-up, make it Open Drain High (or Analog).

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Defensive Programming:** Before entering Stop Mode, verify that all busy flags (e.g., `SPI_SR_BSY`) are clear. Cutting the clock while a peripheral is busy can freeze the logic.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why does Analog Mode save power?
    *   **A:** It disconnects the input Schmitt Trigger. A floating voltage on a digital input can cause the Schmitt Trigger to toggle rapidly (oscillation) or stay in the linear region, causing high shoot-through current.
2.  **Q:** What is "Race to Sleep"?
    *   **A:** The strategy of running the CPU at max speed to finish the task as quickly as possible, maximizing the time spent in the lowest power mode.

### Challenge Task
> **Task:** Implement a "Battery Monitor". Wake up every hour. Enable ADC. Read V_Battery. If < 3.0V, flash Red LED. If > 3.0V, flash Green. Return to Standby. Average current must be < 50 µA.

---

## 📚 Further Reading & References
- [Ultra-low-power STM32L4 Series (Comparison)](https://www.st.com/en/microcontrollers-microprocessors/stm32l4-series.html)

---
