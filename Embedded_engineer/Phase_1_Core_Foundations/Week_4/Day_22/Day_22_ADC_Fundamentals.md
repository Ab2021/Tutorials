# Day 22: ADC Fundamentals
## Phase 1: Core Embedded Engineering Foundations | Week 4: Analog Interfacing

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
1.  **Explain** the Successive Approximation Register (SAR) architecture used in STM32 ADCs.
2.  **Calculate** the sampling time and total conversion time based on clock cycles.
3.  **Configure** the ADC for Single Conversion Mode using Polling.
4.  **Convert** raw 12-bit ADC values into physical voltage units (Volts).
5.  **Design** a voltage divider circuit to measure signals higher than 3.3V safely.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Potentiometer (10kΩ)
    *   Breadboard, Jumper Wires
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 13 (Clocks)
    *   Day 5 (Memory Map)
*   **Datasheets:**
    *   [STM32F407 Reference Manual (ADC Section)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: ADC Architecture (SAR)

#### 1.1 How SAR Works
The STM32 uses a **Successive Approximation Register (SAR)** ADC. It works like a binary search algorithm.
1.  **Sample & Hold:** A capacitor charges to match the input voltage.
2.  **Bit 11 (MSB):** The DAC sets internal voltage to $V_{REF}/2$. Comparator checks: Is Input > Internal?
    *   Yes: Keep Bit 11 = 1.
    *   No: Set Bit 11 = 0.
3.  **Bit 10:** DAC sets voltage to $V_{REF}/4$ (plus previous result). Check again.
4.  Repeat for all 12 bits.

**Key Characteristic:** The conversion time is fixed (in cycles) regardless of the input voltage.

#### 1.2 Resolution & Quantization
*   **Resolution:** 12-bit (0 to 4095).
*   **Reference Voltage ($V_{REF}$):** Usually connected to $V_{DD}$ (3.3V).
*   **LSB Size (Step):** $V_{LSB} = \frac{V_{REF}}{2^{12}} = \frac{3.3V}{4096} \approx 0.8 mV$.
*   **Formula:** $Voltage = \frac{ADC\_Value \times V_{REF}}{4095}$.

### 🔹 Part 2: Timing & Sampling

#### 2.1 The Sampling Capacitor
The internal capacitor ($C_{SH}$) needs time to charge. If the source impedance ($R_{AIN}$) is high (e.g., a large resistor divider), it takes longer to charge.
*   **Sampling Time ($t_S$):** Programmable (3 to 480 cycles).
*   **Conversion Time ($t_C$):** Fixed 12 cycles for 12-bit resolution.
*   **Total Time:** $T_{total} = t_S + t_C$.

**Example:**
*   ADC Clock = 30 MHz.
*   Sample Time = 3 cycles.
*   Total = 3 + 12 = 15 cycles.
*   Time = $15 / 30 MHz = 0.5 \mu s$ (2 MSPS).

**Impedance Rule:** For $t_S = 3$ cycles, $R_{AIN}$ must be $< 0.5 k\Omega$. If $R_{AIN}$ is 10k, you MUST increase sampling time, or the reading will be wrong (too low).

---

## 💻 Implementation: Reading a Potentiometer

> **Instruction:** We will configure ADC1 on PA1 to read a potentiometer value in a blocking loop.

### 🛠️ Hardware/System Configuration
*   **Pin:** PA1 (ADC1_IN1).
*   **Potentiometer:** Pin 1 -> 3.3V, Pin 3 -> GND, Wiper -> PA1.

### 👨‍💻 Code Implementation

#### Step 1: Initialization (`adc.c`)

```c
#include "stm32f4xx.h"

void ADC1_Init(void) {
    // 1. Enable Clocks
    RCC->AHB1ENR |= (1 << 0); // GPIOA
    RCC->APB2ENR |= (1 << 8); // ADC1

    // 2. Configure PA1 as Analog
    GPIOA->MODER |= (3 << 2); // 11: Analog Mode

    // 3. Configure ADC Common Control Register (CCR)
    // PCLK2 div 4 (ADCCLK = 84MHz / 4 = 21MHz)
    ADC->CCR |= (1 << 16); 

    // 4. Configure ADC1
    // Resolution: 12-bit (Default)
    // Mode: Single Conversion (Default)
    // Data Align: Right (Default)
    
    // 5. Configure Sequence
    // SQR1: L = 0 (1 conversion)
    // SQR3: SQ1 = 1 (Channel 1)
    ADC1->SQR3 = 1;

    // 6. Configure Sample Time
    // SMPR2: Channel 1. Set to 84 cycles (Bits 3-5 = 100)
    // Safe for 10k Potentiometer
    ADC1->SMPR2 |= (4 << 3);

    // 7. Enable ADC
    ADC1->CR2 |= (1 << 0); // ADON
}
```

#### Step 2: Read Function
```c
uint16_t ADC1_Read(void) {
    // 1. Start Conversion
    ADC1->CR2 |= (1 << 30); // SWSTART

    // 2. Wait for Conversion Complete (EOC)
    while(!(ADC1->SR & (1 << 1)));

    // 3. Read Data
    return ADC1->DR;
}
```

#### Step 3: Main Loop
```c
#include <stdio.h>

int main(void) {
    // Init UART...
    ADC1_Init();
    
    while(1) {
        uint16_t raw = ADC1_Read();
        
        // Convert to Voltage (Float)
        // Note: Use integer math in production if possible
        float voltage = (raw * 3.3f) / 4095.0f;
        
        printf("Raw: %d, Voltage: %.2f V\r", raw, voltage);
        
        for(int i=0; i<500000; i++); // Delay
    }
}
```

---

## 🔬 Lab Exercise: Lab 22.1 - The Voltage Divider

### 1. Lab Objectives
- Measure a 5V signal using a 3.3V ADC.
- Understand the effect of resistor tolerance.

### 2. Step-by-Step Guide

#### Phase A: Theory
You cannot put 5V into STM32 pins (most Analog pins are NOT 5V tolerant).
Divider: $V_{out} = V_{in} \times \frac{R_2}{R_1 + R_2}$.
Target: $V_{out} = 3.3V$ when $V_{in} = 5V$.
Ratio: $0.66$.
Values: $R_1 = 10k\Omega, R_2 = 20k\Omega$. (Ratio = 0.666).

#### Phase B: Setup
1.  Connect 5V source to $R_1$.
2.  Connect $R_1$ to $R_2$ and PA1.
3.  Connect $R_2$ to GND.

#### Phase C: Coding
Modify the conversion formula:
`float actual_voltage = adc_voltage * ((R1 + R2) / R2);`

### 3. Verification
Measure the 5V source with a multimeter. Compare with the printf output.
*   **Error Source:** Resistors are usually +/- 5%. A 10k might be 9.5k. This introduces measurement error.
*   **Fix:** Measure exact resistance and hardcode it, or use 1% resistors.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Internal Temperature Sensor
- **Goal:** Read the internal chip temperature.
- **Task:**
    1.  Enable Temp Sensor in `ADC->CCR` (`TSVREFE` bit).
    2.  Select Channel 16 (Temp Sensor).
    3.  Sample Time must be > 10us (use max cycles).
    4.  Formula: $Temp = \frac{V_{sense} - V_{25}}{Avg\_Slope} + 25$. (Values in Datasheet).

### Lab 3: Sampling Rate Test
- **Goal:** See the effect of impedance.
- **Task:**
    1.  Use a high impedance source (e.g., 100k resistor in series with pot).
    2.  Set Sample Time to minimum (3 cycles).
    3.  Observe values are lower than expected (capacitor didn't charge).
    4.  Increase Sample Time to max (480 cycles).
    5.  Observe values become accurate.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Reading is Always 0 or 4095
*   **Cause:** Pin not in Analog Mode (GPIO push-pull fighting the signal).
*   **Cause:** Wrong Channel selected in `SQR3`.
*   **Solution:** Check `MODER` and `SQR` settings.

#### 2. Unstable Readings (Noise)
*   **Cause:** Power supply noise, long wires, breadboard capacitance.
*   **Solution:**
    *   Add a 0.1uF capacitor between Pin and GND (Low-pass filter).
    *   Take multiple samples and average them (Software averaging).

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **ADC Clock:** Don't run ADC at max speed if you don't need to. Lower speed saves power and improves accuracy for high-impedance sources.

### Code Quality
- **Calibration:** ADCs have offset and gain errors. For precision, measure a known reference (like the internal VREFINT, Channel 17) to calculate the actual $V_{DDA}$ voltage instead of assuming 3.3V.
    *   $V_{DDA} = 1.21V \times \frac{4095}{ADC_{VREFINT}}$

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the maximum input voltage for the STM32 ADC?
    *   **A:** $V_{DDA}$ (usually 3.3V). Exceeding this can damage the pin or the ADC.
2.  **Q:** Why do we need a Sample & Hold circuit?
    *   **A:** To keep the voltage stable while the SAR logic performs the 12-step comparison. If the voltage changed during conversion, the result would be garbage.

### Challenge Task
> **Task:** Implement a "Battery Monitor". Read the voltage. If < 3.0V, blink Red LED. If > 3.0V, blink Green LED. Use hysteresis (e.g., threshold 3.0V rising, 2.9V falling) to prevent flickering at the boundary.

---

## 📚 Further Reading & References
- [STM32 ADC Application Note (AN2834)](https://www.st.com/resource/en/application_note/cd00211314-how-to-get-the-best-adc-accuracy-in-stm32-microcontrollers-stmicroelectronics.pdf)

---
