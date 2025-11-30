# Day 27: Analog Comparators and OpAmps
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
1.  **Explain** the function of an Analog Comparator and how it differs from an ADC.
2.  **Implement** a "Software Comparator" using the STM32 ADC Analog Watchdog (AWD) feature.
3.  **Interface** external Op-Amps to STM32 ADC inputs (Buffering, Amplification).
4.  **Design** a Zero Crossing Detector (ZCD) for AC signal analysis.
5.  **Configure** hysteresis to prevent noise-induced toggling.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   LDR (Light Dependent Resistor) or Photodiode
    *   Op-Amp (e.g., LM358) - Optional but recommended
    *   Potentiometer
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 22 (ADC)
    *   Day 11 (Interrupts)
*   **Datasheets:**
    *   [STM32F407 Reference Manual (ADC Watchdog)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Comparators vs. ADCs

#### 1.1 The Comparator
A comparator takes two analog voltages ($V_{in+}$ and $V_{in-}$) and outputs a digital logic level.
*   If $V_{in+} > V_{in-}$, Output = 1 (High).
*   If $V_{in+} < V_{in-}$, Output = 0 (Low).
*   **Speed:** Very fast (nanoseconds).
*   **Use Case:** Zero crossing detection, Over-voltage protection.

#### 1.2 The STM32F4 Limitation
The STM32F407 **does not** have a dedicated internal Comparator peripheral (unlike STM32F3/L4).
*   **Workaround:** Use the **ADC Analog Watchdog (AWD)**.
*   **AWD:** Hardware feature inside the ADC. It compares the converted result against High and Low thresholds. If outside window -> Interrupt.

### 🔹 Part 2: Analog Watchdog (AWD)

The AWD monitors one or all ADC channels.
*   **HTR (High Threshold Register):** Upper limit.
*   **LTR (Low Threshold Register):** Lower limit.
*   **Logic:** If $ADC < LTR$ OR $ADC > HTR$, set `AWD` flag and trigger interrupt.
*   **Latency:** Depends on ADC sampling rate (slower than a real comparator).

### 🔹 Part 3: Op-Amp Interfacing

#### 3.1 Why Op-Amps?
1.  **Impedance Matching:** ADC needs low impedance source (< 10k). Sensors (pH probe, Piezo) have high impedance. Op-Amp Buffer (Voltage Follower) fixes this.
2.  **Amplification:** ADC range is 0-3.3V. If signal is 0-10mV (Shunt resistor), we need Gain.
3.  **Level Shifting:** If signal is +/- 1V (Audio), we need to shift it to 0-3.3V (center at 1.65V).

---

## 💻 Implementation: Analog Watchdog (Light Trigger)

> **Instruction:** We will use an LDR. If the light level drops below a threshold (Shadow), turn on an LED. We will use AWD, not software `if` statements.

### 🛠️ Hardware/System Configuration
*   **Input:** PA1 (LDR Voltage Divider).
*   **Threshold:** Determine experimentally (e.g., 1.5V -> ADC 1860).

### 👨‍💻 Code Implementation

#### Step 1: Initialization (`awd.c`)

```c
#include "stm32f4xx.h"

void ADC_AWD_Init(void) {
    // 1. Enable Clocks (GPIOA, ADC1)
    RCC->AHB1ENR |= (1 << 0);
    RCC->APB2ENR |= (1 << 8);

    // 2. Configure PA1 Analog
    GPIOA->MODER |= (3 << 2);

    // 3. Configure ADC1 (Continuous Mode)
    ADC1->CR2 |= (1 << 1); // CONT
    ADC1->SQR3 = 1; // Channel 1

    // 4. Configure Analog Watchdog
    // Select Channel 1 (AWDCH = 1)
    // Enable on Regular Channels (AWDEN = 1)
    ADC1->CR1 |= (1 << 0) | (1 << 23); // AWDCH=1 (Bits 0-4), AWDEN (Bit 23)

    // 5. Set Thresholds
    // High Threshold: 4095 (Max) - We only care about Low
    // Low Threshold: 1500 (Darkness)
    ADC1->HTR = 4095;
    ADC1->LTR = 1500;

    // 6. Enable Interrupt
    ADC1->CR1 |= (1 << 6); // AWDIE
    NVIC_EnableIRQ(ADC_IRQn);

    // 7. Enable ADC and Start
    ADC1->CR2 |= (1 << 0); // ADON
    ADC1->CR2 |= (1 << 30); // SWSTART
}
```

#### Step 2: Interrupt Handler
```c
void ADC_IRQHandler(void) {
    if (ADC1->SR & (1 << 0)) { // AWD Flag
        // Clear Flag
        ADC1->SR &= ~(1 << 0);
        
        // Action: Turn ON LED (PD12)
        GPIOD->ODR |= (1 << 12);
        
        // Note: In a real app, you might want hysteresis here.
        // The interrupt will fire continuously while < 1500.
    }
}
```

#### Step 3: Main Loop
```c
int main(void) {
    // Init LED
    RCC->AHB1ENR |= (1 << 3);
    GPIOD->MODER |= (1 << 24);
    
    ADC_AWD_Init();
    
    while(1) {
        // If light returns, turn off LED?
        // We need to read ADC to know.
        if (ADC1->DR > 1600) { // Hysteresis (1500 + 100)
            GPIOD->ODR &= ~(1 << 12);
        }
    }
}
```

---

## 🔬 Lab Exercise: Lab 27.1 - Zero Crossing Detector

### 1. Lab Objectives
- Detect when a sine wave crosses 1.65V (Virtual Ground).
- Measure the frequency of the AC signal.

### 2. Step-by-Step Guide

#### Phase A: Hardware
1.  Signal Generator: Sine Wave, 3Vpp, Offset 1.65V.
2.  Connect to PA1.

#### Phase B: AWD Config
1.  **Window:** We want to detect crossing 1.65V (ADC ~2048).
2.  **Trick:** Set HTR = 2050, LTR = 2046.
3.  **Interrupt:** Fires when signal is *outside* this narrow window.
    *   Wait, AWD fires when *outside*. This isn't quite a ZCD.
    *   **Better Approach:** Use AWD to detect *peaks*? No.
    *   **Real ZCD:** Use an external Op-Amp comparator (LM358) to drive a GPIO Interrupt (EXTI).

#### Phase C: External Comparator Implementation
1.  **Op-Amp:**
    *   In+ = Signal.
    *   In- = 1.65V (Potentiometer).
    *   Out = Square Wave (0V or 3.3V).
2.  **STM32:**
    *   Connect Out to PA0.
    *   Configure PA0 as EXTI (Rising Edge).
    *   ISR: `Frequency = SystemCoreClock / (CurrentTime - LastTime)`.

### 3. Verification
Compare the measured frequency with the Signal Generator display.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Software Hysteresis
- **Goal:** Prevent LED flickering at twilight.
- **Task:**
    1.  State Machine: `LIGHT` vs `DARK`.
    2.  If `LIGHT`: Switch to `DARK` only if ADC < 1400.
    3.  If `DARK`: Switch to `LIGHT` only if ADC > 1600.
    4.  Gap (200) is the hysteresis band.

### Lab 3: Microphone Pre-Amp
- **Goal:** Interface an Electret Mic.
- **Task:**
    1.  Mic output is tiny (~20mV).
    2.  Build Non-Inverting Amplifier with Op-Amp (Gain = 100).
    3.  Bias output to 1.65V.
    4.  Feed to ADC.
    5.  Visualize waveform on Serial Plotter.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. AWD Interrupt Storm
*   **Symptom:** Main loop never runs.
*   **Cause:** The signal stays outside the threshold. The ISR clears the flag, returns, and immediately fires again.
*   **Solution:** Disable the AWD interrupt inside the ISR. Re-enable it only when the signal returns to "normal" (checked in Main loop or Timer).

#### 2. Op-Amp Clipping
*   **Symptom:** Signal looks flat at top/bottom.
*   **Cause:** Gain too high, or Op-Amp not Rail-to-Rail. LM358 cannot reach 3.3V (max ~2V on 3.3V supply).
*   **Solution:** Use a Rail-to-Rail Op-Amp (e.g., MCP6002) or lower the gain.

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **AWD vs Polling:** AWD is zero-overhead monitoring. Use it for safety features (e.g., Battery Over-voltage, Motor Over-current) where reaction time is critical but rare.

### Code Quality
- **Thresholds:** Don't hardcode `1500`. Use `#define` or configurable variables stored in Flash.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Does the STM32F407 have a hardware comparator?
    *   **A:** No. (Some other STM32 families do). We simulate it with ADC AWD or external hardware.
2.  **Q:** What is the purpose of Hysteresis?
    *   **A:** To prevent rapid toggling of the output when the input signal is noisy and hovering near the threshold.

### Challenge Task
> **Task:** Implement a "Clap Switch". Use the Microphone circuit. Use AWD to detect a sudden spike (Clap). Toggle an LED. Ignore background noise (slow changes).

---

## 📚 Further Reading & References
- [Op-Amps for Everyone (TI)](https://web.mit.edu/6.101/www/reference/op_amps_everyone.pdf)

---
