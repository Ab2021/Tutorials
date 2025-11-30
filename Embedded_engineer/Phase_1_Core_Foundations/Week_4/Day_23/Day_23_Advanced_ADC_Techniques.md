# Day 23: Advanced ADC Techniques
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
1.  **Configure** the ADC in Scan Mode to read multiple channels automatically.
2.  **Integrate** DMA to transfer ADC data to memory without CPU intervention.
3.  **Implement** Continuous Mode for high-speed signal acquisition.
4.  **Calibrate** ADC readings using the internal Voltage Reference (VREFINT).
5.  **Synchronize** ADC conversions with a Timer (Triggered Injection).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   2x Potentiometers (or 1 Pot + Internal Temp Sensor)
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 22 (ADC Basics)
    *   Day 13 (DMA)
*   **Datasheets:**
    *   [STM32F407 Reference Manual (ADC & DMA)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Scan Mode & DMA

#### 1.1 The Problem with Polling
Reading one channel is easy. Reading 3 channels (e.g., X, Y, Z accelerometer) with polling is inefficient:
1.  Select Ch1 -> Start -> Wait -> Read.
2.  Select Ch2 -> Start -> Wait -> Read.
3.  Select Ch3 -> Start -> Wait -> Read.
This wastes CPU time and introduces jitter between samples.

#### 1.2 Scan Mode
The ADC can be programmed with a **Sequence** of channels (e.g., Ch1, then Ch2, then Ch3).
*   **EOC (End of Conversion):** Fired after *each* channel.
*   **DMA Request:** Fired after each conversion to move the data to RAM.

#### 1.3 DMA Circular Mode
We set up a buffer in RAM (array of 3 uint16_t). The DMA automatically fills:
*   `Buffer[0]` <-- Ch1 Data
*   `Buffer[1]` <-- Ch2 Data
*   `Buffer[2]` <-- Ch3 Data
*   Wrap around to `Buffer[0]`.

The CPU just reads the array whenever it wants the latest values.

### 🔹 Part 2: Triggering & Injection

#### 2.1 Regular vs. Injected Groups
*   **Regular Group:** The normal sequence (up to 16 channels). Can be interrupted by Injected group.
*   **Injected Group:** High priority (up to 4 channels). Like an "Analog Interrupt".
    *   *Use Case:* Motor control. Read current *exactly* when the PWM is in the middle of the ON pulse. Triggered by Timer Output Compare.

#### 2.2 Timer Triggering
Instead of software starting the ADC (`SWSTART`), a Timer can start it. This guarantees exact sampling frequency (e.g., 44.1 kHz for audio).

---

## 💻 Implementation: Multi-Channel DMA ADC

> **Instruction:** We will read PA1 (Pot 1), PA2 (Pot 2), and the Internal Temperature Sensor (Ch 16) continuously using DMA.

### 🛠️ Hardware/System Configuration
*   **PA1:** ADC1_IN1.
*   **PA2:** ADC1_IN2.
*   **Internal:** ADC1_IN16.

### 👨‍💻 Code Implementation

#### Step 1: DMA Configuration (`adc_dma.c`)

```c
#include "stm32f4xx.h"

#define ADC_CHANNELS 3
volatile uint16_t adc_buffer[ADC_CHANNELS];

void DMA2_Init(void) {
    // ADC1 is connected to DMA2 Stream 0 Channel 0 or Stream 4 Channel 0
    // Let's use Stream 0 Channel 0.
    
    RCC->AHB1ENR |= (1 << 22); // DMA2 Clock

    DMA2_Stream0->CR &= ~(1 << 0); // Disable
    while(DMA2_Stream0->CR & (1 << 0));

    DMA2_Stream0->PAR = (uint32_t)&(ADC1->DR);
    DMA2_Stream0->M0AR = (uint32_t)adc_buffer;
    DMA2_Stream0->NDTR = ADC_CHANNELS;

    // Channel 0 (000)
    // MSIZE 16-bit (01), PSIZE 16-bit (01)
    // MINC (1), PINC (0)
    // Circular Mode (1)
    // Dir: P2M (00)
    DMA2_Stream0->CR = (1 << 10) | (1 << 8) | (1 << 11) | (1 << 13);

    DMA2_Stream0->CR |= (1 << 0); // Enable
}
```

#### Step 2: ADC Configuration
```c
void ADC1_DMA_Init(void) {
    RCC->AHB1ENR |= (1 << 0); // GPIOA
    RCC->APB2ENR |= (1 << 8); // ADC1

    // PA1, PA2 Analog
    GPIOA->MODER |= (3 << 2) | (3 << 4);

    // Common Config
    ADC->CCR |= (1 << 23); // TSVREFE (Enable Temp Sensor)
    ADC->CCR |= (1 << 16); // Prescaler 4

    // ADC1 Config
    ADC1->CR1 |= (1 << 8); // SCAN Mode
    ADC1->CR2 |= (1 << 1); // CONT (Continuous Mode)
    ADC1->CR2 |= (1 << 8); // DMA Mode
    ADC1->CR2 |= (1 << 9); // DDS (DMA Disable Selection) - Keep DMA requests coming in Circular mode

    // Sequence
    ADC1->SQR1 = (2 << 20); // L = 2 (3 conversions: 0, 1, 2)
    ADC1->SQR3 = (1 << 0) | (2 << 5) | (16 << 10); // SQ1=Ch1, SQ2=Ch2, SQ3=Ch16

    // Sample Times
    // Temp Sensor needs > 10us. 480 cycles.
    ADC1->SMPR1 |= (7 << 18); // Ch16 = 480 cyc
    ADC1->SMPR2 |= (3 << 3) | (3 << 6); // Ch1, Ch2 = 56 cyc

    // Enable
    ADC1->CR2 |= (1 << 0);
}
```

#### Step 3: Main Loop
```c
#include <stdio.h>

int main(void) {
    DMA2_Init();
    ADC1_DMA_Init();
    
    // Start Conversion
    ADC1->CR2 |= (1 << 30); // SWSTART

    while(1) {
        // Data is automatically updated in adc_buffer!
        uint16_t pot1 = adc_buffer[0];
        uint16_t pot2 = adc_buffer[1];
        uint16_t temp_raw = adc_buffer[2];

        // Calculate Temp
        // V_sense = temp_raw * 3.3 / 4095
        // Temp = (V_sense - V_25) / Avg_Slope + 25
        // V_25 = 0.76V, Slope = 2.5 mV/C
        float v_sense = temp_raw * 3.3f / 4095.0f;
        float temp = ((v_sense - 0.76f) / 0.0025f) + 25.0f;

        printf("P1: %d, P2: %d, Temp: %.1f C\r", pot1, pot2, temp);
        
        for(int i=0; i<1000000; i++);
    }
}
```

---

## 🔬 Lab Exercise: Lab 23.1 - Calibration

### 1. Lab Objectives
- Use the internal VREFINT (1.21V) to calculate the actual VDD.
- Correct the Potentiometer readings based on actual VDD.

### 2. Step-by-Step Guide

#### Phase A: Theory
The ADC assumes VREF+ is 3.3V. If your USB power is 3.2V, all readings are scaled incorrectly.
$V_{DDA} = 1.21V \times \frac{4095}{ADC_{VREFINT}}$.

#### Phase B: Implementation
1.  Add Channel 17 (VREFINT) to the sequence (make it 4 channels).
2.  Enable `TSVREFE` (it enables both Temp and Vref).
3.  In Main Loop:
    ```c
    uint16_t vref_raw = adc_buffer[3];
    float actual_vdd = 1.21f * 4095.0f / vref_raw;
    
    // Corrected Pot Voltage
    float pot_volts = pot1 * actual_vdd / 4095.0f;
    ```

### 3. Verification
Measure the 3.3V pin with a multimeter. Compare with `actual_vdd`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Timer Triggered ADC
- **Goal:** Sample at exactly 1 kHz.
- **Task:**
    1.  Disable Continuous Mode (`CONT=0`).
    2.  Configure TIM3 to update at 1 kHz.
    3.  Configure TIM3 TRGO (Trigger Output) on Update Event.
    4.  Configure ADC `EXTEN` (External Trigger Enable) to Rising Edge.
    5.  Configure ADC `EXTSEL` to TIM3 TRGO.
    6.  Observe that `adc_buffer` updates exactly 1000 times a second.

### Lab 3: Triple ADC Interleaved Mode (Advanced)
- **Goal:** Double/Triple the sampling rate.
- **Task:**
    1.  Use ADC1, ADC2, and ADC3 on the *same* pin.
    2.  Configure Interleaved Mode in `ADC->CCR`.
    3.  ADC1 starts. After 5 cycles, ADC2 starts. After 5 cycles, ADC3 starts.
    4.  Result: Effective sampling rate is 3x the individual ADC speed.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. DMA Overrun
*   **Symptom:** `OVR` flag in ADC SR. Data is lost.
*   **Cause:** The ADC is converting faster than the DMA can move data (or CPU is hogging the bus).
*   **Solution:** Increase ADC Sample Time (slow it down) or optimize Bus Matrix priority.

#### 2. Wrong Data Order
*   **Symptom:** Pot1 data appears in Pot2 variable.
*   **Cause:** Sequence Length (`L` in `SQR1`) doesn't match DMA buffer size, or DMA wasn't reset properly.
*   **Solution:** Always check `NDTR` and `SQR1`.

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **DMA Priority:** Give DMA2 Stream 0 "High" or "Very High" priority if you are doing audio or motor control.
- **Data Alignment:** Use `half-word` (16-bit) transfers for ADC data to save bus bandwidth and RAM.

### Code Quality
- **Double Buffering:** For block processing (e.g., FFT), use DMA Double Buffer Mode (Circular with Half-Transfer Interrupt). Process the first half while DMA fills the second half.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we use Circular Mode for DMA with ADC?
    *   **A:** So the DMA automatically resets the pointer to the start of the buffer after filling it, allowing continuous background scanning without CPU intervention.
2.  **Q:** What is the purpose of VREFINT?
    *   **A:** It provides a stable 1.21V reference internally, allowing the software to calculate the actual analog supply voltage ($V_{DDA}$) and calibrate readings.

### Challenge Task
> **Task:** Implement a "Peak Detector". Sample a signal at 100 kHz using Timer Trigger + DMA. In the DMA Transfer Complete interrupt, find the maximum value in the buffer and print it.

---

## 📚 Further Reading & References
- [STM32 ADC Modes Application Note (AN3116)](https://www.st.com/resource/en/application_note/cd00258017-stm32-adc-modes-and-their-applications-stmicroelectronics.pdf)

---
