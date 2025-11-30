# Day 26: Advanced DAC & Audio
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
1.  **Configure** a Timer to trigger DAC conversions at a precise sampling rate (e.g., 44.1 kHz).
2.  **Implement** DMA to feed the DAC from a Lookup Table (LUT) automatically.
3.  **Generate** a pure Sine Wave using pre-calculated values.
4.  **Understand** the concept of Direct Digital Synthesis (DDS).
5.  **Play** a short audio clip (raw PCM data) stored in Flash.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Headphones or Speaker (with amplifier) connected to PA4
    *   RC Low-Pass Filter (Optional but recommended)
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   Python (to generate Sine Table)
*   **Prior Knowledge:**
    *   Day 25 (DAC Basics)
    *   Day 13 (DMA)
    *   Day 16 (Timers)
*   **Datasheets:**
    *   [STM32F407 Reference Manual (DAC & DMA)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Waveform Generation Strategies

#### 1.1 CPU Bit-Banging (Bad)
`while(1) { DAC = val; delay(); }`.
*   **Pros:** Simple.
*   **Cons:** Jittery timing. CPU is 100% busy. Cannot do anything else.

#### 1.2 Timer Interrupt (Better)
Timer ISR fires at 44.1 kHz -> Write to DAC.
*   **Pros:** Precise timing.
*   **Cons:** High CPU load (44k interrupts/sec). Context switching overhead.

#### 1.3 DMA + Timer Trigger (Best)
1.  **Timer:** Runs at 44.1 kHz. Sends "Trigger" signal to DAC.
2.  **DAC:** On Trigger, moves data from Holding Register (DHR) to Output Register (DOR). Sends "DMA Request".
3.  **DMA:** On Request, moves next sample from RAM/Flash to DHR.
*   **Pros:** Zero CPU usage. Perfect timing.

### 🔹 Part 2: Direct Digital Synthesis (DDS)
DDS is a method to generate arbitrary frequencies from a fixed clock.
*   **Phase Accumulator:** A counter that increments by a "Tuning Word".
*   **Lookup Table (LUT):** Contains one cycle of a sine wave.
*   **Output:** `DAC = LUT[Phase >> Shift]`.
*   **Frequency:** $f_{out} = \frac{f_{clk} \times TuningWord}{2^{N}}$.

---

## 💻 Implementation: Sine Wave Generator

> **Instruction:** We will generate a 1 kHz Sine Wave using DMA and TIM6.

### 🛠️ Hardware/System Configuration
*   **Pin:** PA4 (DAC1).
*   **Timer:** TIM6 (Basic Timer, dedicated for DAC triggering).
*   **DMA:** DMA1 Stream 5 Channel 7 (DAC1).

### 👨‍💻 Code Implementation

#### Step 1: Generate LUT (Python)
Run this locally to get the array:
```python
import math
N = 100
print("const uint16_t sine_wave[100] = {")
for i in range(N):
    val = int(2047.5 * (1 + math.sin(2 * math.pi * i / N)))
    print(f"{val}, ", end="")
print("};")
```

#### Step 2: C Code (`dac_dma.c`)

```c
#include "stm32f4xx.h"

#define SAMPLES 100
const uint16_t sine_wave[SAMPLES] = {
    2047, 2176, 2304, /* ... paste python output ... */ 2047
};

void TIM6_Init(void) {
    // Enable TIM6 Clock (APB1)
    RCC->APB1ENR |= (1 << 4);

    // F_timer = 84 MHz (APB1 x2).
    // Target: 1 kHz Sine. 100 Samples/Cycle.
    // Sample Rate = 100 kHz.
    // Prescaler = 0 (84 MHz).
    // ARR = 840 - 1.
    
    TIM6->PSC = 0;
    TIM6->ARR = 839;

    // Enable Trigger Output (TRGO) on Update
    TIM6->CR2 &= ~(0x70);
    TIM6->CR2 |= (0x20); // 010: Update Event
    
    // Enable Timer
    TIM6->CR1 |= (1 << 0);
}

void DMA1_Init(void) {
    // Enable DMA1 Clock
    RCC->AHB1ENR |= (1 << 21);

    // DMA1 Stream 5 Channel 7 -> DAC1
    DMA1_Stream5->CR &= ~(1 << 0); // Disable
    while(DMA1_Stream5->CR & (1 << 0));

    DMA1_Stream5->PAR = (uint32_t)&(DAC->DHR12R1);
    DMA1_Stream5->M0AR = (uint32_t)sine_wave;
    DMA1_Stream5->NDTR = SAMPLES;

    // Channel 7 (111)
    // MSIZE 16-bit (01), PSIZE 16-bit (01)
    // MINC (1), PINC (0)
    // Circular Mode (1)
    // Dir: M2P (01)
    DMA1_Stream5->CR = (7 << 25) | (1 << 13) | (1 << 11) | (1 << 10) | (1 << 8) | (1 << 6);

    DMA1_Stream5->CR |= (1 << 0); // Enable
}

void DAC_DMA_Init(void) {
    // Enable GPIOA, DAC
    RCC->AHB1ENR |= (1 << 0);
    RCC->APB1ENR |= (1 << 29);
    GPIOA->MODER |= (3 << 8); // PA4 Analog

    // Configure DAC
    // TEN1 = 1 (Trigger Enable)
    // TSEL1 = 000 (TIM6 TRGO) - Check Mapping!
    // DMAEN1 = 1 (DMA Enable)
    // EN1 = 1
    
    DAC->CR |= (1 << 12) | (1 << 2) | (1 << 0);
    // TSEL default is 000 (TIM6_TRGO) on F407.
}

int main(void) {
    DMA1_Init();
    DAC_DMA_Init();
    TIM6_Init(); // Start Timer last
    
    while(1) {
        // CPU is free!
    }
}
```

---

## 🔬 Lab Exercise: Lab 26.1 - Audio Player

### 1. Lab Objectives
- Play a short "beep" or sound effect stored in Flash.
- Understand memory constraints (audio takes a lot of space).

### 2. Step-by-Step Guide

#### Phase A: Data Prep
1.  Find a short .wav file (e.g., 0.5 sec).
2.  Convert to 8-bit unsigned, 8 kHz, Mono using Audacity.
3.  Export as C array (HxD or `xxd -i`).
4.  Size: 0.5s * 8000 = 4000 bytes. Fits easily in Flash.

#### Phase B: Configuration Changes
1.  **Timer:** 8 kHz. (84MHz / 10500).
2.  **DMA:** Point to `audio_data` array. Size = 4000. Mode = Normal (Not Circular) to play once.
3.  **DAC:** Use `DHR8R1` (8-bit register).

#### Phase C: Playback Logic
```c
void Play_Audio(void) {
    DMA1_Stream5->CR &= ~(1 << 0); // Disable
    DMA1_Stream5->NDTR = AUDIO_LEN;
    DMA1_Stream5->M0AR = (uint32_t)audio_data;
    DMA1_Stream5->CR |= (1 << 0); // Enable
}
```

### 3. Verification
Connect headphones (through a capacitor/resistor to remove DC offset). Call `Play_Audio()` on button press. You should hear the sound.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Arbitrary Waveform Generator (AWG)
- **Goal:** Change waveform on the fly.
- **Task:**
    1.  Create arrays for Sine, Square, Triangle.
    2.  Use a Button to switch the DMA `M0AR` pointer.
    3.  Restart DMA to apply change.

### Lab 3: DTMF Generator (Dual Tone Multi-Frequency)
- **Goal:** Simulate phone keypad tones.
- **Task:**
    1.  Sum two sine waves in software (e.g., 697 Hz + 1209 Hz for '1').
    2.  Fill a buffer.
    3.  Play via DMA.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Distorted Sound
*   **Cause:** Clipping. The sum of waves exceeds 4095 (12-bit) or 255 (8-bit).
*   **Solution:** Scale down the values in the LUT.

#### 2. Wrong Frequency
*   **Cause:** Timer calculation error.
*   **Solution:** Check `ARR` and `PSC`. Remember $f = f_{clk} / ((PSC+1)(ARR+1))$.

#### 3. Clicking Noise
*   **Cause:** Sudden jump in voltage when stopping/starting playback.
*   **Solution:** Ramp down the volume (multiply samples by 0.9, 0.8...) before stopping.

---

## ⚡ Optimization & Best Practices

### Memory Management
- **Double Buffering:** For long audio (streaming from SD Card), use Double Buffering. DMA fills Buffer A while CPU reads Buffer B from SD Card.

### Code Quality
- **Const Correctness:** Store LUTs in `const` memory (Flash) to save RAM. `const uint16_t sine_wave[]`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we use TIM6 for DAC triggering?
    *   **A:** TIM6 is a "Basic Timer" specifically designed for DAC triggering. It doesn't have external I/O pins, leaving them free for other uses.
2.  **Q:** What is the advantage of 8-bit audio over 12-bit?
    *   **A:** Saves 50% memory (1 byte vs 2 bytes per sample). Quality is lower (more quantization noise), but acceptable for speech/beeps.

### Challenge Task
> **Task:** Implement a "Variable Frequency Oscillator". Connect a Potentiometer to ADC. Use the ADC value to update the TIM6 `ARR` register in the main loop. Result: Turning the knob changes the pitch of the sine wave.

---

## 📚 Further Reading & References
- [STM32 Audio Playback Application Note](https://www.st.com/resource/en/application_note/dm00040808-audio-playback-and-recording-using-the-stm32f4discovery-stmicroelectronics.pdf)

---
