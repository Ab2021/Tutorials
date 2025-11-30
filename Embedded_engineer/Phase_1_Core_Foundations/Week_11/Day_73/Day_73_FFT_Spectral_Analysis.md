# Day 73: FFT & Spectral Analysis
## Phase 1: Core Embedded Engineering Foundations | Week 11: DSP & Audio Processing

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
1.  **Explain** the principles of the Fast Fourier Transform (FFT) and its output (Bins, Magnitude, Phase).
2.  **Calculate** Frequency Resolution and Bin Frequency.
3.  **Apply** Window Functions (Hanning, Hamming) to reduce Spectral Leakage.
4.  **Implement** FFT using `arm_cfft_f32` and `arm_cmplx_mag_f32`.
5.  **Identify** the dominant frequency in a sampled signal.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   [CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP)
*   **Prior Knowledge:**
    *   Day 71 (CMSIS-DSP Basics)
    *   Day 22 (ADC Sampling)
*   **Datasheets:**
    *   [CMSIS-DSP Transform Functions](https://arm-software.github.io/CMSIS_5/DSP/html/group__groupTransforms.html)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Time vs Frequency
*   **Time Domain:** Amplitude vs Time (Oscilloscope).
*   **Frequency Domain:** Magnitude vs Frequency (Spectrum Analyzer).
*   **DFT/FFT:** Converts Time samples to Frequency bins.
    *   **Input:** $N$ Real samples (or Complex).
    *   **Output:** $N/2$ Complex bins (Real + Imaginary).
    *   **Magnitude:** $\sqrt{Re^2 + Im^2}$.

### 🔹 Part 2: Key Parameters
*   **Sample Rate ($F_s$):** e.g., 10 kHz.
*   **FFT Size ($N$):** e.g., 1024. Must be power of 2.
*   **Frequency Resolution:** $F_s / N$. e.g., $10000 / 1024 \approx 9.7$ Hz per bin.
*   **Nyquist Limit:** Max detectable frequency is $F_s / 2$.

### 🔹 Part 3: Spectral Leakage & Windowing
If the signal doesn't fit perfectly into the window (integer number of cycles), energy "leaks" into adjacent bins.
*   **Solution:** Multiply input by a Window Function (e.g., Hanning) which tapers the edges to zero.

---

## 💻 Implementation: FFT on STM32

> **Instruction:** Generate a 50Hz and 200Hz mixed signal. Perform FFT. Find peaks.

### 👨‍💻 Code Implementation

#### Step 1: Buffers
FFT works in-place or out-of-place. `arm_cfft_f32` is in-place.
Input buffer must be $2 \times N$ (Real, Imag, Real, Imag...).
```c
#define FFT_SIZE 1024
float32_t fft_input[FFT_SIZE * 2];
float32_t fft_output[FFT_SIZE];
```

#### Step 2: Generate Signal
```c
void Generate_Signal(void) {
    for(int i=0; i<FFT_SIZE; i++) {
        // 50Hz + 200Hz
        float32_t t = (float32_t)i / 1000.0f; // Fs = 1kHz
        float32_t val = arm_sin_f32(2*PI*50*t) + 0.5f * arm_sin_f32(2*PI*200*t);
        
        fft_input[i*2] = val;     // Real
        fft_input[i*2 + 1] = 0;   // Imaginary
    }
}
```

#### Step 3: Perform FFT
```c
void Process_FFT(void) {
    // 1. Initialize CFFT Instance (1024)
    const arm_cfft_instance_f32 *S = &arm_cfft_sR_f32_len1024;
    
    // 2. Run FFT
    arm_cfft_f32(S, fft_input, 0, 1);
    
    // 3. Calculate Magnitude
    arm_cmplx_mag_f32(fft_input, fft_output, FFT_SIZE);
    
    // Note: fft_output[0] is DC. fft_output[1] is Fs/N...
}
```

#### Step 4: Find Peak
```c
void Find_Peak(void) {
    float32_t maxValue;
    uint32_t maxIndex;
    
    // Skip DC (Index 0)
    arm_max_f32(&fft_output[1], FFT_SIZE/2 - 1, &maxValue, &maxIndex);
    
    maxIndex++; // Compensate for skip
    
    float32_t freq = (float32_t)maxIndex * 1000.0f / FFT_SIZE;
    printf("Dominant Freq: %.2f Hz\n", freq);
}
```

---

## 🔬 Lab Exercise: Lab 73.1 - The Tuner

### 1. Lab Objectives
- Connect a Microphone (ADC) or Signal Generator.
- Detect the note being played (e.g., A4 = 440Hz).

### 2. Step-by-Step Guide

#### Phase A: ADC Setup
1.  Configure ADC1, Timer Trigger (10kHz), DMA.
2.  Fill `fft_input` buffer (Real part).

#### Phase B: Processing
1.  When buffer full:
    *   Apply Hanning Window (Multiply).
    *   Run FFT.
    *   Find Max Index.
    *   Calculate Freq.

#### Phase C: Test
1.  Play 440Hz tone on phone.
2.  **Observation:** STM32 prints "439.5 Hz" (Resolution dependent).

### 3. Verification
If resolution is too low (e.g., 10Hz), increase FFT Size or Decrease Sample Rate (Zoom in).

---

## 🧪 Additional / Advanced Labs

### Lab 2: Real FFT (RFFT)
- **Goal:** Optimization.
- **Task:**
    1.  Use `arm_rfft_fast_f32`.
    2.  Takes $N$ real inputs (no imaginary padding).
    3.  Faster and uses half the RAM.

### Lab 3: Spectrogram
- **Goal:** Time-Frequency Analysis.
- **Task:**
    1.  Compute FFT every 100ms.
    2.  Send Magnitude array to PC via UART.
    3.  PC (Python) plots a "Waterfall" graph (Heatmap).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. DC Offset
*   **Cause:** ADC measures 0-3.3V. Sine wave is centered at 1.65V.
*   **Result:** Huge peak at Bin 0 (0 Hz).
*   **Solution:** Subtract Mean (Average) from signal before FFT.

#### 2. Mirror Image
*   **Cause:** FFT output is symmetric around Nyquist ($N/2$).
*   **Solution:** Only look at indices $0$ to $N/2$. Ignore the rest.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Bit Reversal:** FFT requires bit-reversed addressing. CMSIS handles this, but ensure your buffers are aligned if required (usually handled by compiler).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** If $F_s = 48$ kHz and $N = 1024$, what is the bin width?
    *   **A:** $48000 / 1024 = 46.875$ Hz.
2.  **Q:** How do I distinguish 440Hz from 445Hz with that resolution?
    *   **A:** You can't directly. You need "Zero Padding" (add zeros to end of input) to interpolate the spectrum, or use a larger $N$.

### Challenge Task
> **Task:** Implement "Goertzel Algorithm". It detects a *single* specific frequency (e.g., DTMF tones) much faster than a full FFT. Detect if a 1kHz tone is present.

---

## 📚 Further Reading & References
- [The Fast Fourier Transform (Visual Explanation)](https://www.youtube.com/watch?v=spUNpyF58BY)

---
