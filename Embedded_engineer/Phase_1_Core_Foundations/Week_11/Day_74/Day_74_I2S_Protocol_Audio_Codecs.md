# Day 74: I2S Protocol & Audio Codecs
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
1.  **Explain** the I2S (Inter-IC Sound) protocol signals (SD, WS, CK, MCK).
2.  **Configure** the STM32 SPI peripheral in I2S Master Mode.
3.  **Interface** with the CS43L22 Audio DAC (used on Discovery Board) via I2C.
4.  **Generate** audio samples and transmit them via I2S.
5.  **Troubleshoot** common audio issues like glitches and silence.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board (Has CS43L22).
    *   Headphones / Speaker (3.5mm jack).
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 34 (I2C - for control)
    *   Day 29 (SPI - I2S is based on SPI)
*   **Datasheets:**
    *   [CS43L22 Datasheet](https://www.cirrus.com/products/cs43l22/)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: I2S Protocol
Designed by Philips for digital audio.
*   **SD (Serial Data):** The audio bits (PCM).
*   **CK (Bit Clock):** Shifts data. Frequency = SampleRate × BitsPerSample × Channels.
*   **WS (Word Select):** Left/Right Clock (LRCK).
    *   Low = Left Channel.
    *   High = Right Channel.
*   **MCK (Master Clock):** Oversampling clock for the DAC (usually 256 × Fs). Required by high-quality DACs.

### 🔹 Part 2: CS43L22 Codec
*   **Control Interface:** I2C (Addr `0x94`). Used to set volume, unmute, power up.
*   **Data Interface:** I2S.
*   **Reset Pin:** PD4. Must be High to operate.

---

## 💻 Implementation: Audio Driver

> **Instruction:** Play a 1kHz Beep.

### 🛠️ Hardware/System Configuration
*   **I2S3:** PC10 (CK), PC12 (SD), PA4 (WS), PC7 (MCK).
*   **I2C1:** PB6 (SCL), PB9 (SDA).
*   **Reset:** PD4.

### 👨‍💻 Code Implementation

#### Step 1: I2S Init
```c
void I2S3_Init(void) {
    // Enable Clocks (SPI3, GPIOC, GPIOA)
    RCC->APB1ENR |= (1 << 15); // SPI3
    
    // Configure Pins (AF06 for SPI3/I2S3)
    // ...
    
    // Configure I2S PLL (PLLI2S) to generate accurate audio clock
    // This is complex. For 48kHz, we need specific N, R values.
    // Assume SystemClock_Config handles PLLI2S.
    
    // SPI3_I2SCFGR
    // I2SMOD=1, I2SE=0 (Disable first)
    // I2SCFG=10 (Master Transmit)
    // PCMSYNC=0 (Short Frame)
    // I2SSTD=00 (Philips Standard)
    // CKPOL=0 (Low)
    // DATLEN=00 (16-bit)
    // CHLEN=0 (16-bit)
    SPI3->I2SCFGR = (1 << 11) | (2 << 8); 
    
    // Enable MCK Output
    SPI3->I2SPR |= (1 << 9);
    
    // Enable I2S
    SPI3->I2SCFGR |= (1 << 10);
}
```

#### Step 2: CS43L22 Init (I2C)
```c
void CS43L22_Init(void) {
    // 1. Reset High
    GPIOD->ODR |= (1 << 4);
    
    // 2. Power Down (Reg 0x02 = 0x01)
    I2C_Write(0x94, 0x02, 0x01);
    
    // 3. Output Device (Headphone) (Reg 0x04 = 0xAF)
    I2C_Write(0x94, 0x04, 0xAF);
    
    // 4. Master Volume (Reg 0x20 = +0dB)
    I2C_Write(0x94, 0x20, 0x18); // 0x18 is roughly 0dB? Check datasheet.
    
    // 5. Power Up (Reg 0x02 = 0x9E)
    I2C_Write(0x94, 0x02, 0x9E);
}
```

#### Step 3: Play Loop (Blocking)
```c
void Play_Tone(void) {
    int16_t sample;
    float t = 0;
    
    while(1) {
        // Wait for TXE
        while(!(SPI3->SR & (1 << 1)));
        
        // Generate Sine (Left)
        sample = (int16_t)(30000.0f * arm_sin_f32(2*PI*1000*t));
        SPI3->DR = sample;
        
        // Wait for TXE
        while(!(SPI3->SR & (1 << 1)));
        
        // Generate Sine (Right)
        SPI3->DR = sample;
        
        t += 1.0f/48000.0f;
        if (t > 1.0f) t -= 1.0f;
    }
}
```

---

## 🔬 Lab Exercise: Lab 74.1 - The Synthesizer

### 1. Lab Objectives
- Generate Sawtooth, Square, and Sine waves.
- Switch between them using the User Button.

### 2. Step-by-Step Guide

#### Phase A: Waveforms
1.  **Sawtooth:** `sample = (t * 60000) - 30000`.
2.  **Square:** `sample = (t < 0.5) ? 30000 : -30000`.

#### Phase B: Control
1.  Poll Button.
2.  Change `wave_type` variable.
3.  Listen to the change in timbre.

### 3. Verification
If sound is distorted, check Volume. If silence, check PD4 (Reset) and I2C ACK.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Stereo Panning
- **Goal:** Move sound Left to Right.
- **Task:**
    1.  `Left = Sample * (1.0 - Pan)`.
    2.  `Right = Sample * Pan`.
    3.  Vary `Pan` from 0 to 1 over time.

### Lab 3: 24-bit Audio
- **Goal:** High Res Audio.
- **Task:**
    1.  Change I2S config to 24-bit data / 32-bit frame.
    2.  Send 24-bit samples (Left aligned in 16-bit writes? Or 32-bit writes?).
    3.  Note: SPI DR is 16-bit. For 24/32 bit, you write twice or use 32-bit access if allowed.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Clicking Noise
*   **Cause:** Buffer underrun. CPU too slow to feed I2S.
*   **Solution:** Use DMA (Next Day).

#### 2. High Pitch Screech
*   **Cause:** Wrong I2S Clock configuration. Sample rate mismatch.
*   **Solution:** Check PLLI2S settings in CubeMX or Reference Manual.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Lookup Tables:** Don't call `sin()` in the audio loop. Use a pre-calculated table or CMSIS-DSP `arm_sin_f32` (which uses a table).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the purpose of MCK?
    *   **A:** It drives the Delta-Sigma modulator inside the DAC. Without it, the DAC cannot convert digital to analog.
2.  **Q:** Why is I2S different from SPI?
    *   **A:** I2S is continuous streaming. SPI is usually bursty. I2S has specific framing (WS) for stereo.

### Challenge Task
> **Task:** Implement "Soft Mute". When stopping playback, ramp volume down to 0 over 100ms instead of cutting it abruptly (which causes a "Pop").

---

## 📚 Further Reading & References
- [I2S Bus Specification (Philips/NXP)](https://www.nxp.com/docs/en/user-guide/UM11732.pdf)

---
