# Day 87: Logic Analyzers and Oscilloscopes
## Phase 1: Core Embedded Engineering Foundations | Week 13: Debugging and Testing

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
1.  **Capture** and **Decode** digital signals (UART, I2C, SPI) using a Logic Analyzer.
2.  **Measure** signal integrity parameters (Rise Time, Overshoot, Jitter) using an Oscilloscope.
3.  **Identify** common hardware issues like "Bus Contention" and "Missing Pull-ups".
4.  **Correlate** firmware execution with hardware events using GPIO toggles.
5.  **Debug** timing violations in high-speed interfaces.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Logic Analyzer (e.g., Saleae Logic 8 or cheap clone with Sigrok).
    *   Oscilloscope (Optional but recommended).
*   **Software Required:**
    *   PulseView (Sigrok) or Saleae Logic Software.
*   **Prior Knowledge:**
    *   Protocols: UART, I2C, SPI.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Logic Analyzer vs Oscilloscope
*   **Oscilloscope:**
    *   **Domain:** Analog (Voltage vs Time).
    *   **Resolution:** High Vertical (mV), High Bandwidth (MHz/GHz).
    *   **Use Case:** Signal Quality, Noise, Power Rails, Rise Time.
    *   **Channels:** Usually 2 or 4.
*   **Logic Analyzer:**
    *   **Domain:** Digital (1 or 0 vs Time).
    *   **Resolution:** 1-bit Vertical.
    *   **Use Case:** Protocol Decoding, Timing relationships, State Machines.
    *   **Channels:** 8, 16, or more.

### 🔹 Part 2: Sampling Rate & Aliasing
*   **Nyquist Theorem:** You must sample at least 2x the signal frequency.
*   **Rule of Thumb:** For digital signals, sample at **4x to 10x** the bit rate to see timing clearly.
    *   UART 115200 baud -> Sample > 1 MS/s.
    *   SPI 10 MHz -> Sample > 50 MS/s.

### 🔹 Part 3: Triggering
*   **Edge Trigger:** Start capturing when Channel 0 goes Low (e.g., UART Start Bit, I2C Start Condition).
*   **Pulse Width Trigger:** Capture only "Glitch" pulses < 10ns.
*   **Protocol Trigger:** Capture when I2C Address = 0x50.

---

## 💻 Implementation: Debugging I2C

> **Instruction:** Generate I2C traffic and capture it.

### 👨‍💻 Code Implementation

#### Step 1: Generate Traffic
```c
// Send data to CS43L22 (Addr 0x94)
void I2C_Test_Loop(void) {
    uint8_t data = 0xAF;
    while(1) {
        HAL_I2C_Master_Transmit(&hi2c1, 0x94, &data, 1, 100);
        HAL_Delay(10); // 10ms gap
    }
}
```

#### Step 2: Capture Setup
1.  Connect Logic Analyzer CH0 to SCL (PB6).
2.  Connect Logic Analyzer CH1 to SDA (PB9).
3.  Connect GND to GND.
4.  Software: Set Sample Rate to 24 MS/s (plenty for 100kHz I2C).
5.  Trigger: Falling Edge on SDA (Start Condition).

#### Step 3: Analysis
1.  **Start Condition:** SDA falls while SCL is High.
2.  **Address:** 1001010 (0x4A) + Write (0). Wait, 0x94 >> 1 = 0x4A. Correct.
3.  **ACK:** SDA pulled Low by Slave on 9th clock.
4.  **Data:** 10101111 (0xAF).
5.  **ACK:** SDA Low.
6.  **Stop Condition:** SDA rises while SCL is High.

---

## 🔬 Lab Exercise: Lab 87.1 - The Missing Pull-up

### 1. Lab Objectives
- Simulate a hardware fault.
- Identify it using the Logic Analyzer.

### 2. Step-by-Step Guide

#### Phase A: The Fault
1.  Configure PB6/PB9 as `GPIO_MODE_AF_PP` (Push-Pull) instead of `GPIO_MODE_AF_OD` (Open-Drain).
2.  Or, remove external pull-ups if using a breakout board (STM32 has internal, but weak).

#### Phase B: Capture
1.  Run the I2C code.
2.  Capture.
3.  **Observation:**
    *   **Open Drain without Pull-up:** Lines stay Low or float. No square waves.
    *   **Push-Pull:** Works, BUT dangerous if Slave stretches clock (Short circuit!).
    *   **Weak Pull-up:** Rise time is very slow (shark fin shape).

### 3. Verification
If you see "Shark Fins" (slow rise), calculate the RC time constant. $T = R \times C$.
*   Internal Pull-up $\approx 40k\Omega$.
*   Trace Capacitance $\approx 10pF$.
*   $T = 400ns$.
*   For 400kHz I2C ($T_{period} = 2.5\mu s$), 400ns is borderline. Use external $4.7k\Omega$.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Measuring ISR Latency
- **Goal:** How long does the interrupt take?
- **Task:**
    1.  `EXTI0_IRQHandler`:
        ```c
        GPIOD->BSRR = GPIO_PIN_12; // High
        // ... ISR Code ...
        GPIOD->BSRR = GPIO_PIN_12 << 16; // Low
        ```
    2.  Trigger EXTI.
    3.  Measure Pulse Width on Logic Analyzer.
    4.  **Result:** 5us? 10us? This is your ISR overhead + execution time.

### Lab 3: SPI Glitch Hunting
- **Goal:** Find why SPI fails at high speed.
- **Task:**
    1.  Run SPI at 42MHz (Max for APB2).
    2.  Capture with short leads.
    3.  Capture with long jumper wires (20cm).
    4.  **Observation:** Crosstalk or Reflections causing double clock edges.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Frame Error" in Decoder
*   **Cause:** Baud rate mismatch.
*   **Solution:** Use "Auto-Baud" in PulseView or measure the bit width manually ($1 / Width = Baud$).

#### 2. Ground Loops
*   **Cause:** PC USB Ground != Board Ground.
*   **Result:** Noise or blown USB port.
*   **Solution:** Use a USB Isolator or ensure common ground.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **GPIO Debugging:** Dedicate 2-3 pins solely for debugging.
    *   Pin 1: Task A Running.
    *   Pin 2: Task B Running.
    *   Pin 3: ISR Active.
    *   **Logic Analyzer:** Shows Task Switching and Preemption visually!

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why can't I see a 10ns glitch with a 24MS/s logic analyzer?
    *   **A:** 24MS/s = 41ns period. A 10ns pulse can happen entirely between two samples. You need > 200MS/s.
2.  **Q:** What is "Decoding"?
    *   **A:** Software processing that turns raw 1s and 0s into human readable text (e.g., "0xAF", "Hello").

### Challenge Task
> **Task:** Reverse Engineer a Remote Control. Connect an IR Receiver (TSOP) to the Logic Analyzer. Capture the waveform. Decode the protocol (NEC, RC5, or Sony).

---

## 📚 Further Reading & References
- [Sigrok / PulseView Wiki](https://sigrok.org/wiki/Main_Page)
- [Saleae Protocol Decoding Guide](https://support.saleae.com/protocol-analyzers/protocol-analyzers)

---
