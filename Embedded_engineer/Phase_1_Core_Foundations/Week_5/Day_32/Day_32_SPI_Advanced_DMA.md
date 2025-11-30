# Day 32: SPI Advanced (DMA)
## Phase 1: Core Embedded Engineering Foundations | Week 5: Serial Communication Protocols

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
1.  **Configure** SPI with DMA for high-throughput data transfer (e.g., Display updates).
2.  **Implement** a Circular DMA buffer to read sensor data continuously.
3.  **Interface** a TFT LCD (ILI9341) using SPI DMA.
4.  **Manage** Chip Select (CS) timing when using DMA (Hardware vs Software control).
5.  **Debug** DMA FIFO errors and bus contention.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   ILI9341 TFT Display (SPI) - Optional, or use Logic Analyzer
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 31 (SPI Basics)
    *   Day 13 (DMA)
*   **Datasheets:**
    *   [STM32F407 Reference Manual (SPI/DMA)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why DMA for SPI?
*   **Displays:** A 320x240 display has 76,800 pixels. At 16-bit color, that's 153 KB per frame. Sending this byte-by-byte with a CPU loop is slow and blocks the CPU.
*   **Sensors:** Reading an IMU at 1 kHz requires precise timing. DMA can grab the data automatically when triggered by a timer (if supported) or simply offload the transfer once initiated.

### 🔹 Part 2: SPI DMA Architecture
*   **TX DMA:** Moves data from RAM -> SPI_DR. Triggered when TXE=1.
*   **RX DMA:** Moves data from SPI_DR -> RAM. Triggered when RXNE=1.
*   **Circular Mode:** Useful for continuous sensor reading (if the sensor supports continuous stream mode).

### 🔹 Part 3: Chip Select Handling
DMA handles the data, but what about CS?
*   **Hardware NSS:** The SPI peripheral can control the NSS pin, but it usually only toggles it per *byte* or keeps it low forever. Not flexible for "Command + Data" sequences.
*   **Software CS:** We must pull CS Low, start DMA, wait for DMA Complete Interrupt, then pull CS High.

---

## 💻 Implementation: High-Speed Display Driver

> **Instruction:** We will simulate sending a frame buffer to an SPI Display using DMA.

### 🛠️ Hardware/System Configuration
*   **SPI1:** Master Mode.
*   **DMA:** DMA2 Stream 3 (TX).

### 👨‍💻 Code Implementation

#### Step 1: DMA Initialization (`spi_dma.c`)

```c
#include "stm32f4xx.h"

#define BUFFER_SIZE 320 // One Line of pixels

uint16_t frame_buffer[BUFFER_SIZE];

void DMA2_Stream3_Init(void) {
    RCC->AHB1ENR |= (1 << 22); // DMA2

    DMA2_Stream3->CR &= ~(1 << 0); // Disable
    while(DMA2_Stream3->CR & (1 << 0));

    DMA2_Stream3->PAR = (uint32_t)&(SPI1->DR);
    DMA2_Stream3->M0AR = (uint32_t)frame_buffer;
    DMA2_Stream3->NDTR = BUFFER_SIZE;

    // Channel 3 (011)
    // MSIZE 16-bit (01), PSIZE 16-bit (01) - SPI in 16-bit mode?
    // MINC (1), PINC (0)
    // Dir: M2P (01)
    // TCIE (1) - Interrupt on Complete
    DMA2_Stream3->CR = (3 << 25) | (1 << 13) | (1 << 11) | (1 << 10) | (1 << 6) | (1 << 4);
    
    NVIC_EnableIRQ(DMA2_Stream3_IRQn);
}
```

#### Step 2: SPI 16-bit Mode
To go fast, we can switch SPI to 16-bit data frame format (`DFF` bit).
```c
void SPI1_Fast_Init(void) {
    // ... Basic Init ...
    SPI1->CR1 |= (1 << 11); // DFF: 16-bit
    SPI1->CR2 |= (1 << 1);  // TXDMAEN: Tx Buffer DMA Enable
    SPI1->CR1 |= (1 << 6);  // SPE: Enable
}
```

#### Step 3: Send Function
```c
volatile uint8_t transfer_done = 1;

void SPI_DMA_Send(void) {
    if (!transfer_done) return; // Busy
    transfer_done = 0;
    
    CS_LOW();
    
    // Clear Flags
    DMA2->LIFCR = (1 << 27); // Clear TCIF3
    
    // Enable DMA
    DMA2_Stream3->CR |= (1 << 0);
}

void DMA2_Stream3_IRQHandler(void) {
    if (DMA2->LISR & (1 << 27)) { // TCIF3
        DMA2->LIFCR = (1 << 27);
        
        // Wait for SPI to finish sending the last byte!
        // DMA finishes when it puts the last byte in DR.
        // SPI is still shifting it out.
        while (SPI1->SR & (1 << 7)); // BUSY flag
        
        CS_HIGH();
        transfer_done = 1;
    }
}
```

#### Step 4: Main Loop
```c
int main(void) {
    // Fill Buffer with Red Color (RGB565)
    for(int i=0; i<BUFFER_SIZE; i++) frame_buffer[i] = 0xF800;
    
    SPI1_Fast_Init();
    DMA2_Stream3_Init();
    
    while(1) {
        SPI_DMA_Send();
        // CPU is free to calculate next line!
        while(!transfer_done);
    }
}
```

---

## 🔬 Lab Exercise: Lab 32.1 - Throughput Test

### 1. Lab Objectives
- Measure the actual data rate on the MOSI line.
- Compare CPU Copy vs DMA Copy.

### 2. Step-by-Step Guide

#### Phase A: CPU Copy
1.  Write a loop to send 1000 bytes.
2.  Toggle a GPIO before and after.
3.  Measure time.

#### Phase B: DMA Copy
1.  Setup DMA for 1000 bytes.
2.  Toggle GPIO before starting DMA.
3.  Toggle GPIO in DMA ISR.
4.  Measure time.

### 3. Verification
The DMA time should be slightly faster (no loop overhead), but more importantly, the CPU utilization during that time is 0%.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Continuous Sensor Reading
- **Goal:** Read 6 bytes (XYZ) from Accelerometer continuously.
- **Task:**
    1.  Configure SPI RX DMA (Circular Mode).
    2.  Set `NDTR` to 6.
    3.  Problem: You need to send dummy bytes to read.
    4.  Solution: Configure TX DMA *also* in Circular Mode to send dummy bytes continuously.
    5.  Sync: Connect a Timer Output to the SPI `NSS` pin? Or just let it run free (if sensor supports it).
    6.  Better: Use Timer Triggered DMA to start the transaction.

### Lab 3: Daisy Chain DMA
- **Goal:** Control 100 LEDs (WS2812 style, but using SPI).
- **Task:**
    1.  WS2812 protocol can be emulated with SPI MOSI.
    2.  0 bit = 11000000 (SPI byte).
    3.  1 bit = 11111100 (SPI byte).
    4.  Prepare a buffer in RAM. Use DMA to blast it out.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. CS goes High too early
*   **Cause:** Raising CS in the DMA ISR without checking `SPI_SR_BSY`.
*   **Detail:** DMA TC (Transfer Complete) means the last byte has left the *Memory* and entered the *SPI Data Register*. The SPI hardware still needs 8/16 clock cycles to shift it out.
*   **Solution:** `while (SPI1->SR & SPI_SR_BSY);` before `CS_HIGH()`.

#### 2. Data Corruption
*   **Cause:** Modifying the buffer while DMA is sending it.
*   **Solution:** Double Buffering or Wait for TC.

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **Priority:** Give SPI TX DMA high priority to prevent FIFO underrun (which causes gaps in transmission).

### Code Quality
- **Cache Coherency:** On Cortex-M7 (STM32F7), if you use D-Cache, you must clean the cache before starting TX DMA and invalidate cache after RX DMA. (Not an issue on F407, but good to know).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can you use DMA for SPI in Slave Mode?
    *   **A:** Yes! It's very useful for receiving data packets from a Master without interrupting the CPU for every byte.
2.  **Q:** What is the `DFF` bit?
    *   **A:** Data Frame Format. 0 = 8-bit, 1 = 16-bit.

### Challenge Task
> **Task:** Implement "Ping-Pong" Buffering. Buffer A is being sent to Display. CPU fills Buffer B. When DMA finishes A, switch DMA to B and CPU starts filling A. This allows continuous 60FPS animation.

---

## 📚 Further Reading & References
- [Mastering STM32 (Carmine Noviello)](https://leanpub.com/mastering-stm32) - Chapter on SPI & DMA.

---
