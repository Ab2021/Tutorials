# Day 13: DMA Controller
## Phase 1: Core Embedded Engineering Foundations | Week 2: ARM Cortex-M Architecture

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
1.  **Explain** the role of the DMA (Direct Memory Access) controller in offloading the CPU.
2.  **Configure** DMA Streams and Channels for Memory-to-Memory and Peripheral-to-Memory transfers.
3.  **Implement** Circular Mode for continuous data buffering (e.g., Audio or ADC).
4.  **Handle** DMA interrupts (Transfer Complete, Half Transfer, Error).
5.  **Solve** cache coherency issues (if applicable) and bus matrix contention.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 5 (Memory Map)
    *   Day 11 (Interrupts)
*   **Datasheets:**
    *   [STM32F407 Reference Manual (DMA Section)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The DMA Concept

#### 1.1 CPU vs. DMA
*   **CPU Copy:** To move data from ADC to RAM, the CPU reads a register, writes to RAM, increments pointers, and checks loop counters. This consumes 100% of CPU time during the transfer.
*   **DMA Copy:** The CPU configures the DMA (Source, Dest, Size) and says "Go". The DMA hardware takes over the bus, moves the data, and interrupts the CPU only when finished. The CPU is free to do other math or sleep.

#### 1.2 STM32F4 DMA Architecture
The STM32F4 has two DMA controllers:
*   **DMA1:** Connected to APB1 peripherals (UART, I2C, SPI, DAC). **Cannot** access AHB1/AHB2 (GPIO) or Memory-to-Memory.
*   **DMA2:** Connected to APB2 peripherals (ADC, SPI1) and can do **Memory-to-Memory**.

#### 1.3 Streams and Channels
*   **Streams:** Each DMA controller has 8 Streams (0-7). A Stream is an independent data path.
*   **Channels:** Each Stream can select from 8 Channels (0-7). A Channel corresponds to a specific peripheral request (e.g., Channel 4 on Stream 0 might be SPI3_RX).
*   **Arbiter:** Decides which Stream gets access to the bus if multiple are active.

```mermaid
graph LR
    Periph[Peripheral (ADC)] -->|Request| DMA[DMA Controller]
    DMA -->|Read| Periph
    DMA -->|Write| RAM[SRAM]
    CPU[CPU Core] -.->|Config| DMA
    DMA -.->|Interrupt| CPU
```

### 🔹 Part 2: DMA Configuration Modes

#### 2.1 Direction
*   **Peripheral-to-Memory (P2M):** ADC -> RAM.
*   **Memory-to-Peripheral (M2P):** RAM -> UART (TX).
*   **Memory-to-Memory (M2M):** Flash -> RAM (memcpy). Only DMA2.

#### 2.2 Circular Mode
The DMA automatically reloads the initial address and count when the transfer finishes. Essential for ring buffers (Audio, continuous ADC).

#### 2.3 FIFO vs. Direct Mode
*   **Direct Mode:** Data is transferred immediately.
*   **FIFO Mode:** DMA has a 4-word FIFO. Allows packing/unpacking (e.g., Source is byte, Dest is word).

### 🔹 Part 3: Interrupts
*   **TC (Transfer Complete):** All data moved.
*   **HT (Half Transfer):** Half the data moved. Useful for "Double Buffering" processing.
*   **TE (Transfer Error):** Bus error.
*   **DME (Direct Mode Error):** FIFO error.

---

## 💻 Implementation: Memory-to-Memory Transfer

> **Instruction:** We will use DMA2 to copy a large array from Flash to SRAM, measuring the time against a CPU `memcpy`.

### 🛠️ Hardware/System Configuration
STM32F4 Discovery.

### 👨‍💻 Code Implementation

#### Step 1: Configuration (`dma_driver.c`)

```c
#include "stm32f4xx.h"

// Copy 1KB of data
#define DATA_SIZE 256 // 256 words = 1024 bytes

const uint32_t src_data[DATA_SIZE] = {0xDEADBEEF, /* ... fill ... */ };
uint32_t dst_data[DATA_SIZE];

void DMA2_Stream0_Init(void) {
    // 1. Enable DMA2 Clock
    RCC->AHB1ENR |= (1 << 22);

    // 2. Disable Stream (must be disabled to configure)
    DMA2_Stream0->CR &= ~(1 << 0); // EN bit
    while(DMA2_Stream0->CR & (1 << 0));

    // 3. Configure Addresses
    DMA2_Stream0->PAR = (uint32_t)src_data; // Peripheral Address (Source in M2M)
    DMA2_Stream0->M0AR = (uint32_t)dst_data; // Memory 0 Address (Dest in M2M)
    DMA2_Stream0->NDTR = DATA_SIZE; // Number of items

    // 4. Configure Control Register (CR)
    // Channel 0 (Bits 25-27: 000)
    // MSIZE = 32-bit (Bits 13-14: 10)
    // PSIZE = 32-bit (Bits 11-12: 10)
    // MINC = Increment Memory (Bit 10: 1)
    // PINC = Increment Peripheral (Bit 9: 1) - Yes, for M2M both increment
    // DIR = Memory-to-Memory (Bits 6-7: 10)
    // TCIE = Transfer Complete Interrupt Enable (Bit 4: 1)
    
    DMA2_Stream0->CR = (0 << 25) | (2 << 13) | (2 << 11) | (1 << 10) | (1 << 9) | (2 << 6) | (1 << 4);

    // 5. Enable NVIC
    NVIC_EnableIRQ(DMA2_Stream0_IRQn);
}

void DMA_Start(void) {
    // Clear flags
    DMA2->LIFCR = 0x3D; // Clear all flags for Stream 0
    // Enable Stream
    DMA2_Stream0->CR |= (1 << 0);
}

volatile uint8_t transfer_complete = 0;

void DMA2_Stream0_IRQHandler(void) {
    // Check TC Flag (Bit 5 of LISR)
    if (DMA2->LISR & (1 << 5)) {
        // Clear Flag (Bit 5 of LIFCR)
        DMA2->LIFCR |= (1 << 5);
        transfer_complete = 1;
    }
}
```

#### Step 2: Main Loop
```c
int main(void) {
    DMA2_Stream0_Init();
    
    // Start Transfer
    DMA_Start();
    
    // CPU can do other things here...
    while(!transfer_complete);
    
    // Verify Data
    if (dst_data[0] == 0xDEADBEEF) {
        // Success
    }
    
    while(1);
}
```

---

## 🔬 Lab Exercise: Lab 13.1 - DMA vs CPU Race

### 1. Lab Objectives
- Measure the performance difference between DMA and CPU copy.

### 2. Step-by-Step Guide

#### Phase A: Setup
1.  Configure a GPIO pin (e.g., PD12) as Output.
2.  Create a large array (e.g., 4KB).

#### Phase B: CPU Test
1.  Set PD12 High.
2.  Run `memcpy(dst, src, 4096)`.
3.  Set PD12 Low.
4.  Measure pulse width with Logic Analyzer.

#### Phase C: DMA Test
1.  Set PD12 High.
2.  Start DMA.
3.  In DMA ISR, Set PD12 Low.
4.  Measure pulse width.

### 3. Verification
The DMA pulse might actually be *longer* for small data sizes due to setup overhead. But for large data, DMA wins. More importantly, during the DMA pulse, the CPU could be blinking another LED!

---

## 🧪 Additional / Advanced Labs

### Lab 2: UART TX with DMA
- **Goal:** Send "Hello World" continuously without blocking the CPU.
- **Task:**
    1.  Configure USART2 (PA2).
    2.  Configure DMA1 Stream 6 Channel 4 (USART2_TX).
    3.  Set Direction = Memory-to-Peripheral.
    4.  Enable `DMAT` bit in USART CR3.
    5.  Start DMA.

### Lab 3: Circular Buffer (ADC)
- **Goal:** Read ADC continuously into a buffer.
- **Task:**
    1.  Configure ADC1 in Continuous Mode.
    2.  Configure DMA2 Stream 0 Channel 0.
    3.  Set Circular Mode (CIRC bit).
    4.  Enable Half-Transfer (HT) and Transfer-Complete (TC) interrupts.
    5.  In HT ISR: Process first half of buffer.
    6.  In TC ISR: Process second half.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. DMA Not Starting
*   **Cause:**
    *   Forgot to enable DMA Clock in RCC.
    *   Stream was not disabled before configuration.
    *   Wrong Channel selected (check Mapping Table in Datasheet).
*   **Solution:** Check `EN` bit in CR. If it clears immediately, there's a configuration error (check `TEIF` flag).

#### 2. Data Corruption
*   **Cause:**
    *   Source/Dest size mismatch (MSIZE/PSIZE).
    *   Pointers not incrementing (MINC/PINC).
    *   Cache coherency (Cortex-M7 issue, less common on M4 unless using CCM RAM for DMA).
*   **Solution:** Verify alignment. Remember DMA cannot access CCM RAM!

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **Burst Mode:** Use DMA FIFO bursts (4 beats, 8 beats) to maximize bus usage.
- **Priority:** Give high-bandwidth peripherals (Camera, Display) "Very High" DMA priority.

### Code Quality
- **Alignment:** Ensure buffers are aligned to 4 bytes (`__attribute__((aligned(4)))`) to allow 32-bit transfers.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can DMA1 copy data from Flash to RAM?
    *   **A:** No. DMA1 is for APB1 peripherals. Only DMA2 can access the Memory bus for M2M transfers.
2.  **Q:** What is the advantage of Circular Mode?
    *   **A:** It allows continuous data processing (like audio streaming) without CPU intervention to reset the DMA pointers.

### Challenge Task
> **Task:** Implement a "Scatter-Gather" emulation. Use the Transfer Complete interrupt to reconfigure the DMA source address to a new buffer on the fly, creating a linked-list effect.

---

## 📚 Further Reading & References
- [STM32F4 DMA Controller Application Note (AN4031)](https://www.st.com/resource/en/application_note/dm00046011-using-the-stm32f2-stm32f4-and-stm32f7-series-dma-controller-stmicroelectronics.pdf)

---
