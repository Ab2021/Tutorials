# Day 34: I2C Advanced (DMA, Listeners)
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
1.  **Configure** I2C with DMA to offload large transfers (e.g., OLED buffer update).
2.  **Implement** an I2C Slave device that responds to Master requests.
3.  **Handle** complex I2C events (EV5, EV6, EV8) using interrupts.
4.  **Debug** I2C bus lockups and implement a bus recovery sequence.
5.  **Design** a robust state machine for non-blocking I2C communication.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Second STM32 Board (or Arduino) to act as Master/Slave partner.
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 33 (I2C Basics)
    *   Day 13 (DMA)
*   **Datasheets:**
    *   [STM32F407 Reference Manual (I2C DMA)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: I2C with DMA
I2C is slow (100-400 kHz). Sending 1KB takes ~25ms. Blocking the CPU for 25ms is unacceptable.
*   **TX DMA:** Triggered when `TXE` is set. Moves data from RAM to `DR`.
*   **RX DMA:** Triggered when `RXNE` is set. Moves data from `DR` to RAM.
*   **The Catch:** DMA handles the *Data*, but the CPU must still handle the *Events* (Start, Address, Stop).
    *   **Solution:** Use Interrupts for Events (EV) and DMA for Data.

### 🔹 Part 2: I2C Slave Mode
In Slave Mode, the STM32 waits for a Start condition matching its Own Address (`OAR1`).
*   **Events:**
    *   **ADDR:** Matched Address.
    *   **RXNE:** Master wrote data.
    *   **TXE:** Master wants to read data.
    *   **STOPF:** Stop condition detected.

---

## 💻 Implementation: I2C Slave Echo

> **Instruction:** We will configure the STM32 as an I2C Slave (Address 0x30). It will receive bytes and store them.

### 🛠️ Hardware/System Configuration
*   **I2C1:** Slave Mode.
*   **Master:** Another MCU sending data to 0x30.

### 👨‍💻 Code Implementation

#### Step 1: Slave Initialization (`i2c_slave.c`)

```c
#include "stm32f4xx.h"

void I2C1_Slave_Init(void) {
    // ... Clock & GPIO Init (Same as Day 33) ...
    RCC->AHB1ENR |= (1 << 1);
    RCC->APB1ENR |= (1 << 21);
    GPIOB->MODER |= (0xA000A000);
    GPIOB->OTYPER |= (1 << 6) | (1 << 9);
    GPIOB->AFR[0] |= (4 << 24);
    GPIOB->AFR[1] |= (4 << 4);

    I2C1->CR1 |= (1 << 15); // Reset
    I2C1->CR1 &= ~(1 << 15);

    I2C1->CR2 = 16; // 16 MHz
    I2C1->CCR = 80;
    I2C1->TRISE = 17;

    // Configure Own Address 1
    // Bit 15: AddrMode (0=7bit)
    // Bits 7-1: Address (0x30)
    I2C1->OAR1 = (0x30 << 1); 
    
    // Enable Events Interrupt & Error Interrupt
    I2C1->CR2 |= (1 << 9) | (1 << 8); // ITEVTEN, ITERREN
    // Note: ITBUFEN (Buffer Interrupt) is usually enabled too for TXE/RXNE
    I2C1->CR2 |= (1 << 10); 

    NVIC_EnableIRQ(I2C1_EV_IRQn);
    NVIC_EnableIRQ(I2C1_ER_IRQn);

    I2C1->CR1 |= (1 << 0); // Enable
    I2C1->CR1 |= (1 << 10); // ACK Enable
}
```

#### Step 2: Interrupt Handler
```c
volatile uint8_t rx_data;

void I2C1_EV_IRQHandler(void) {
    uint16_t sr1 = I2C1->SR1;
    uint16_t sr2 = I2C1->SR2;

    // 1. Address Matched (ADDR)
    if (sr1 & (1 << 1)) {
        // Cleared by reading SR1 followed by SR2 (Done above)
    }

    // 2. Data Received (RXNE)
    if (sr1 & (1 << 6)) {
        rx_data = I2C1->DR;
        // Process data...
    }

    // 3. Stop Detected (STOPF)
    if (sr1 & (1 << 4)) {
        // Cleared by reading SR1 then writing CR1
        I2C1->CR1 |= (1 << 0); 
    }
    
    // 4. Transmit Empty (TXE) - Master reading from us
    if (sr1 & (1 << 7)) {
        I2C1->DR = rx_data; // Echo back last received byte
    }
}

void I2C1_ER_IRQHandler(void) {
    // Handle Errors (Bus Error, Acknowledge Failure, etc.)
    if (I2C1->SR1 & (1 << 10)) { // AF (NACK)
        I2C1->SR1 &= ~(1 << 10); // Clear
    }
}
```

---

## 🔬 Lab Exercise: Lab 34.1 - DMA OLED Update

### 1. Lab Objectives
- Update a 128x64 OLED (SSD1306) using I2C DMA.
- Framebuffer size: 128 * 64 / 8 = 1024 bytes.

### 2. Step-by-Step Guide

#### Phase A: Config
1.  Initialize I2C1 Master.
2.  Initialize DMA1 Stream 6 Channel 1 (I2C1_TX).

#### Phase B: Sending Frame
```c
void OLED_Update_DMA(uint8_t *buffer) {
    // 1. Send Start + Address + Control Byte manually (or via DMA)
    // Usually easier to send the header manually, then DMA the 1024 bytes.
    
    I2C_Start();
    I2C_Address(0x78);
    I2C_Write(0x40); // Co=0, D/C=1 (Data)
    
    // 2. Setup DMA
    DMA1_Stream6->CR &= ~(1 << 0);
    DMA1_Stream6->M0AR = (uint32_t)buffer;
    DMA1_Stream6->NDTR = 1024;
    DMA1_Stream6->CR |= (1 << 0);
    
    // 3. Enable I2C DMA Request
    I2C1->CR2 |= (1 << 11); // DMAEN
    
    // 4. Wait for DMA TC (Transfer Complete)
    // 5. Send Stop
}
```

### 3. Verification
The screen should update instantly without CPU blocking.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Bus Recovery
- **Goal:** Unstick a hung bus.
- **Scenario:** Slave holds SDA Low and resets. Master sees SDA Low and thinks bus is busy.
- **Task:**
    1.  Detect Busy flag stuck for > 10ms.
    2.  Change SCL pin to GPIO Output.
    3.  Toggle SCL 9 times.
    4.  Change back to AF.
    5.  Send Stop.

### Lab 3: Multi-Master
- **Goal:** Two STM32s sharing a bus.
- **Task:**
    1.  Both configured as Master/Slave.
    2.  Randomly try to send data to each other.
    3.  Handle `ARLO` (Arbitration Lost) error. If lost, switch to Slave mode immediately.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. DMA stops halfway
*   **Cause:** NACK received from Slave. I2C hardware stops, but DMA doesn't know.
*   **Solution:** Check `AF` flag in I2C ISR. Disable DMA if error occurs.

#### 2. Slave not responding
*   **Cause:** `OAR1` not set correctly. Bit 0 is typically reserved/don't care in 7-bit mode, but the register alignment matters.
*   **Solution:** Use the macro `(Addr << 1)`.

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **DMA Priority:** I2C DMA should have lower priority than Audio/Video DMA, but higher than mem-to-mem.

### Code Quality
- **State Machine:** For complex I2C transactions (Start -> Write -> Restart -> Read -> Stop), use a software state machine driven by the I2C Event Interrupt. This is how professional drivers (like HAL) work.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is Clock Stretching?
    *   **A:** When a Slave holds SCL Low to tell the Master "Wait, I'm processing". The Master must wait until SCL goes High.
2.  **Q:** Can DMA handle the Start and Stop conditions?
    *   **A:** No. DMA only writes to `DR`. Start/Stop are control bits in `CR1`. You must trigger them via CPU (Interrupts).

### Challenge Task
> **Task:** Implement an "I2C Sniffer". Configure the STM32 to listen to *all* addresses (Promiscuous Mode? No, I2C doesn't have it easily). Workaround: Use two GPIO interrupts on SDA/SCL to capture the bit stream and decode it in software.

---

## 📚 Further Reading & References
- [Understanding I2C with DMA (ST AN2824)](https://www.st.com/resource/en/application_note/cd00209826-stm32f10xxx-i2c-optimized-examples-stmicroelectronics.pdf)

---
