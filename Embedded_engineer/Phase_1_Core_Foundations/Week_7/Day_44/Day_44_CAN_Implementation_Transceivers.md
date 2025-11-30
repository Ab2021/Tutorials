# Day 44: CAN Implementation (Transceivers)
## Phase 1: Core Embedded Engineering Foundations | Week 7: Advanced Peripherals

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
1.  **Interface** an STM32 with a CAN Transceiver (TJA1050, MCP2551, or SN65HVD230).
2.  **Connect** two nodes to form a physical CAN network.
3.  **Implement** a robust Interrupt-Driven CAN Driver.
4.  **Decode** CAN frames using a Logic Analyzer.
5.  **Troubleshoot** termination resistor issues (120Ω).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   2x STM32F4 Discovery Boards (or 1 Board + USB-CAN Adapter).
    *   2x CAN Transceiver Modules (3.3V compatible like SN65HVD230 is best, TJA1050 needs 5V but works with 3.3V logic usually).
    *   Twisted Pair Wire.
    *   2x 120Ω Resistors.
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 43 (CAN Basics)
*   **Datasheets:**
    *   [SN65HVD230 Datasheet](https://www.ti.com/lit/ds/symlink/sn65hvd230.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Transceiver
The STM32 CAN controller outputs logic levels (TX/RX). The Transceiver converts these to Differential Voltages (CANH/CANL).
*   **TX (MCU) -> D (Transceiver):** Logic 0 -> Dominant (Diff Voltage). Logic 1 -> Recessive (0V Diff).
*   **RX (MCU) <- R (Transceiver):** Reads the bus state.
*   **Loopback:** The Transceiver always echoes TX back to RX. The CAN controller monitors this to detect errors.

### 🔹 Part 2: Termination
*   **Reflections:** High-speed signals reflect off the ends of the wire.
*   **Solution:** Place a 120Ω resistor at **each end** of the bus (total 60Ω parallel).
*   **Stub Length:** Keep stubs (branches) short (< 30cm).

---

## 💻 Implementation: Two-Node Chat

> **Instruction:** Node A sends a counter. Node B receives it and toggles an LED.

### 🛠️ Hardware/System Configuration
*   **Node A:** ID 0x100.
*   **Node B:** ID 0x200.
*   **Wiring:**
    *   MCU TX -> Transceiver D.
    *   MCU RX -> Transceiver R.
    *   CANH -> CANH.
    *   CANL -> CANL.
    *   GND -> GND.
    *   120Ω Resistors at ends.

### 👨‍💻 Code Implementation

#### Step 1: Interrupt Driver (`can_irq.c`)

```c
#include "stm32f4xx.h"

void CAN1_Init_Normal(void) {
    // ... Same as Day 43, but:
    // 1. Loopback Mode OFF (Bit 30 = 0 in BTR).
    // 2. Silent Mode OFF (Bit 31 = 0).
    
    // Enable RX0 Interrupt
    CAN1->IER |= (1 << 1); // FMPIE0 (FIFO Message Pending)
    NVIC_EnableIRQ(CAN1_RX0_IRQn);
}

void CAN1_RX0_IRQHandler(void) {
    if (CAN1->RF0R & 3) {
        uint32_t id = (CAN1->sFIFOMailBox[0].RIR >> 21) & 0x7FF;
        uint32_t data = CAN1->sFIFOMailBox[0].RDLR;
        
        // Handle Message
        if (id == 0x100) {
            // Toggle LED
            GPIOD->ODR ^= (1 << 12);
        }
        
        // Release FIFO
        CAN1->RF0R |= (1 << 5);
    }
}
```

#### Step 2: Node A (Sender)
```c
int main(void) {
    CAN1_Init_Normal();
    CAN1_Filter_Init(); // Accept All
    
    uint32_t counter = 0;
    
    while(1) {
        uint8_t payload[4];
        payload[0] = counter & 0xFF;
        // ...
        
        CAN1_Tx(0x100, payload, 4);
        counter++;
        
        Delay_ms(500);
    }
}
```

#### Step 3: Node B (Receiver)
```c
int main(void) {
    CAN1_Init_Normal();
    CAN1_Filter_Init();
    
    while(1) {
        // Do nothing, ISR handles LED
    }
}
```

---

## 🔬 Lab Exercise: Lab 44.1 - Bus Sniffing

### 1. Lab Objectives
- Use a Logic Analyzer (Saleae/Clone) to view the CAN frames.
- Verify the Stuff Bits.

### 2. Step-by-Step Guide

#### Phase A: Setup
1.  Connect Logic Analyzer Ch0 to CAN RX pin (Digital).
2.  Connect Ch1 to CANH (Analog, if supported, or use Oscilloscope).

#### Phase B: Capture
1.  Trigger on Falling Edge (Start Bit).
2.  Decode using Protocol Analyzer (CAN, 500kbps).
3.  **Observation:** You should see ID 0x100, DLC 4, Data, CRC, ACK.
4.  **ACK Slot:** If Node B is connected, the ACK bit will be Dominant (Low). If disconnected, it will be Recessive (High), and Node A will retransmit forever (Error Frame).

### 3. Verification
If you see Error Frames (6 dominant bits in a row), check termination and baud rate.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Remote Request (RTR)
- **Goal:** Node B asks Node A for data.
- **Task:**
    1.  Node B sends a message with `RTR` bit set (Remote Transmission Request) and ID 0x100.
    2.  Node A sees RTR.
    3.  Node A immediately replies with a Data Frame ID 0x100.

### Lab 3: Bus Load Stress Test
- **Goal:** Flood the bus.
- **Task:**
    1.  Node A sends messages as fast as possible (`while(1)`).
    2.  Check `TSR` (Transmit Status Register) to ensure Mailbox is empty before writing.
    3.  Measure how many messages/sec Node B receives.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Stuff Error"
*   **Cause:** Noise on the bus interpreted as a violation of the bit-stuffing rule (5 consecutive bits of same polarity -> next must be opposite).
*   **Solution:** Check wiring, ground connection, and termination.

#### 2. Transceiver gets hot
*   **Cause:** Bus contention (Two nodes driving opposite states for too long? Unlikely with CAN).
*   **Cause:** Short circuit CANH to GND.

#### 3. 5V vs 3.3V
*   **Issue:** TJA1050 is 5V. STM32 is 3.3V.
*   **Solution:** TJA1050 RX pin output is 5V. STM32 pins are usually 5V tolerant, but check the datasheet! PD0/PD1 are 5V tolerant. Power TJA1050 with 5V.

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **Filters:** In a real car, there are thousands of messages. Set up hardware filters to only accept the IDs you care about (e.g., Engine RPM 0x201), otherwise the CPU will be overwhelmed by interrupts.

### Code Quality
- **Error Passive:** Monitor the `TEC` (Transmit Error Counter) and `REC`. If they increase, your bus is unhealthy. Log this.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What happens if I remove the termination resistors?
    *   **A:** For short cables (< 1m), it might work. For longer cables, reflections will corrupt the bits, causing Error Frames.
2.  **Q:** Can I use UART transceivers (RS485) for CAN?
    *   **A:** No. RS485 doesn't support the dominant/recessive arbitration (collision detection). CAN transceivers are specialized.

### Challenge Task
> **Task:** Implement a "Priority Inversion" demo. Configure Mailbox 0 with Low Priority ID (0x700) and Mailbox 1 with High Priority ID (0x100). Fill both. Verify that 0x100 is sent first.

---

## 📚 Further Reading & References
- [CAN Physical Layer and Termination Guide (TI)](https://www.ti.com/lit/an/slla270/slla270.pdf)

---
