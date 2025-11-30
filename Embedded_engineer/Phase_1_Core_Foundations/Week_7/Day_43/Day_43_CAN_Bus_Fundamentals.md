# Day 43: CAN Bus Fundamentals
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
1.  **Explain** the Controller Area Network (CAN) physical layer (Differential Voltage).
2.  **Analyze** the CAN Frame format (ID, DLC, Data, CRC, ACK).
3.  **Understand** Arbitration and Priority (Lower ID = Higher Priority).
4.  **Calculate** Bit Timing segments (Sync, Prop, Phase1, Phase2) for a specific baud rate.
5.  **Configure** the STM32 bxCAN peripheral in Loopback Mode.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Optional: CAN Transceiver (TJA1050/SN65HVD230) for physical bus.
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 29 (UART - Asynchronous comms)
    *   Day 15 (GPIO)
*   **Datasheets:**
    *   [STM32F407 Reference Manual (bxCAN Section)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)
    *   [Bosch CAN Specification 2.0](https://www.bosch-semiconductors.com/media/ip_modules/pdf_2/can2spec.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Physical Layer
CAN uses **Differential Signaling** to be robust against noise.
*   **CAN_H and CAN_L:** Two wires.
*   **Recessive State (Logic 1):** CAN_H $\approx$ 2.5V, CAN_L $\approx$ 2.5V. Diff $\approx$ 0V. Bus is Idle.
*   **Dominant State (Logic 0):** CAN_H $\approx$ 3.5V, CAN_L $\approx$ 1.5V. Diff $\approx$ 2V.
*   **Wired-AND Logic:** If one node drives Dominant (0) and another drives Recessive (1), the bus becomes Dominant (0). This is crucial for arbitration.

### 🔹 Part 2: The CAN Frame
Standard CAN 2.0A (11-bit ID) vs 2.0B (29-bit ID).
1.  **SOF (Start of Frame):** 1 Dominant bit.
2.  **Arbitration Field:** 11-bit ID + RTR bit.
3.  **Control Field:** DLC (Data Length Code, 4 bits).
4.  **Data Field:** 0 to 8 bytes.
5.  **CRC Field:** 15-bit Checksum.
6.  **ACK Field:** Slot for receivers to acknowledge.
7.  **EOF (End of Frame):** 7 Recessive bits.

### 🔹 Part 3: Arbitration
What if two nodes start talking at once?
*   They both send their ID bit by bit.
*   They monitor the bus.
*   If Node A sends '1' (Recessive) but sees '0' (Dominant) on the bus, it knows Node B (sending '0') has won.
*   Node A stops transmitting immediately and starts listening.
*   **Result:** No data is lost. The higher priority message (Lower ID) gets through.

---

## 💻 Implementation: CAN Loopback Mode

> **Instruction:** We will configure CAN1 in Loopback Mode. This allows the MCU to receive its own messages without external transceivers.

### 🛠️ Hardware/System Configuration
*   **CAN1_RX:** PD0 (AF9).
*   **CAN1_TX:** PD1 (AF9).
*   **Mode:** Loopback (Silent).

### 👨‍💻 Code Implementation

#### Step 1: Initialization (`can.c`)

```c
#include "stm32f4xx.h"

void CAN1_Init(void) {
    // 1. Enable Clocks
    RCC->AHB1ENR |= (1 << 3);  // GPIOD
    RCC->APB1ENR |= (1 << 25); // CAN1

    // 2. Configure GPIO (PD0, PD1)
    GPIOD->MODER |= (2 << 0) | (2 << 2); // AF
    GPIOD->AFR[0] |= (9 << 0) | (9 << 4); // AF9

    // 3. Enter Initialization Mode
    CAN1->MCR |= (1 << 0); // INRQ
    while(!(CAN1->MSR & (1 << 0))); // Wait for INAK

    // 4. Exit Sleep Mode
    CAN1->MCR &= ~(1 << 1); // SLEEP
    while(CAN1->MSR & (1 << 1)); // Wait for SLAK clear

    // 5. Bit Timing (500 kbps)
    // APB1 = 42 MHz.
    // Prescaler = 3 -> Tq = 1/14 MHz = 71.4ns
    // Time Quanta per bit = 14M / 500k = 28.
    // Sync=1, BS1=18, BS2=9. Total = 1+18+9 = 28.
    // BTR Register:
    // SILM (Silent) | LBKM (Loopback) | SJW | TS2 | TS1 | BRP
    
    CAN1->BTR = (1 << 30) | // Loopback Mode
                (0 << 24) | // SJW = 1
                (8 << 20) | // TS2 = 9 (8+1)
                (17 << 16)| // TS1 = 18 (17+1)
                (2 << 0);   // BRP = 3 (2+1)

    // 6. Leave Init Mode
    CAN1->MCR &= ~(1 << 0);
    while(CAN1->MSR & (1 << 0));
}
```

#### Step 2: Filter Configuration
By default, CAN rejects everything. We must enable a filter.
```c
void CAN1_Filter_Init(void) {
    CAN1->FMR |= (1 << 0); // Filter Init Mode

    // Filter 0: Accept All
    CAN1->sFilterRegister[0].FR1 = 0; // ID Mask 0
    CAN1->sFilterRegister[0].FR2 = 0; // ID Mask 0
    
    CAN1->FA1R |= (1 << 0); // Activate Filter 0
    CAN1->FMR &= ~(1 << 0); // Active Mode
}
```

#### Step 3: Send Message
```c
void CAN1_Tx(uint16_t id, uint8_t *data, uint8_t len) {
    // Check if Mailbox 0 is empty
    if ((CAN1->TSR & (1 << 26))) {
        CAN1->sTxMailBox[0].TIR = (id << 21); // Std ID
        CAN1->sTxMailBox[0].TDTR = (len & 0xF);
        
        CAN1->sTxMailBox[0].TDLR = data[0] | (data[1]<<8) | (data[2]<<16) | (data[3]<<24);
        CAN1->sTxMailBox[0].TDHR = data[4] | (data[5]<<8) | (data[6]<<16) | (data[7]<<24);
        
        CAN1->sTxMailBox[0].TIR |= (1 << 0); // Request TX
    }
}
```

#### Step 4: Receive Message (Polling)
```c
void CAN1_Rx(void) {
    // Check FIFO 0
    if (CAN1->RF0R & 3) { // FMP0 > 0
        uint32_t id = (CAN1->sFIFOMailBox[0].RIR >> 21) & 0x7FF;
        uint32_t d_low = CAN1->sFIFOMailBox[0].RDLR;
        uint32_t d_high = CAN1->sFIFOMailBox[0].RDHR;
        
        printf("Rx ID: 0x%X, Data: %08X %08X\n", id, d_low, d_high);
        
        CAN1->RF0R |= (1 << 5); // Release FIFO
    }
}
```

---

## 🔬 Lab Exercise: Lab 43.1 - Loopback Test

### 1. Lab Objectives
- Verify that the CAN controller works.
- Send a message with ID 0x123 and Data "HELLO".

### 2. Step-by-Step Guide

#### Phase A: Main Loop
```c
int main(void) {
    CAN1_Init();
    CAN1_Filter_Init();
    
    uint8_t msg[] = {0x48, 0x45, 0x4C, 0x4C, 0x4F, 0, 0, 0};
    
    while(1) {
        CAN1_Tx(0x123, msg, 5);
        Delay_ms(100);
        CAN1_Rx();
    }
}
```

### 3. Verification
You should see "Rx ID: 0x123..." printed in the console.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Filter Masking
- **Goal:** Accept only IDs 0x100 to 0x10F.
- **Task:**
    1.  Set Filter ID = 0x100.
    2.  Set Filter Mask = 0x7F0 (Bits 10-4 must match).
    3.  Send 0x105 (Should Rx).
    4.  Send 0x200 (Should Ignore).

### Lab 3: Error Handling
- **Goal:** Trigger an error.
- **Task:**
    1.  Disable Loopback (`BTR` bit 30 = 0).
    2.  Don't connect a transceiver.
    3.  Try to TX.
    4.  Observe `LEC` (Last Error Code) in `ESR`. It should be "Acknowledgment Error" because no one ACKed.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Stuck in Init
*   **Cause:** `INRQ` not acknowledged.
*   **Solution:** Check if CAN1 clock is enabled.

#### 2. No RX
*   **Cause:** Filter not configured. By default, STM32 CAN discards all messages if no filter is active.
*   **Solution:** Enable Filter 0 in Mask Mode with 0/0 (Accept All).

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **Interrupts:** Use `CAN1_RX0_IRQn` instead of polling.
- **Mailboxes:** Use all 3 TX mailboxes to queue messages.

### Code Quality
- **Bit Timing:** Don't guess. Use a calculator (e.g., Kvaser Bit Timing Calculator) to find optimal BS1/BS2/SJW for your clock and sample point (usually 87.5%).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is the CAN frame max 8 bytes?
    *   **A:** To keep latency low. A high-priority message (e.g., Brake Command) shouldn't wait for a huge packet to finish.
2.  **Q:** What is "Bus Off"?
    *   **A:** If a node generates too many errors (TEC > 255), it disconnects itself from the bus to prevent bringing down the network.

### Challenge Task
> **Task:** Implement a "Heartbeat". Send a message ID 0x700 every 1 second. If the message is not received back (in Loopback), turn on the Error LED.

---

## 📚 Further Reading & References
- [CAN Bus Explained (CSS Electronics)](https://www.csselectronics.com/pages/can-bus-simple-intro-tutorial)

---
