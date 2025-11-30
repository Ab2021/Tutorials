# Day 102: LoRa Fundamentals
## Phase 1: Core Embedded Engineering Foundations | Week 15: Wireless Communication Basics

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
1.  **Explain** the principles of LoRa (Long Range) modulation, including Chirp Spread Spectrum (CSS).
2.  **Analyze** the trade-offs between Spreading Factor (SF), Bandwidth (BW), and Data Rate.
3.  **Interface** an STM32 with a Semtech SX1278/SX1276 LoRa module via SPI.
4.  **Implement** a basic Point-to-Point (P2P) communication system ("Ping-Pong").
5.  **Calculate** Time on Air (ToA) to ensure regulatory compliance (Duty Cycle).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   2x STM32F4 Discovery Boards (or 1x + another LoRa device).
    *   2x SX1278 (433MHz) or SX1276 (868/915MHz) LoRa Modules.
    *   Antennas (Wire or SMA).
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 29 (SPI) - Critical.
    *   Day 99 (RF Basics).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Chirp Spread Spectrum (CSS)
LoRa doesn't use fixed frequencies for 0 and 1. It uses **Chirps**.
*   **Up-Chirp:** Frequency sweeps from $f_{min}$ to $f_{max}$. Represents a symbol.
*   **Down-Chirp:** Used for Preamble detection.
*   **Immunity:** Narrowband noise (like a single tone interference) only affects a tiny slice of the chirp. The receiver integrates the energy over the whole chirp and ignores the noise. This allows LoRa to receive signals **below the noise floor** (-20dB SNR!).

### 🔹 Part 2: LoRa Parameters
1.  **Spreading Factor (SF):** SF7 to SF12.
    *   SF7: Fast chirps, high data rate (5kbps), lower range.
    *   SF12: Slow chirps, low data rate (300bps), max range.
    *   Each step up doubles the Time on Air.
2.  **Bandwidth (BW):** Usually 125kHz, 250kHz, or 500kHz.
    *   Wider BW = Faster data, but more noise (less sensitivity).
3.  **Coding Rate (CR):** 4/5, 4/6, 4/7, 4/8.
    *   Forward Error Correction (FEC). 4/8 means 4 data bits + 4 redundancy bits.

### 🔹 Part 3: LoRa vs LoRaWAN
*   **LoRa:** The Physical Layer (PHY). Modulation only. P2P possible.
*   **LoRaWAN:** The MAC Layer (Network). Gateways, Servers, Encryption, Class A/B/C.
*   **Today:** We focus on raw LoRa PHY (P2P).

```mermaid
graph LR
    Tx[Transmitter] -->|SPI| SX1278_Tx
    SX1278_Tx -->|Chirps| Air
    Air -->|Chirps| SX1278_Rx
    SX1278_Rx -->|SPI| Rx[Receiver]
```

---

## 💻 Implementation: SX127x Driver

> **Instruction:** Build a minimal driver to send a packet.

### 👨‍💻 Code Implementation

#### Step 1: Registers (sx1278.h)
```c
#define REG_FIFO        0x00
#define REG_OP_MODE     0x01
#define REG_FRF_MSB     0x06
#define REG_PA_CONFIG   0x09
#define REG_IRQ_FLAGS   0x12
#define REG_FIFO_ADDR_PTR 0x0D

#define MODE_SLEEP      0x00
#define MODE_STDBY      0x01
#define MODE_TX         0x03
#define MODE_RX_CONT    0x05
#define MODE_LORA       0x80
```

#### Step 2: Initialization
```c
void LoRa_Init(void) {
    // 1. Reset Module
    HAL_GPIO_WritePin(RST_PORT, RST_PIN, 0);
    HAL_Delay(1);
    HAL_GPIO_WritePin(RST_PORT, RST_PIN, 1);
    HAL_Delay(10);
    
    // 2. Set Sleep Mode to Switch to LoRa
    WriteReg(REG_OP_MODE, MODE_SLEEP | MODE_LORA);
    
    // 3. Set Frequency (433MHz)
    // Frf = (Freq * 2^19) / 32MHz
    uint64_t frf = ((uint64_t)433000000 * 524288) / 32000000;
    WriteReg(REG_FRF_MSB, (frf >> 16) & 0xFF);
    WriteReg(REG_FRF_MID, (frf >> 8) & 0xFF);
    WriteReg(REG_FRF_LSB, frf & 0xFF);
    
    // 4. Config PA (Power)
    WriteReg(REG_PA_CONFIG, 0xFF); // Max Power (+17dBm)
    
    // 5. Standby
    WriteReg(REG_OP_MODE, MODE_STDBY | MODE_LORA);
}
```

#### Step 3: Transmit
```c
void LoRa_Tx(uint8_t *data, uint8_t len) {
    // 1. Standby
    WriteReg(REG_OP_MODE, MODE_STDBY | MODE_LORA);
    
    // 2. Set FIFO Ptr
    WriteReg(REG_FIFO_ADDR_PTR, 0);
    WriteReg(REG_PAYLOAD_LENGTH, len);
    
    // 3. Fill FIFO
    for(int i=0; i<len; i++) WriteReg(REG_FIFO, data[i]);
    
    // 4. Start TX
    WriteReg(REG_OP_MODE, MODE_TX | MODE_LORA);
    
    // 5. Wait for TxDone
    while(!(ReadReg(REG_IRQ_FLAGS) & 0x08));
    
    // 6. Clear IRQ
    WriteReg(REG_IRQ_FLAGS, 0x08);
}
```

#### Step 4: Receive (Polling)
```c
int LoRa_Rx(uint8_t *buf) {
    // 1. Set RX Continuous
    WriteReg(REG_OP_MODE, MODE_RX_CONT | MODE_LORA);
    
    // 2. Check RxDone
    if (ReadReg(REG_IRQ_FLAGS) & 0x40) {
        // Clear IRQ
        WriteReg(REG_IRQ_FLAGS, 0x40);
        
        // Read Length & Ptr
        uint8_t len = ReadReg(REG_RX_NB_BYTES);
        uint8_t ptr = ReadReg(REG_FIFO_RX_CURRENT_ADDR);
        WriteReg(REG_FIFO_ADDR_PTR, ptr);
        
        // Read FIFO
        for(int i=0; i<len; i++) buf[i] = ReadReg(REG_FIFO);
        
        return len;
    }
    return 0;
}
```

---

## 🔬 Lab Exercise: Lab 102.1 - Ping Pong

### 1. Lab Objectives
- Device A sends "PING".
- Device B receives "PING", waits 500ms, sends "PONG".
- Device A receives "PONG", waits 500ms, sends "PING".

### 2. Step-by-Step Guide

#### Phase A: Logic
```c
void App_Loop(void) {
    uint8_t buf[32];
    
    if (IsMaster) {
        LoRa_Tx("PING", 4);
        HAL_Delay(1000); // Wait for reply
        int len = LoRa_Rx(buf);
        if (len > 0 && strncmp(buf, "PONG", 4) == 0) {
            BSP_LED_Toggle(LED_GREEN);
        }
    } else {
        int len = LoRa_Rx(buf);
        if (len > 0 && strncmp(buf, "PING", 4) == 0) {
            BSP_LED_Toggle(LED_BLUE);
            HAL_Delay(100);
            LoRa_Tx("PONG", 4);
        }
    }
}
```

#### Phase B: Run
1.  Flash Board A as Master.
2.  Flash Board B as Slave.
3.  **Observation:** LEDs toggle in sync.
4.  **Range Test:** Move Board B to another room. It should still work.

### 3. Verification
If no communication:
*   Check Frequency (must match).
*   Check Sync Word (Reg 0x39). Default is 0x12.
*   Check Antenna connection (Do not transmit without antenna! Can damage PA).

---

## 🧪 Additional / Advanced Labs

### Lab 2: CAD (Channel Activity Detection)
- **Goal:** Save power.
- **Task:**
    1.  Instead of RX Continuous (High Power), use CAD.
    2.  Module wakes up, listens for Chirp preamble.
    3.  If detected -> Switch to RX.
    4.  If not -> Sleep.

### Lab 3: Variable Data Rate
- **Goal:** Adaptive Data Rate (ADR).
- **Task:**
    1.  Start at SF7.
    2.  If RSSI is low (< -110dBm), switch to SF8.
    3.  If RSSI is high (> -80dBm), switch to SF7.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. SPI Comms Fail
*   **Cause:** Module not powered or Reset pin held low.
*   **Solution:** Check wiring. Read Reg 0x42 (Version). Should be 0x12.

#### 2. CRC Error
*   **Cause:** Interference or distance too far.
*   **Solution:** Check `PayloadCrcError` flag in IRQ register. Enable `PayloadCrcOn` in Reg 0x1E.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **DIO Pins:** Use DIO0 pin interrupt for RxDone/TxDone instead of polling. This allows the MCU to sleep during transmission.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the relationship between SF and Range?
    *   **A:** Higher SF = Higher Sensitivity = Longer Range = Lower Data Rate = Higher Energy consumption.
2.  **Q:** Why is "Time on Air" important?
    *   **A:** ISM bands (868MHz) have strict Duty Cycle limits (e.g., 1%). If ToA is 1s, you can only send once every 100s.

### Challenge Task
> **Task:** "LoRa Chat". Connect UART to PC. Type message -> Send via LoRa -> Receive on other PC. Implement a simple addressing scheme `[To:1][From:2][Msg...]`.

---

## 📚 Further Reading & References
- [Semtech SX1276 Datasheet](https://www.semtech.com/products/wireless-rf/lora-core/sx1276)
- [LoRa Modulation Basics (App Note)](https://www.semtech.com/uploads/documents/an1200.22.pdf)

---
