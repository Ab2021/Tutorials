# Day 47: Ethernet MAC/PHY (MII/RMII)
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
1.  **Explain** the role of MAC (Media Access Control) and PHY (Physical Layer) in Ethernet.
2.  **Differentiate** between MII (Media Independent Interface) and RMII (Reduced MII).
3.  **Configure** the STM32 ETH MAC peripheral and DMA descriptors.
4.  **Communicate** with the PHY using SMI (MDC/MDIO) to read Link Status.
5.  **Send** a raw Ethernet Frame (ARP/UDP) without a stack.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board (Note: The base board has NO Ethernet. You need a breakout board like LAN8720 or DP83848, or use an STM32F407-EVAL/Nucleo-144 board which has Ethernet).
    *   Ethernet Cable + Router/Switch.
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   Wireshark (on PC)
*   **Prior Knowledge:**
    *   Day 13 (DMA)
    *   Day 15 (GPIO AF)
*   **Datasheets:**
    *   [LAN8720A Datasheet (Common PHY)](https://ww1.microchip.com/downloads/en/DeviceDoc/8720a.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: MAC vs PHY
*   **MAC (STM32 Internal):** Handles the logic. Framing, CRC generation, Address filtering, DMA.
*   **PHY (External Chip):** Handles the analog signals on the cable (100Base-TX). Encodes data (4B/5B, MLT-3).
*   **Interface:**
    *   **MII:** 17 pins. 25 MHz clock (for 100Mbps).
    *   **RMII:** 9 pins. 50 MHz clock. Saves GPIOs. (Used on most dev boards).

### 🔹 Part 2: SMI (Station Management Interface)
Also known as MDIO. It's like I2C (2 wires: MDC, MDIO). Used by MAC to configure PHY (e.g., Force 100Mbps, Auto-Negotiation, Check Link Status).

### 🔹 Part 3: DMA Descriptors
Ethernet DMA uses a Linked List of Descriptors in RAM.
*   **TDES (Tx Descriptor):** Points to buffer, contains status (Own bit).
*   **RDES (Rx Descriptor):** Points to buffer, contains status.
*   **Ownership:**
    *   Own=1: DMA owns it (can write/read).
    *   Own=0: CPU owns it (can process/fill).

---

## 💻 Implementation: PHY Initialization

> **Instruction:** We will initialize the MAC in RMII mode and talk to the LAN8720 PHY.

### 🛠️ Hardware/System Configuration
*   **RMII_REF_CLK:** PA1 (50 MHz from PHY or MCO).
*   **RMII_MDIO:** PA2.
*   **RMII_MDC:** PC1.
*   **RMII_CRS_DV:** PA7.
*   **RMII_RXD0:** PC4.
*   **RMII_RXD1:** PC5.
*   **RMII_TX_EN:** PB11.
*   **RMII_TXD0:** PB12.
*   **RMII_TXD1:** PB13.

### 👨‍💻 Code Implementation

#### Step 1: GPIO & Clock Init (`eth.c`)

```c
#include "stm32f4xx.h"

void ETH_GPIO_Init(void) {
    // Enable Clocks: GPIOA, GPIOB, GPIOC, ETHMAC, ETHMAC_TX, ETHMAC_RX
    RCC->AHB1ENR |= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 25) | (1 << 26) | (1 << 27);
    
    // Select RMII Mode (SYSCFG_PMC bit 23 = 1)
    RCC->APB2ENR |= (1 << 14); // SYSCFG
    SYSCFG->PMC |= (1 << 23);
    
    // Configure Pins as AF11 (ETH)
    // ... (PA1, PA2, PA7, PB11, PB12, PB13, PC1, PC4, PC5) ...
    // Set Speed to Very High for all pins!
}
```

#### Step 2: MAC Reset & Config
```c
void ETH_MAC_Init(void) {
    // 1. Software Reset
    ETH->DMABMR |= (1 << 0);
    while(ETH->DMABMR & (1 << 0));
    
    // 2. MAC Config
    // Speed 100M (FES), Full Duplex (DM)
    ETH->MACCR |= (1 << 14) | (1 << 11);
    // IPv4 Checksum Offload
    ETH->MACCR |= (1 << 10);
    
    // 3. SMI Clock (MDC)
    // HCLK = 168 MHz. Range 150-168 -> Div 102 (CR = 100)
    ETH->MACMIIAR &= ~(7 << 2);
    ETH->MACMIIAR |= (4 << 2);
}
```

#### Step 3: Read PHY Register
```c
uint16_t ETH_ReadPHY(uint16_t PHYAddress, uint16_t PHYReg) {
    // Set PHY Address and Reg
    ETH->MACMIIAR = (ETH->MACMIIAR & ~0xF800) | (PHYAddress << 11);
    ETH->MACMIIAR = (ETH->MACMIIAR & ~0x07C0) | (PHYReg << 6);
    
    // Start Read (MB = 1)
    ETH->MACMIIAR |= (1 << 0);
    
    // Wait for Busy (MB = 0)
    while(ETH->MACMIIAR & (1 << 0));
    
    return (uint16_t)(ETH->MACMIIDR);
}
```

---

## 🔬 Lab Exercise: Lab 47.1 - Link Status

### 1. Lab Objectives
- Detect if the Ethernet cable is plugged in.

### 2. Step-by-Step Guide

#### Phase A: Logic
1.  Read PHY Register 1 (Basic Status Register).
2.  Check Bit 2 (Link Status).
    *   1 = Link Up.
    *   0 = Link Down.

#### Phase B: Implementation
```c
int main(void) {
    ETH_GPIO_Init();
    ETH_MAC_Init();
    
    while(1) {
        // LAN8720 default address is usually 0 or 1
        uint16_t status = ETH_ReadPHY(0, 1);
        
        if (status & (1 << 2)) {
            GPIOD->ODR |= (1 << 12); // Green LED On
        } else {
            GPIOD->ODR &= ~(1 << 12); // Green LED Off
        }
        
        Delay_ms(500);
    }
}
```

### 3. Verification
Plug/Unplug cable. LED should toggle.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Sending a Raw Packet
- **Goal:** Send a broadcast packet.
- **Task:**
    1.  Setup TX Descriptor (TDES0, TDES1, TDES2, TDES3).
    2.  Fill buffer with `FF FF FF FF FF FF` (Dest MAC) + `Source MAC` + `Type` + `Data`.
    3.  Set `OWN` bit in TDES0.
    4.  Trigger DMA (`ETH->DMATPDR = 0`).
    5.  Check Wireshark for the packet.

### Lab 3: Promiscuous Mode
- **Goal:** Receive all traffic.
- **Task:**
    1.  Set `RA` (Receive All) bit in `ETH->MACFFR`.
    2.  Setup RX Descriptors.
    3.  Dump received headers to UART.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. No Link
*   **Cause:** 50 MHz Clock missing. RMII needs a perfect 50 MHz clock. If using MCO from STM32, ensure PLL is correct. If using crystal on PHY, ensure it's oscillating.
*   **Cause:** Reset pin on PHY not released.

#### 2. MDIO Read Fails (Returns 0xFFFF)
*   **Cause:** Wrong PHY Address. Try scanning 0-31.
*   **Cause:** MDC Clock too fast. Check `MACMIIAR` divider.

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **Checksum Offload:** Enable hardware calculation of IP/TCP/UDP checksums in `MACCR`. This saves a huge amount of CPU time.

### Code Quality
- **Descriptors:** Place Descriptors in non-cached memory (if using M7) or ensure proper alignment. On F4, standard SRAM is fine.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between MAC Address and IP Address?
    *   **A:** MAC is physical (Layer 2), fixed at factory. IP is logical (Layer 3), assigned by network (DHCP).
2.  **Q:** Why do we need magnetics (RJ45 jack with transformer)?
    *   **A:** Isolation. To prevent ground loops and protect the chip from voltage spikes on the long cable.

### Challenge Task
> **Task:** Implement "Cable Diagnostics". Some PHYs (like LAN8720) have TDR (Time Domain Reflectometry) registers to detect cable length and open/short faults. Read these registers and print cable status.

---

## 📚 Further Reading & References
- [STM32 Ethernet Application Note (AN3966)](https://www.st.com/resource/en/application_note/dm00036052-ethernet-interface-for-stm32-microcontrollers-stmicroelectronics.pdf)

---
