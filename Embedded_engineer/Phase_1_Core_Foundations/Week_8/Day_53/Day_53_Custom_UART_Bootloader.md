# Day 53: Custom UART Bootloader Implementation
## Phase 1: Core Embedded Engineering Foundations | Week 8: Power Management & Bootloaders

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
1.  **Design** a communication protocol (XMODEM or Custom) for firmware transfer.
2.  **Erase** and **Program** the STM32 Internal Flash memory at runtime.
3.  **Implement** a UART Bootloader that accepts a `.bin` file and writes it to the Application region.
4.  **Verify** the integrity of the flashed firmware using CRC32.
5.  **Develop** a Python script to act as the Host Uploader.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   USB-UART Bridge (CP2102/FTDI).
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   Python 3 (pyserial).
*   **Prior Knowledge:**
    *   Day 52 (Bootloader Basics)
    *   Day 29 (UART)
*   **Datasheets:**
    *   [STM32F4 Flash Programming Manual (PM0081)](https://www.st.com/resource/en/programming_manual/pm0081-stm32f40xxx-and-stm32f41xxx-flash-memory-programming-stmicroelectronics.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Flash Programming
Writing to internal Flash is not like writing to RAM.
1.  **Unlock:** The Flash Control Register (`FLASH_CR`) is locked. Write keys (`KEY1`, `KEY2`) to `FLASH_KEYR`.
2.  **Erase:** You can only write '0's. To write '1's, you must erase the sector first (sets all bits to 1).
3.  **Program:** Write data (byte, half-word, word) to the address. Wait for `BSY` flag.
4.  **Lock:** Lock the Flash again to prevent accidental corruption.

### 🔹 Part 2: XMODEM Protocol
A simple, packet-based protocol from the 1970s.
*   **Packet:** `[SOH] [Blk#] [255-Blk#] [128 Bytes Data] [Checksum]`
*   **Handshake:** Receiver sends `NAK` to start. Sender sends Packet 1. Receiver sends `ACK`.
*   **End:** Sender sends `EOT`. Receiver sends `ACK`.

---

## 💻 Implementation: Flash Driver

> **Instruction:** Helper functions to erase/write Flash.

### 👨‍💻 Code Implementation

#### Step 1: Unlock/Lock
```c
#include "stm32f4xx.h"

void Flash_Unlock(void) {
    if (FLASH->CR & (1 << 31)) { // LOCK bit
        FLASH->KEYR = 0x45670123;
        FLASH->KEYR = 0xCDEF89AB;
    }
}

void Flash_Lock(void) {
    FLASH->CR |= (1 << 31);
}
```

#### Step 2: Erase Sector
```c
void Flash_EraseSector(uint8_t sector) {
    // Wait for Busy
    while(FLASH->SR & (1 << 16));
    
    // Set Sector (SNB bits 3-6)
    FLASH->CR &= ~(0xF << 3);
    FLASH->CR |= (sector << 3);
    
    // Set SER (Sector Erase)
    FLASH->CR |= (1 << 1);
    
    // Start
    FLASH->CR |= (1 << 16); // STRT
    
    // Wait for Busy
    while(FLASH->SR & (1 << 16));
    
    // Clear SER
    FLASH->CR &= ~(1 << 1);
}
```

#### Step 3: Program Word
```c
void Flash_WriteWord(uint32_t addr, uint32_t data) {
    while(FLASH->SR & (1 << 16));
    
    // Set PG (Programming)
    FLASH->CR |= (1 << 0);
    
    // Set PSIZE to x32 (10)
    FLASH->CR &= ~(3 << 8);
    FLASH->CR |= (2 << 8);
    
    // Write Data
    *(__IO uint32_t*)addr = data;
    
    while(FLASH->SR & (1 << 16));
    
    FLASH->CR &= ~(1 << 0);
}
```

---

## 💻 Implementation: UART Bootloader Logic

> **Instruction:** Simple protocol.
> 1. Host sends 'U' (Update).
> 2. MCU sends 'R' (Ready).
> 3. Host sends 4 bytes Size.
> 4. Host sends Data.
> 5. MCU writes to Flash.

### 👨‍💻 Code Implementation

```c
#define APP_START_ADDR 0x08004000

void Bootloader_Process(void) {
    uint8_t cmd;
    UART_Receive(&cmd, 1);
    
    if (cmd == 'U') {
        UART_Transmit("R", 1);
        
        // Receive Size
        uint32_t size;
        UART_Receive((uint8_t*)&size, 4);
        
        // Unlock Flash
        Flash_Unlock();
        
        // Erase App Sectors (Sector 1, 2, 3...)
        // Assume App fits in Sector 1 (16KB) for demo
        Flash_EraseSector(1);
        
        // Receive and Write
        uint32_t received = 0;
        uint32_t addr = APP_START_ADDR;
        
        while (received < size) {
            uint32_t data;
            // Receive 4 bytes (or pad if last chunk)
            UART_Receive((uint8_t*)&data, 4);
            
            Flash_WriteWord(addr, data);
            
            addr += 4;
            received += 4;
        }
        
        Flash_Lock();
        UART_Transmit("K", 1); // OK
    }
}
```

---

## 🔬 Lab Exercise: Lab 53.1 - The Python Host

### 1. Lab Objectives
- Write a Python script to send `app.bin` to the Bootloader.

### 2. Step-by-Step Guide

#### Phase A: Python Script (`uploader.py`)
```python
import serial
import struct
import sys

ser = serial.Serial('COM3', 115200, timeout=1)

with open('app.bin', 'rb') as f:
    data = f.read()
    size = len(data)

# Handshake
ser.write(b'U')
resp = ser.read(1)
if resp != b'R':
    print("No response")
    sys.exit()

# Send Size
ser.write(struct.pack('<I', size))

# Send Data
# Pad to 4 bytes
padding = (4 - (size % 4)) % 4
data += b'\xFF' * padding

for i in range(0, len(data), 4):
    chunk = data[i:i+4]
    ser.write(chunk)
    # Optional: Wait for ACK every chunk for reliability

resp = ser.read(1)
if resp == b'K':
    print("Update Success!")
else:
    print("Update Failed")
```

#### Phase B: Testing
1.  Flash Bootloader to STM32.
2.  Run `python uploader.py`.
3.  Observe STM32 LED behavior (should change from BL blink to App blink).

### 3. Verification
If it fails, check Endianness. STM32 is Little Endian. Python `struct.pack('<I')` ensures Little Endian.

---

## 🧪 Additional / Advanced Labs

### Lab 2: CRC Verification
- **Goal:** Ensure data integrity.
- **Task:**
    1.  Host calculates CRC32 of `app.bin`.
    2.  Sends CRC32 after data.
    3.  STM32 calculates CRC32 of Flash content.
    4.  If mismatch, erase Flash (don't jump).

### Lab 3: AES Decryption
- **Goal:** Secure Bootloader.
- **Task:**
    1.  Host encrypts `app.bin` -> `app.enc`.
    2.  STM32 receives `app.enc`.
    3.  Decrypts 16-byte block using AES Key (stored in BL).
    4.  Writes decrypted data to Flash.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Flash Write Error (PGSERR/PGAERR)
*   **Cause:** Writing to a not-erased address.
*   **Cause:** Writing byte to a device configured for word access (PSIZE).
*   **Solution:** Always erase before write. Check PSIZE matches Vcc (3.3V allows x32).

#### 2. UART Overrun
*   **Cause:** Python sends too fast. Flash write takes time (~16µs per word, but Erase takes ~1s).
*   **Solution:** Use Flow Control (RTS/CTS) or Software Handshake (ACK after every 1KB).

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Dual Bank:** High-end STM32s have Dual Bank Flash. You can write to Bank 2 while running from Bank 1. Then swap banks on Reset. This allows "Zero Downtime" updates.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we pad the data to 4 bytes?
    *   **A:** Because we are writing 32-bit words (`uint32_t`). Flash programming must be aligned.
2.  **Q:** What happens if power is lost during Erase?
    *   **A:** The App is gone. The Bootloader must detect this (e.g., check first word of App) and stay in Bootloader mode to allow retry.

### Challenge Task
> **Task:** Implement "YMODEM". It sends filename and size in Block 0. It allows multiple files. Use a terminal program (TeraTerm) to send the file instead of Python.

---

## 📚 Further Reading & References
- [XMODEM Protocol Specification](http://wiki.synchro.net/ref:xmodem)

---
