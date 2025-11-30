# Day 41: SD Cards & File Systems
## Phase 1: Core Embedded Engineering Foundations | Week 6: Sensors and Actuators

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
1.  **Explain** the SD Card communication modes (SPI vs SDIO).
2.  **Initialize** an SD Card in SPI Mode (CMD0, CMD8, CMD55, ACMD41).
3.  **Read/Write** raw blocks (512 bytes) to the card.
4.  **Integrate** the ElmChan FatFs library to manage files (TXT, CSV).
5.  **Log** sensor data to a file (`log.csv`) periodically.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   MicroSD Card Module (SPI Interface)
    *   MicroSD Card (formatted FAT32, < 32GB)
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   [FatFs Library Source](http://elm-chan.org/fsw/ff/00index_e.html)
*   **Prior Knowledge:**
    *   Day 31 (SPI)
*   **Datasheets:**
    *   [SD Physical Layer Simplified Spec](https://www.sdcard.org/downloads/pls/)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: SD Card Protocol
*   **SDIO:** 4-bit parallel, fast. Native on STM32F4 (SDIO peripheral).
*   **SPI:** 1-bit serial, slower, but works on any MCU. We use SPI for simplicity and compatibility.
*   **Initialization:** Complex state machine.
    1.  Power Up (Clock 80 cycles).
    2.  CMD0 (Go Idle).
    3.  CMD8 (Check Voltage).
    4.  ACMD41 (Initialize).
    5.  CMD58 (Read OCR).

### 🔹 Part 2: File Systems (FAT)
Writing raw blocks is hard to manage on a PC. We use FAT32.
*   **FatFs:** Generic FAT file system module for embedded systems.
*   **DiskIO:** The low-level driver we must write to glue FatFs to our SPI driver.

---

## 💻 Implementation: Low-Level SPI SD Driver

> **Instruction:** SPI1 (PA5/PA6/PA7) + CS (PB0).

### 👨‍💻 Code Implementation

#### Step 1: Send Command (`sd_spi.c`)

```c
uint8_t SD_SendCmd(uint8_t cmd, uint32_t arg, uint8_t crc) {
    SD_CS_LOW();
    SPI_Transfer(cmd | 0x40); // Start bit + Index
    SPI_Transfer(arg >> 24);
    SPI_Transfer(arg >> 16);
    SPI_Transfer(arg >> 8);
    SPI_Transfer(arg);
    SPI_Transfer(crc);
    
    uint8_t res;
    int timeout = 10;
    do {
        res = SPI_Transfer(0xFF);
    } while ((res & 0x80) && --timeout);
    
    return res;
}
```

#### Step 2: Initialization
```c
uint8_t SD_Init(void) {
    uint8_t res;
    
    // 1. Power Up: 80 clocks with CS High
    SD_CS_HIGH();
    for(int i=0; i<10; i++) SPI_Transfer(0xFF);
    
    // 2. CMD0: Go Idle
    SD_CS_LOW();
    res = SD_SendCmd(0, 0, 0x95);
    SD_CS_HIGH();
    if (res != 1) return 0; // Error
    
    // ... (Full sequence CMD8, ACMD41) ...
    
    return 1; // OK
}
```

---

## 💻 Implementation: FatFs Integration

#### Step 1: `diskio.c`
Map FatFs calls to our SD functions.
```c
DRESULT disk_read(BYTE pdrv, BYTE *buff, LBA_t sector, UINT count) {
    if (pdrv) return RES_PARERR;
    // Call SD_ReadBlock(sector, buff, count);
    return RES_OK;
}
```

#### Step 2: Main Application
```c
#include "ff.h"

FATFS fs;
FIL file;
FRESULT res;

int main(void) {
    SPI_Init();
    
    // Mount
    if (f_mount(&fs, "", 1) == FR_OK) {
        // Open
        if (f_open(&file, "test.txt", FA_WRITE | FA_CREATE_ALWAYS) == FR_OK) {
            // Write
            f_printf(&file, "Hello SD Card! Timestamp: %d\n", HAL_GetTick());
            // Close
            f_close(&file);
        }
    }
    
    while(1);
}
```

---

## 🔬 Lab Exercise: Lab 41.1 - Data Logger

### 1. Lab Objectives
- Log Accelerometer data to `log.csv`.

### 2. Step-by-Step Guide

#### Phase A: Format
CSV format: `Time,X,Y,Z`.

#### Phase B: Loop
```c
while(1) {
    IMU_Read(&x, &y, &z);
    
    f_open(&file, "log.csv", FA_WRITE | FA_OPEN_APPEND);
    f_printf(&file, "%d,%d,%d,%d\n", ms_ticks, x, y, z);
    f_close(&file); // Close to save data
    
    Delay_ms(100);
}
```

### 3. Verification
Remove card, read on PC. Open in Excel.

---

## 🧪 Additional / Advanced Labs

### Lab 2: High Speed Logging
- **Goal:** Avoid opening/closing every time.
- **Task:**
    1.  Open file once.
    2.  Write data.
    3.  `f_sync(&file)` every 10 writes (flushes buffer to disk).
    4.  Much faster.

### Lab 3: Config File
- **Goal:** Read settings.
- **Task:**
    1.  Create `config.txt`: `RATE=100`.
    2.  Read file, parse integer.
    3.  Set Sensor Sample Rate.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Mount Failed (FR_NOT_READY)
*   **Cause:** SD Init failed. Check wiring. Check 3.3V supply (SD cards draw 100mA+ peaks).

#### 2. Write Error
*   **Cause:** Card is Write Protected (Switch on side).

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **Multi-Block Write:** Use `CMD25` for writing many sectors. Much faster than single block `CMD24`. FatFs handles this if `disk_write` supports count > 1.

### Code Quality
- **DMA:** Use DMA for the 512-byte block transfers.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is SPI slower than SDIO?
    *   **A:** SPI is 1 bit. SDIO is 4 bits. SDIO also supports higher clock speeds (up to 50 MHz or more).
2.  **Q:** What is the purpose of `f_sync`?
    *   **A:** To flush the RAM buffer to the physical card without closing the file handle.

### Challenge Task
> **Task:** Implement a "Bootloader". On startup, check if `firmware.bin` exists on SD. If yes, flash it to internal program memory and reboot.

---

## 📚 Further Reading & References
- [FatFs Module User's Guide](http://elm-chan.org/fsw/ff/00index_e.html)

---
