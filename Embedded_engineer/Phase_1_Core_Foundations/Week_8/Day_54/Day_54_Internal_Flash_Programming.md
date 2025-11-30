# Day 54: Internal Flash Programming & Protection
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
1.  **Explain** the Flash memory organization (Main Memory, System Memory, OTP, Option Bytes).
2.  **Implement** "EEPROM Emulation" to store user settings in the last sector of Flash.
3.  **Secure** the firmware using Read Out Protection (RDP) Levels 1 and 2.
4.  **Configure** Write Protection (WRP) to prevent accidental erasure of the Bootloader.
5.  **Recover** a device from RDP Level 1 (Mass Erase).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   STM32CubeProgrammer (Essential for Option Bytes).
*   **Prior Knowledge:**
    *   Day 53 (Flash Programming)
*   **Datasheets:**
    *   [STM32F4 Flash Programming Manual (PM0081)](https://www.st.com/resource/en/programming_manual/pm0081-stm32f40xxx-and-stm32f41xxx-flash-memory-programming-stmicroelectronics.pdf)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Flash Organization
*   **Main Memory:** Where code lives. Divided into Sectors (16KB, 64KB, 128KB).
*   **System Memory:** Bootloader (Factory ROM). Cannot be erased.
*   **OTP (One-Time Programmable):** 528 bytes. Once written, cannot be erased. Good for Serial Numbers / Keys.
*   **Option Bytes:** Configuration bits loaded at reset.
    *   **RDP:** Read Protection.
    *   **WRP:** Write Protection.
    *   **BOR:** Brown Out Reset Level.

### 🔹 Part 2: Read Out Protection (RDP)
*   **Level 0:** No protection. Debugger has full access.
*   **Level 1:** Read Protection. Debugger cannot read Flash. If Debugger connects, it can only perform a **Mass Erase** (wiping the code) to unlock. This protects IP while allowing reuse.
*   **Level 2:** Chip Secured. Debugger disabled permanently. **Irreversible**.

---

## 💻 Implementation: EEPROM Emulation

> **Instruction:** Store a struct `Settings` in Sector 11 (Last 128KB sector on F407).

### 👨‍💻 Code Implementation

#### Step 1: Define Settings
```c
typedef struct {
    uint32_t magic; // 0xCAFEBABE
    uint32_t baud_rate;
    uint8_t  device_id;
    uint8_t  padding[3];
} Settings_t;

#define SETTINGS_ADDR 0x080E0000 // Sector 11 Start
```

#### Step 2: Save Settings
```c
void Save_Settings(Settings_t *cfg) {
    Flash_Unlock();
    
    // 1. Erase Sector 11
    Flash_EraseSector(11);
    
    // 2. Write Data
    uint32_t *p = (uint32_t*)cfg;
    for(int i=0; i<sizeof(Settings_t)/4; i++) {
        Flash_WriteWord(SETTINGS_ADDR + (i*4), p[i]);
    }
    
    Flash_Lock();
}
```

#### Step 3: Load Settings
```c
void Load_Settings(Settings_t *cfg) {
    Settings_t *flash_cfg = (Settings_t*)SETTINGS_ADDR;
    
    if (flash_cfg->magic == 0xCAFEBABE) {
        *cfg = *flash_cfg;
    } else {
        // Default
        cfg->magic = 0xCAFEBABE;
        cfg->baud_rate = 115200;
        cfg->device_id = 1;
    }
}
```

---

## 💻 Implementation: Option Bytes (RDP)

> **Instruction:** Change RDP Level via Code.

### 👨‍💻 Code Implementation

```c
void Set_RDP_Level1(void) {
    Flash_Unlock();
    
    // Unlock Option Bytes
    FLASH->OPTKEYR = 0x08192A3B;
    FLASH->OPTKEYR = 0x4C5D6E7F;
    
    while(FLASH->SR & (1 << 16)); // Wait BSY
    
    // Check current level
    if ((FLASH->OPTCR & (0xFF << 8)) != (0x55 << 8)) { // 0xAA=Lvl0, 0x55=Lvl1
        // Set Level 1
        FLASH->OPTCR &= ~(0xFF << 8);
        FLASH->OPTCR |= (0x55 << 8);
        
        // Start
        FLASH->OPTCR |= (1 << 1);
        while(FLASH->SR & (1 << 16));
        
        // Reload Option Bytes (System Reset)
        FLASH->OPTCR |= (1 << 0); // OPTSTRT
    }
    
    Flash_Lock();
}
```

---

## 🔬 Lab Exercise: Lab 54.1 - Secure the Device

### 1. Lab Objectives
- Enable RDP Level 1.
- Try to read Flash with CubeProgrammer.
- Perform Mass Erase to unlock.

### 2. Step-by-Step Guide

#### Phase A: Lock
1.  Run the `Set_RDP_Level1()` code.
2.  Board resets.

#### Phase B: Hack Attempt
1.  Open STM32CubeProgrammer.
2.  Connect.
3.  **Observation:** "Read Protection Active". Flash content is `?? ??`.
4.  You cannot debug.

#### Phase C: Unlock
1.  In CubeProgrammer, go to Option Bytes.
2.  Set RDP to Level 0 (AA).
3.  Apply.
4.  **Observation:** Chip performs Mass Erase. Flash is empty.

### 3. Verification
This confirms that your IP is safe. A thief can erase the chip to use it, but cannot steal your code.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Write Protection (WRP)
- **Goal:** Protect Bootloader (Sector 0) from accidental erase.
- **Task:**
    1.  Modify `nWRP` bits in `OPTCR`.
    2.  Bit 0 corresponds to Sector 0. Set to 0 (Protected).
    3.  Try to run `Flash_EraseSector(0)`.
    4.  **Result:** `WRPERR` (Write Protection Error) flag in `SR`.

### Lab 3: OTP Area
- **Goal:** Write a Serial Number.
- **Task:**
    1.  Write `0x12345678` to `0x1FFF 7800` (OTP Block 0).
    2.  Lock the block (Write 0x00 to Lock Byte).
    3.  Try to overwrite.
    4.  **Result:** Failed. Permanent ID.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. RDP Level 2 Accident
*   **Cause:** Writing 0xCC (Level 2) instead of 0x55.
*   **Result:** Brick. The chip is now a permanent black box. You cannot erase it. You cannot debug it.
*   **Solution:** Desolder and replace chip. **BE CAREFUL.**

#### 2. Option Byte Loading
*   **Cause:** Changes to Option Bytes don't take effect until `OBL_LAUNCH` bit is set or Power On Reset.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Wear Leveling:** Flash has ~10k erase cycles. If you save settings every minute, the sector dies in a week.
- **Solution:** Use a file system (LittleFS) or a circular buffer algorithm. Write new settings to next free address. Only erase when sector is full.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between RDP Level 1 and 2?
    *   **A:** Level 1 is reversible (via Mass Erase). Level 2 is permanent.
2.  **Q:** Can I protect just one sector from reading?
    *   **A:** No. RDP is global. PCROP (Proprietary Code Read Out Protection) exists on newer STM32s (F429, F7, L4) to execute-only specific sectors.

### Challenge Task
> **Task:** Implement "Wear Leveling". Allocate Sector 11. Write settings to offset 0. Next save, write to offset 32. Next, offset 64. When full, erase and start at 0.

---

## 📚 Further Reading & References
- [AN4701: Proprietary Code Read-Out Protection](https://www.st.com/resource/en/application_note/dm00186528-proprietary-code-readout-protection-on-stm32-microcontrollers-stmicroelectronics.pdf)

---
