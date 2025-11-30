# Day 52: Bootloader Fundamentals & Vector Table
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
1.  **Define** the role of a Bootloader in an embedded system.
2.  **Explain** the memory map of a system with a Bootloader and an Application.
3.  **Configure** the Vector Table Offset Register (`SCB->VTOR`) to relocate interrupts.
4.  **Implement** the "Jump to Application" logic (Stack Pointer & Program Counter).
5.  **Create** two separate linker scripts for Bootloader and Application.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   STM32CubeProgrammer (to view Flash memory).
*   **Prior Knowledge:**
    *   Day 10 (Memory Architecture)
    *   Day 9 (Cortex-M Registers)
*   **Datasheets:**
    *   [Cortex-M4 Generic User Guide (SCB Section)](https://developer.arm.com/documentation/dui0553/a/)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is a Bootloader?
A Bootloader is a small program that runs immediately after Reset. Its job is to:
1.  **Check** for a firmware update (via UART, USB, SD Card).
2.  **Update** the main application Flash if needed.
3.  **Jump** to the main application.

### 🔹 Part 2: Memory Map
The STM32F407 has 1MB Flash (`0x0800 0000` to `0x0810 0000`).
*   **Bootloader:** Sector 0 (16KB). `0x0800 0000` - `0x0800 4000`.
*   **Application:** Sector 1+ (Remaining). `0x0800 4000` - `0x0810 0000`.

### 🔹 Part 3: The Vector Table
When an interrupt occurs, the CPU looks at the Vector Table (start of Flash) to find the ISR address.
*   **Problem:** If the App is at `0x0800 4000`, its Vector Table is there too. But the CPU looks at `0x0000 0000` (aliased to `0x0800 0000`) by default.
*   **Solution:** `SCB->VTOR` (Vector Table Offset Register).
    *   The Bootloader leaves VTOR at 0.
    *   The Application **MUST** set `SCB->VTOR = 0x0800 4000` as its first instruction.

---

## 💻 Implementation: Linker Scripts

> **Instruction:** We need two projects. Project A (Bootloader) and Project B (App).

### 👨‍💻 Code Implementation

#### Step 1: Bootloader Linker (`bootloader.ld`)
Standard linker script.
```ld
MEMORY
{
  FLASH (rx)      : ORIGIN = 0x08000000, LENGTH = 16K
  RAM (xrw)       : ORIGIN = 0x20000000, LENGTH = 128K
}
```

#### Step 2: Application Linker (`app.ld`)
Offset by 16KB.
```ld
MEMORY
{
  FLASH (rx)      : ORIGIN = 0x08004000, LENGTH = 1008K
  RAM (xrw)       : ORIGIN = 0x20000000, LENGTH = 128K
}
```

---

## 💻 Implementation: The Jump Logic

> **Instruction:** This code goes in the Bootloader's `main()`.

### 👨‍💻 Code Implementation

```c
#include "stm32f4xx.h"

#define APP_ADDRESS 0x08004000

typedef void (*pFunction)(void);

void JumpToApp(void) {
    // 1. Check if the Application address is valid
    // The first word at APP_ADDRESS is the Stack Pointer (MSP).
    // It should point to RAM (0x2000xxxx).
    uint32_t msp = *(__IO uint32_t*)APP_ADDRESS;
    
    if ((msp & 0x2FF00000) == 0x20000000) {
        
        // 2. Disable all interrupts
        __disable_irq();
        
        // 3. Get the Reset Handler address (Second word)
        uint32_t reset_handler_addr = *(__IO uint32_t*)(APP_ADDRESS + 4);
        pFunction app_reset_handler = (pFunction)reset_handler_addr;
        
        // 4. Set the MSP (Main Stack Pointer)
        __set_MSP(msp);
        
        // 5. Jump!
        app_reset_handler();
    }
}

int main(void) {
    HAL_Init();
    
    // Blink LED to show Bootloader is running
    for(int i=0; i<5; i++) {
        Toggle_LED();
        Delay_ms(100);
    }
    
    JumpToApp();
    
    while(1); // Should not reach here
}
```

---

## 💻 Implementation: The Application Side

> **Instruction:** This code goes in the App's `SystemInit` or `main`.

### 👨‍💻 Code Implementation

```c
// In system_stm32f4xx.c or main.c

void SystemInit(void) {
    // ... Standard Init ...
    
    // Relocate Vector Table
    SCB->VTOR = 0x08004000; 
    
    // Re-enable interrupts (Bootloader disabled them)
    __enable_irq();
}

int main(void) {
    // Blink LED slowly to show App is running
    while(1) {
        Toggle_LED();
        Delay_ms(500);
    }
}
```

---

## 🔬 Lab Exercise: Lab 52.1 - The Dual Flash

### 1. Lab Objectives
- Flash the Bootloader.
- Flash the Application.
- Verify the Bootloader runs first, then jumps to App.

### 2. Step-by-Step Guide

#### Phase A: Build & Flash Bootloader
1.  Compile Bootloader.
2.  Flash to `0x0800 0000`.
3.  Reset. LED blinks fast (5 times). Then stops (because App is empty/invalid).

#### Phase B: Build & Flash App
1.  Compile App.
2.  **Important:** Flash to `0x0800 4000`.
    *   In VS Code / OpenOCD: `program app.elf verify reset exit 0x08004000` (Need to ensure offset is handled, usually ELF has address built-in).
    *   Or use STM32CubeProgrammer: Load ELF, it detects `0x0800 4000`.

#### Phase C: Verify
1.  Reset Board.
2.  **Observe:** Fast Blinks (Bootloader) -> Pause -> Slow Blinks (App).
3.  **Success!**

### 3. Verification
If it crashes:
*   Did you set `VTOR` in App?
*   Did you disable interrupts in Bootloader? (If a Systick fires just as you jump, and the App hasn't set up the vector table yet -> HardFault).

---

## 🧪 Additional / Advanced Labs

### Lab 2: Shared RAM
- **Goal:** Pass data from Bootloader to App.
- **Task:**
    1.  Define a specific RAM address (e.g., `0x2000 0000`) as "Shared".
    2.  Bootloader writes `0xDEADBEEF`.
    3.  App reads it. If match, print "Booted from BL".

### Lab 3: De-initialization
- **Goal:** Clean up before jump.
- **Task:**
    1.  In Bootloader, enable Timer 2.
    2.  Jump to App.
    3.  App tries to use Timer 2. It might behave oddly if BL left it running.
    4.  **Fix:** Always call `HAL_DeInit()` or manually reset RCC registers before jumping.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. HardFault immediately after Jump
*   **Cause:** `VTOR` not set. CPU tries to execute ISR using Bootloader's vector table (which might point to invalid addresses relative to App state).
*   **Cause:** Interrupts enabled during jump.

#### 2. App doesn't start
*   **Cause:** MSP check failed. Is the first word at `0x0800 4000` really the stack pointer? Check binary in CubeProgrammer.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Safety:** The Bootloader should verify the Integrity (CRC) of the Application before jumping. If CRC fails, stay in Bootloader mode (Safe Mode).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we need to set the MSP before jumping?
    *   **A:** The Application expects a fresh stack. If we use the Bootloader's dirty stack, we waste RAM and risk corruption.
2.  **Q:** Can the Bootloader be updated?
    *   **A:** Yes, but it's dangerous. If power fails while updating the Bootloader, the device is bricked (needs JTAG). Usually, Bootloaders are read-only.

### Challenge Task
> **Task:** Implement a "Button Entry". If Button is held during Reset, stay in Bootloader (Fast Blink loop). If not, Jump to App.

---

## 📚 Further Reading & References
- [ARM Cortex-M4 Processor Technical Reference Manual](https://developer.arm.com/documentation/100166/0001)

---
