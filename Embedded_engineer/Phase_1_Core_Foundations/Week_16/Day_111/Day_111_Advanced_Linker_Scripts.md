# Day 111: Advanced Linker Scripts
## Phase 1: Core Embedded Engineering Foundations | Week 16: Advanced C & Optimization

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
1.  **Define** multiple memory regions (CCM RAM, Backup SRAM, External SDRAM) in the linker script.
2.  **Place** specific functions or variables into these regions using `__attribute__((section))`.
3.  **Implement** "Overlays" to run code from RAM that is larger than the available RAM size.
4.  **Export** symbols from the linker script to C code (e.g., `_estack`, `_sdata`).
5.  **Protect** critical memory regions using MPU (Memory Protection Unit) alignment rules.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 10 (Memory Architecture)
    *   Day 109 (Boot Time - Copy Loop)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Memory Regions
STM32F407 has:
*   **Flash:** 1MB (0x08000000).
*   **SRAM1/2:** 112KB + 16KB (0x20000000).
*   **CCM RAM:** 64KB (0x10000000). Core Coupled Memory. Fast, but **no DMA access**.
*   **Backup SRAM:** 4KB (0x40024000).

### 🔹 Part 2: The `MEMORY` Command
```ld
MEMORY
{
  FLASH (rx)      : ORIGIN = 0x08000000, LENGTH = 1024K
  RAM (xrw)       : ORIGIN = 0x20000000, LENGTH = 128K
  CCMRAM (xrw)    : ORIGIN = 0x10000000, LENGTH = 64K
}
```

### 🔹 Part 3: The `SECTIONS` Command
We define output sections (`.text`, `.data`) and map them to memory regions.
*   `> RAM AT > FLASH`: Load Address (LMA) is Flash, Virtual Address (VMA) is RAM. This triggers the copy loop.

---

## 💻 Implementation: Using CCM RAM

> **Instruction:** Place a high-performance math function in CCM RAM.

### 👨‍💻 Code Implementation

#### Step 1: Linker Script Update (`stm32f4.ld`)
Add a `.ccmram` section.
```ld
.ccmram :
{
  . = ALIGN(4);
  _sccmram = .;       /* Create a global symbol at start */
  *(.ccmram)
  *(.ccmram*)
  . = ALIGN(4);
  _eccmram = .;       /* Create a global symbol at end */
} > CCMRAM AT > FLASH /* Load in Flash, Run in CCM */

_siccmram = LOADADDR(.ccmram); /* Where in Flash is it? */
```

#### Step 2: Startup Code Update (`startup_stm32.s` or `system_stm32.c`)
We must copy the code from Flash to CCM RAM at boot!
```c
void SystemInit(void) {
    // ... Existing Copy Loop ...
    
    // Copy CCMRAM
    extern uint32_t _siccmram, _sccmram, _eccmram;
    uint32_t *src = &_siccmram;
    uint32_t *dst = &_sccmram;
    while(dst < &_eccmram) {
        *dst++ = *src++;
    }
}
```

#### Step 3: C Code Usage
```c
// Place this function in CCM RAM
__attribute__((section(".ccmram")))
void FFT_Process(float *data) {
    // Fast math here...
}

// Place this array in CCM RAM
__attribute__((section(".ccmram")))
float fft_buffer[1024];
```

---

## 💻 Implementation: Linker Symbols in C

> **Instruction:** Calculate stack usage dynamically using linker symbols.

### 👨‍💻 Code Implementation

#### Step 1: Declare Symbols
In C, linker symbols are addresses, not variables.
```c
extern uint32_t _estack; // End of RAM (Top of Stack)
extern uint32_t _ebss;   // End of BSS (Bottom of Heap)
```

#### Step 2: Calculate Free Space
```c
void Print_Memory_Info(void) {
    uint32_t stack_top = (uint32_t)&_estack;
    uint32_t heap_bottom = (uint32_t)&_ebss;
    
    printf("Stack Top: 0x%08lX\n", stack_top);
    printf("Heap Bottom: 0x%08lX\n", heap_bottom);
    printf("Total RAM Available for Heap/Stack: %lu bytes\n", stack_top - heap_bottom);
}
```
**Note:** Use `&` operator. `_estack` is a symbol whose *address* is the value we want. `_estack`'s *value* is whatever is at that memory location (garbage).

---

## 🔬 Lab Exercise: Lab 111.1 - The DMA Trap

### 1. Lab Objectives
- Attempt to use DMA with a buffer in CCM RAM.
- Observe failure (Bus Fault or Transfer Error).
- Fix it by moving buffer to SRAM.

### 2. Step-by-Step Guide

#### Phase A: The Bug
```c
__attribute__((section(".ccmram")))
uint8_t tx_buf[100];

void Test_DMA(void) {
    HAL_UART_Transmit_DMA(&huart2, tx_buf, 100);
}
```

#### Phase B: Execution
1.  Run code.
2.  **Observation:** UART sends nothing or garbage. DMA Controller cannot access 0x10000000 bus matrix.

#### Phase C: The Fix
Change section to `.data` (SRAM1) or remove attribute.
```c
uint8_t tx_buf[100]; // Defaults to SRAM1
```

### 3. Verification
Check the datasheet "Bus Matrix". See that DMA1/DMA2 masters do not have a line connecting to CCM RAM slave.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Memory Protection (MPU)
- **Goal:** Make NULL pointer dereference HardFault.
- **Task:**
    1.  The first 4KB of Flash (0x08000000) is aliased to 0x00000000. So `*NULL` reads the stack pointer!
    2.  Use MPU to mark Region 0 (0x00000000 - 0x000000FF) as "No Access".
    3.  Now `int x = *NULL;` triggers MemManage Fault.

### Lab 3: Shared Memory (Dual Core)
- **Goal:** Prepare for STM32H7 (Dual Core).
- **Task:**
    1.  Define a `SHARED` region in `MEMORY`.
    2.  `SHARED (rw) : ORIGIN = 0x30040000, LENGTH = 32K`
    3.  Place a struct there. `__attribute__((section(".shared")))`.
    4.  (On single core, just proves placement works).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Section .ccmram overlaps .text"
*   **Cause:** Linker script address calculation error.
*   **Solution:** Ensure `ORIGIN` and `LENGTH` are correct and do not overlap.

#### 2. HardFault on Function Call
*   **Cause:** Calling a function in CCM RAM *before* the copy loop ran.
*   **Solution:** Ensure `SystemInit` (or wherever copy happens) runs before `main`.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **NOLOAD:** For large buffers (e.g., Framebuffer in SDRAM) that don't need initialization, use `(NOLOAD)` in linker script. This prevents the output binary from becoming 32MB large (filled with zeros).
    ```ld
    .sdram (NOLOAD) : { *(.sdram) } > SDRAM
    ```

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `VMA` and `LMA`?
    *   **A:** VMA (Virtual Memory Address) is where code runs (RAM). LMA (Load Memory Address) is where it is stored (Flash). For `.text`, VMA=LMA. For `.data`, VMA!=LMA.
2.  **Q:** Why use CCM RAM?
    *   **A:** It is directly connected to the D-Bus of the Core. 0 Wait States. Faster than SRAM1 which goes through the Bus Matrix (contention with DMA).

### Challenge Task
> **Task:** "Function Overlay". Define two functions `TaskA` and `TaskB`. Both are compiled to run at address `0x20001000`. Store both in Flash. At runtime, copy `TaskA` to RAM, run it. Then copy `TaskB` to the *same* RAM address, run it. (Saves RAM).

---

## 📚 Further Reading & References
- [GNU Linker (LD) Command Language](https://ftp.gnu.org/old-gnu/Manuals/ld-2.9.1/html_node/ld_3.html)
- [STM32F4 CCM RAM Application Note](https://www.st.com/resource/en/application_note/dm00083249.pdf)

---
