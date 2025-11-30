# Day 109: Boot Time Optimization
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
1.  **Profile** boot time using GPIO toggles and Logic Analyzers.
2.  **Optimize** the `Reset_Handler` and C Runtime (CRT) initialization.
3.  **Implement** "Lazy Initialization" to defer non-critical peripheral setup.
4.  **Analyze** the impact of `memcpy` speed on copying `.data` from Flash to RAM.
5.  **Design** a fast-boot architecture for time-critical applications (e.g., Automotive Rear Camera).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Logic Analyzer or Oscilloscope.
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 10 (Memory Architecture)
    *   Day 107 (Compiler Opt)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Anatomy of a Boot
1.  **Hardware Reset:** Power rail stabilizes. NRST released.
2.  **Hardware Init:** Fetch SP and PC from Vector Table.
3.  **Reset_Handler:**
    *   Copy `.data` (Flash -> RAM).
    *   Zero `.bss` (RAM).
    *   Call `SystemInit` (Clock setup).
    *   Call `__libc_init_array` (C++ Constructors).
    *   Call `main()`.
4.  **Application Init:** `HAL_Init`, `MX_GPIO_Init`, etc.

### 🔹 Part 2: The Bottlenecks
*   **Clock Setup:** PLL lock time (~200us).
*   **Data Copy:** Copying 10KB of data at 16MHz (HSI) takes time.
*   **C Library:** `printf` initialization or heap setup can be slow.
*   **Peripheral Init:** `HAL_Delay` inside `SD_Init` or `LCD_Init`.

### 🔹 Part 3: Strategies
*   **Run from Flash (XIP):** Don't copy code to RAM unless necessary.
*   **Lazy Init:** Init UART only when user presses a button, not at boot.
*   **Asm Copy:** Use optimized assembly (`LDM`/`STM`) for `.data` copy.

```mermaid
graph LR
    Reset[Reset] -->|Hardware| Vector[Vector Fetch]
    Vector -->|Asm| CopyData[Copy .data / Zero .bss]
    CopyData -->|C| SystemInit[Clock Setup]
    SystemInit -->|C| Main[main()]
    Main -->|App| Critical[Critical Task]
    Critical -->|Lazy| Rest[Init Rest of System]
```

---

## 💻 Implementation: Profiling Boot Time

> **Instruction:** Measure time from Reset to `main()`.

### 👨‍💻 Code Implementation

#### Step 1: GPIO Trigger
In `Reset_Handler` (startup_stm32.s), toggle a pin immediately.
*   *Problem:* GPIO clock not enabled yet.
*   *Solution:* Write directly to RCC and GPIO registers in Assembly.

```assembly
Reset_Handler:
  /* 1. Enable GPIOD Clock (Bit 3 in AHB1ENR) */
  LDR R0, =0x40023830  /* RCC_AHB1ENR */
  LDR R1, [R0]
  ORR R1, R1, #0x08
  STR R1, [R0]

  /* 2. Set PD12 as Output (Bit 24 in MODER) */
  LDR R0, =0x40020C00  /* GPIOD_MODER */
  LDR R1, [R0]
  ORR R1, R1, #0x01000000
  STR R1, [R0]

  /* 3. Set PD12 High */
  LDR R0, =0x40020C14  /* GPIOD_ODR */
  LDR R1, [R0]
  ORR R1, R1, #0x1000
  STR R1, [R0]
  
  /* Continue with LoopCopyDataInit... */
```

#### Step 2: Measure
1.  Connect Logic Analyzer to PD12 and NRST.
2.  Trigger on NRST Rising Edge.
3.  Measure time to PD12 Rising Edge.
4.  **Result:** This is Hardware Latency + Assembly Setup.

#### Step 3: Measure Main Entry
In `main()`:
```c
int main(void) {
    HAL_GPIO_WritePin(GPIOD, GPIO_PIN_13, 1); // Toggle another pin
    // ...
}
```
Time between PD12 High and PD13 High = C Runtime Initialization duration.

---

## 💻 Implementation: Optimized Copy Loop

> **Instruction:** Replace byte-by-byte copy with word copy.

### 👨‍💻 Code Implementation

#### Step 1: The Slow Way (Default)
```assembly
LoopCopyDataInit:
  LDR R3, =_sidata
  LDR R3, [R3, R1]
  STR R3, [R0, R1]
  ADDS R1, R1, #1  /* Byte increment */
```

#### Step 2: The Fast Way (C Implementation)
We can move the copy logic to C (after stack setup) and use `uint32_t`.
```c
void SystemInit(void) {
    // 1. Enable FPU (if used)
    #if (__FPU_PRESENT == 1) && (__FPU_USED == 1)
      SCB->CPACR |= ((3UL << 10*2)|(3UL << 11*2));
    #endif

    // 2. Optimized Data Copy
    extern uint32_t _sidata, _sdata, _edata;
    uint32_t *src = &_sidata;
    uint32_t *dst = &_sdata;
    
    while(dst < &_edata) {
        *dst++ = *src++; // Word copy (4 bytes)
    }
    
    // 3. Zero BSS
    extern uint32_t _sbss, _ebss;
    dst = &_sbss;
    while(dst < &_ebss) {
        *dst++ = 0;
    }
}
```
**Note:** Compilers usually generate efficient copy loops (`memcpy`), but ensuring word alignment helps.

---

## 🔬 Lab Exercise: Lab 109.1 - Lazy Initialization

### 1. Lab Objectives
- Boot fast to toggle LED.
- Init USB later.

### 2. Step-by-Step Guide

#### Phase A: The Slow Boot
```c
int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_USB_HOST_Init(); // Takes 500ms?
    MX_GPIO_Init();
    
    // Critical Task
    HAL_GPIO_WritePin(GPIOD, GPIO_PIN_12, 1);
    
    while(1);
}
```

#### Phase B: The Fast Boot
```c
int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    
    // Critical Task (Immediate)
    HAL_GPIO_WritePin(GPIOD, GPIO_PIN_12, 1);
    
    // Lazy Init
    HAL_Delay(100); // Simulate other work
    MX_USB_HOST_Init(); // Do this later
    
    while(1);
}
```

### 3. Verification
Measure time to LED ON.
*   **Slow:** 500ms + overhead.
*   **Fast:** < 1ms + overhead.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Bootloader Bypass
- **Goal:** Skip bootloader check.
- **Task:**
    1.  If your system has a bootloader (Day 78), it usually waits 1-2s for UART input.
    2.  Check a "Magic Button" on reset.
    3.  If NOT pressed, jump immediately to App.
    4.  If pressed, wait for UART.

### Lab 3: No-Init RAM
- **Goal:** Preserve data across reset without copy.
- **Task:**
    1.  Define a section `.noinit` in linker script (NOLOAD).
    2.  `__attribute__((section(".noinit"))) uint32_t crash_log[100];`
    3.  Remove `.noinit` from startup copy/zero loop.
    4.  Boot time saved: No need to zero out this RAM.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. HardFault during Copy
*   **Cause:** Unaligned access. If `_sdata` is not 4-byte aligned.
*   **Solution:** Use `ALIGN(4)` in Linker Script.

#### 2. Global Variables Garbage
*   **Cause:** Forgot to copy `.data` or zero `.bss`.
*   **Result:** `int count = 0;` starts at random value.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Clock Speed:** Switch to High Speed Clock (HSE/PLL) *before* copying data if the data section is huge (MBs). Copying at 168MHz is 10x faster than at 16MHz.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `.bss`?
    *   **A:** Block Started by Symbol. Uninitialized global variables. Must be zeroed at boot.
2.  **Q:** Why does `printf` slow down boot?
    *   **A:** It might initialize UART, allocate buffers, or wait for PC connection.

### Challenge Task
> **Task:** "Instant On". Achieve < 10ms from Reset to SPI Transmission. Requires stripping HAL, using Registers, and optimizing the startup assembly.

---

## 📚 Further Reading & References
- [Memfault: Zero to Main()](https://interrupt.memfault.com/blog/zero-to-main-1)

---
