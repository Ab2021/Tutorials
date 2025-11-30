# Day 88: Fault Analysis & HardFault Handling
## Phase 1: Core Embedded Engineering Foundations | Week 13: Debugging and Testing

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
1.  **Explain** the ARM Cortex-M Fault Model (HardFault, MemManage, BusFault, UsageFault).
2.  **Implement** a robust `HardFault_Handler` in Assembly to capture the stack frame.
3.  **Decode** the stacked registers (PC, LR) to pinpoint the exact line of code causing the crash.
4.  **Analyze** the Configurable Fault Status Register (CFSR) to determine the root cause (e.g., Divide by Zero, Unaligned Access).
5.  **Debug** a "Imprecise Bus Fault" caused by write buffering.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   GDB
*   **Prior Knowledge:**
    *   Day 12 (Exception Handling)
    *   Day 9 (Registers)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Fault Hierarchy
*   **HardFault:** The catch-all. If another fault handler is disabled or fails, it escalates to HardFault.
*   **MemManage Fault:** MPU violation (e.g., writing to Read-Only memory, executing from NX memory).
*   **BusFault:** Hardware error on the bus (e.g., reading invalid address, peripheral not clocked).
*   **UsageFault:** Instruction execution error (e.g., Undefined Instruction, Divide by Zero, Unaligned Access).

> **Note:** By default, MemManage, BusFault, and UsageFault are **disabled**. They all trigger HardFault. You must enable them in SHCSR (System Handler Control and State Register) to get specific handlers.

### 🔹 Part 2: The Stack Frame
When an exception occurs, the hardware automatically pushes 8 registers onto the stack:
1.  **R0-R3:** Arguments / Scratch.
2.  **R12:** Scratch.
3.  **LR (Link Register):** Return address of the function that was interrupted.
4.  **PC (Program Counter):** The address of the instruction that *caused* the fault (usually).
5.  **xPSR:** Status Register.

We need to read these from the stack pointer (MSP or PSP) inside the handler.

### 🔹 Part 3: Fault Status Registers (SCB->CFSR)
*   **UFSR (Usage):** DIVBYZERO, UNALIGNED, NOCP (No Coprocessor).
*   **BFSR (Bus):** IBUSERR (Instruction Fetch), PRECISERR (Data Read/Write), IMPRECISERR (Buffered Write).
*   **MMFSR (Mem):** IACCVIOL, DACCVIOL.

---

## 💻 Implementation: The Ultimate Handler

> **Instruction:** Write an assembly wrapper to pass the stack pointer to a C function.

### 👨‍💻 Code Implementation

#### Step 1: Assembly Wrapper (`stm32f4xx_it.c`)
```c
__attribute__((naked)) void HardFault_Handler(void) {
    __asm volatile (
        " tst lr, #4                                                \n" // Check EXC_RETURN bit 2
        " ite eq                                                    \n" // If 0 (MSP)
        " mrseq r0, msp                                             \n" // R0 = MSP
        " mrsne r0, psp                                             \n" // Else R0 = PSP
        " ldr r1, [r0, #24]                                         \n" // Load PC from stack (for debugging)
        " b PrvHardFaultHandler                                     \n" // Jump to C function
    );
}
```

#### Step 2: C Handler
```c
void PrvHardFaultHandler(uint32_t *pStack) {
    // Stack Frame Layout:
    // [0] R0, [1] R1, [2] R2, [3] R3, [4] R12, [5] LR, [6] PC, [7] xPSR
    
    volatile uint32_t r0  = pStack[0];
    volatile uint32_t r1  = pStack[1];
    volatile uint32_t r2  = pStack[2];
    volatile uint32_t r3  = pStack[3];
    volatile uint32_t r12 = pStack[4];
    volatile uint32_t lr  = pStack[5];
    volatile uint32_t pc  = pStack[6];
    volatile uint32_t psr = pStack[7];
    
    volatile uint32_t cfsr = SCB->CFSR;
    volatile uint32_t hfsr = SCB->HFSR;
    volatile uint32_t mmfar = SCB->MMFAR;
    volatile uint32_t bfar = SCB->BFAR;
    
    printf("\n=== HARD FAULT ===\n");
    printf("PC : 0x%08lX\n", pc);
    printf("LR : 0x%08lX\n", lr);
    printf("CFSR : 0x%08lX\n", cfsr);
    
    // Decode CFSR
    if (cfsr & (1 << 25)) printf(" [DivByZero]\n");
    if (cfsr & (1 << 24)) printf(" [Unaligned]\n");
    if (cfsr & (1 << 17)) printf(" [Invalid State]\n"); // Thumb bit missing
    if (cfsr & (1 << 8))  printf(" [BusFault]\n");
    
    if (cfsr & (1 << 15)) {
        printf(" [BusFault Address Valid] Addr: 0x%08lX\n", bfar);
    }
    
    while(1); // Trap
}
```

---

## 🔬 Lab Exercise: Lab 88.1 - Fault Injection

### 1. Lab Objectives
- Trigger different faults.
- Verify the handler reports the correct cause.

### 2. Step-by-Step Guide

#### Phase A: Divide by Zero
1.  Enable DivByZero Trap: `SCB->CCR |= SCB_CCR_DIV_0_TRP_Msk;`
2.  Code: `volatile int a = 10, b = 0; int c = a / b;`
3.  **Observation:** Handler prints `[DivByZero]`. PC points to the divide instruction.

#### Phase B: Invalid Address (Bus Fault)
1.  Code: `volatile int *p = (int*)0x80000000; *p = 0;` (Outside RAM/Flash).
2.  **Observation:** Handler prints `[BusFault]`. `BFAR` might show `0x80000000`.

#### Phase C: Unaligned Access
1.  Enable Unaligned Trap: `SCB->CCR |= SCB_CCR_UNALIGN_TRP_Msk;`
2.  Code: `volatile int *p = (int*)0x20000001; *p = 0xDEADBEEF;` (Odd address).
3.  **Observation:** Handler prints `[Unaligned]`.

### 3. Verification
If PC points to `HardFault_Handler` itself, you have a "Double Fault" (Stack overflow usually). Check `SP` value.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Imprecise Bus Fault
- **Goal:** Understand Write Buffering.
- **Task:**
    1.  Write to invalid address.
    2.  Execute a few `NOP`s.
    3.  **Observation:** Fault triggers *lines later*. `IMPRECISERR` bit set.
    4.  **Fix:** Use `__DSB()` (Data Synchronization Barrier) after critical writes to force the fault immediately.

### Lab 3: Post-Mortem Log
- **Goal:** Save crash info.
- **Task:**
    1.  In Handler, write PC, LR, CFSR to Backup Registers (RTC_BKPxR).
    2.  Reset (`NVIC_SystemReset`).
    3.  On Boot, check Backup Regs. If non-zero, print "Crash Detected at PC: ..."

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Optimization Hiding Faults
*   **Cause:** `-O2` might optimize away `a / b` if result unused.
*   **Solution:** Use `volatile` or print the result.

#### 2. PC points to weird location
*   **Cause:** Stack corruption. The PC on stack is garbage.
*   **Solution:** Check `LR` (Link Register) to see who called the function.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Production:** Never `while(1)` in HardFault. Log and Reset. A stuck device is worse than a rebooting one.
- **Watchdog:** Ensure WWDG/IWDG can reset the system even if HardFault handler hangs.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we check `LR` bit 2?
    *   **A:** To determine if the stack used was MSP (Main Stack Pointer) or PSP (Process Stack Pointer). RTOS tasks use PSP. ISRs use MSP.
2.  **Q:** What is `BFAR`?
    *   **A:** Bus Fault Address Register. It holds the address that caused the bus fault (if valid bit is set).

### Challenge Task
> **Task:** Implement a "Stack Dump". In the handler, print the raw stack contents (previous 32 words) to help reconstruct the call stack manually.

---

## 📚 Further Reading & References
- [Joseph Yiu: The Definitive Guide to ARM Cortex-M3/M4](https://www.amazon.com/Definitive-Guide-ARM-Cortex-M3-Cortex-M4-Processors/dp/0124080820)
- [Memfault: Debugging HardFaults on Cortex-M](https://interrupt.memfault.com/blog/cortex-m-fault-debug)

---
