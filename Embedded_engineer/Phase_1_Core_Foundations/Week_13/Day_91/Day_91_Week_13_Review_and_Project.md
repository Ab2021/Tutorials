# Day 91: Week 13 Review and Project
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
1.  **Synthesize** Week 13 concepts (JTAG, GDB, Faults, Logging, Testing) into a robust system.
2.  **Implement** a "Flight Data Recorder" (Black Box) that survives reboots.
3.  **Debug** a complex, intentionally buggy application using the tools mastered this week.
4.  **Create** a comprehensive Test Suite (Unit + System) for the project.
5.  **Conduct** a Code Review focusing on testability and error handling.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code, GDB, OpenOCD, Python.
*   **Prior Knowledge:**
    *   Days 85-90.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "Black Box" Concept
In avionics, a Flight Data Recorder (FDR) captures critical system state. If the plane crashes, the data survives.
*   **Requirements:**
    1.  **Circular Logging:** Always keep the last N events.
    2.  **Persistence:** Data must survive Reset (Backup RAM or Flash).
    3.  **Crash Capture:** HardFault handler must write final state to the Box.
    4.  **Retrieval:** CLI command to dump the Box after reboot.

### 🔹 Part 2: Backup SRAM (BKPSRAM)
STM32F4 has 4KB of Backup SRAM powered by $V_{BAT}$.
*   **Enable:** `PWR->CR |= PWR_CR_DBP` (Disable Backup Protection).
*   **Clock:** `RCC->AHB1ENR |= RCC_AHB1ENR_BKPSRAMEN`.
*   **Regulator:** `PWR->CSR |= PWR_CSR_BRE` (Backup Regulator Enable).

---

## 💻 Implementation: Flight Data Recorder

> **Instruction:** Implement the FDR module using Backup SRAM.

### 👨‍💻 Code Implementation

#### Step 1: Memory Map
```c
#define BKPSRAM_BASE 0x40024000
#define FDR_MAGIC    0xDEADBEEF

typedef struct {
    uint32_t magic;
    uint32_t boot_count;
    uint32_t crash_count;
    uint32_t last_fault_pc;
    uint32_t last_fault_lr;
    uint32_t last_fault_cfsr;
    char     log_buffer[1024]; // Circular
    uint32_t log_head;
} FDR_Data_t;

FDR_Data_t *fdr = (FDR_Data_t*)BKPSRAM_BASE;
```

#### Step 2: Initialization
```c
void FDR_Init(void) {
    // Enable Clocks & Power
    RCC->APB1ENR |= RCC_APB1ENR_PWREN;
    PWR->CR |= PWR_CR_DBP; // Access to Backup Domain
    RCC->AHB1ENR |= RCC_AHB1ENR_BKPSRAMEN;
    PWR->CSR |= PWR_CSR_BRE; // Enable Backup Regulator
    while(!(PWR->CSR & PWR_CSR_BRR)); // Wait for Ready
    
    // Check Magic
    if (fdr->magic != FDR_MAGIC) {
        printf("FDR: First Boot. Initializing...\n");
        memset((void*)fdr, 0, sizeof(FDR_Data_t));
        fdr->magic = FDR_MAGIC;
    } else {
        printf("FDR: Valid Data Found. Boot Count: %lu\n", fdr->boot_count);
        if (fdr->crash_count > 0) {
            printf("!!! PREVIOUS CRASH DETECTED !!!\n");
            printf("PC: %08lX, LR: %08lX, CFSR: %08lX\n", 
                   fdr->last_fault_pc, fdr->last_fault_lr, fdr->last_fault_cfsr);
        }
    }
    fdr->boot_count++;
}
```

#### Step 3: Logging to FDR
```c
void FDR_Log(const char *msg) {
    int len = strlen(msg);
    for(int i=0; i<len; i++) {
        fdr->log_buffer[fdr->log_head] = msg[i];
        fdr->log_head = (fdr->log_head + 1) % 1024;
    }
}
```

#### Step 4: HardFault Integration
In `PrvHardFaultHandler` (Day 88):
```c
void PrvHardFaultHandler(uint32_t *pStack) {
    // ... Capture Registers ...
    
    // Save to FDR
    fdr->last_fault_pc = pc;
    fdr->last_fault_lr = lr;
    fdr->last_fault_cfsr = cfsr;
    fdr->crash_count++;
    
    NVIC_SystemReset();
}
```

---

## 🔬 Lab Exercise: Lab 91.1 - The Crash Test

### 1. Lab Objectives
- Trigger a crash.
- Verify FDR captures it.
- Dump log via CLI.

### 2. Step-by-Step Guide

#### Phase A: CLI Commands
Add `fdr_dump` and `crash` commands to CLI.
*   `crash`: Dereference NULL.
*   `fdr_dump`: Print `fdr->log_buffer` and crash stats.

#### Phase B: Execution
1.  Boot. `FDR: First Boot`.
2.  CLI: `crash`.
3.  Board Resets.
4.  Boot. `!!! PREVIOUS CRASH DETECTED !!!`.
5.  CLI: `fdr_dump`.
6.  **Observation:** See the PC address. Use `addr2line -e build/main.elf <PC>` to find the line number.

### 3. Verification
If Backup SRAM resets on reboot, ensure `PWR_CSR_BRE` is set and `VBAT` is connected (or VDD is stable).

---

## 🧪 Additional / Advanced Labs

### Lab 2: Unit Test the FDR
- **Goal:** Verify Circular Buffer logic.
- **Task:**
    1.  Write `test_fdr.c` (Host).
    2.  Mock the memory address `BKPSRAM_BASE` to a local array.
    3.  Test wrapping behavior of `FDR_Log`.

### Lab 3: Post-Mortem Analysis Script
- **Goal:** Automation.
- **Task:**
    1.  Python script `analyze_crash.py`.
    2.  Connects to UART.
    3.  Sends `fdr_dump`.
    4.  Parses PC/LR.
    5.  Runs `addr2line` automatically and prints the source code line.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. BKPSRAM Access Fault
*   **Cause:** Forgot `PWR_CR_DBP` or Clock Enable.
*   **Result:** BusFault when writing to `0x40024000`.

#### 2. Data Corruption
*   **Cause:** Overwriting FDR from stack overflow?
*   **Solution:** Check linker map. BKPSRAM is separate from Main RAM, so stack shouldn't reach it unless you mapped it there.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Checksum:** Add a CRC32 to the FDR structure. Recalculate on boot to ensure data validity (e.g., battery didn't die).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why use Backup SRAM instead of Flash?
    *   **A:** Flash has limited write cycles (10k-100k) and is slow to write (ms). SRAM is infinite cycles and fast (ns), perfect for continuous logging.
2.  **Q:** How do we debug the HardFault Handler itself?
    *   **A:** Set a breakpoint at the start of the handler in GDB. Trigger a fault. Step through assembly.

### Challenge Task
> **Task:** "Remote Crash Report". On boot, if a crash is detected, use the Network Stack (Day 82) to MQTT Publish the crash report to the Cloud before clearing the flag.

---

## 📚 Further Reading & References
- [STM32F4 Power Controller (PWR) Reference Manual](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)

---
