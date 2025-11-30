# Day 84: Week 12 Review & Phase 1 Integration
## Phase 1: Core Embedded Engineering Foundations | Week 12: Capstone Project Phase 1

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
1.  **Integrate** all modules (HAL, Audio, Net, App) into a final, cohesive firmware binary.
2.  **Perform** system-level validation including stress testing and long-duration stability tests.
3.  **Optimize** the final build for size (`-Os`) and performance.
4.  **Generate** a release package (Binary + Map file + Checksum).
5.  **Reflect** on the entire Phase 1 journey and prepare for the transition to Embedded Linux (Phase 2).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board (Fully assembled Capstone Setup).
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   Wireshark, Logic Analyzer, Multimeter.
*   **Prior Knowledge:**
    *   Days 1-83 (The entire Phase 1).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: System Integration Challenges
Merging 4 weeks of work isn't easy.
*   **Resource Conflicts:** Audio uses DMA1_Stream7. Net uses DMA2_StreamX. Ensure no overlap.
*   **Priority Inversion:** Does the Network Task block the Audio Task? (Check priorities).
*   **Stack Usage:** Total RAM usage = Sum(Task Stacks) + Heap + Globals.
    *   Audio: 1KB
    *   Net: 4KB
    *   App: 2KB
    *   Heap: 30KB (LwIP + mbedTLS)
    *   Globals: 10KB (Buffers)
    *   Total: ~50KB. (STM32F4 has 128KB. Safe).

### 🔹 Part 2: The Release Build
Debug builds (`-O0 -g`) are slow and large. Release builds (`-Os` or `-O3`) are fast but hard to debug.
*   **LTO (Link Time Optimization):** Removes unused code across files.
*   **Strip:** Removes debug symbols from the binary (reduces size).

---

## 💻 Implementation: Final Polish

> **Instruction:** Finalize `main.c` and `FreeRTOSConfig.h`.

### 👨‍💻 Code Implementation

#### Step 1: FreeRTOSConfig.h Tuning
```c
#define configUSE_PREEMPTION                    1
#define configUSE_IDLE_HOOK                     1 // For Power Save
#define configUSE_TICK_HOOK                     0
#define configCPU_CLOCK_HZ                      (SystemCoreClock)
#define configTICK_RATE_HZ                      (1000)
#define configMAX_PRIORITIES                    (7)
#define configMINIMAL_STACK_SIZE                ((unsigned short)128)
#define configTOTAL_HEAP_SIZE                   ((size_t)(40 * 1024)) // Heap_4
#define configMAX_TASK_NAME_LEN                 (16)
#define configUSE_TRACE_FACILITY                1 // For SystemView
#define configUSE_16_BIT_TICKS                  0
#define configIDLE_SHOULD_YIELD                 1

// Software Timer
#define configUSE_TIMERS                        1
#define configTIMER_TASK_PRIORITY               (2)
#define configTIMER_QUEUE_LENGTH                10
#define configTIMER_TASK_STACK_DEPTH            (configMINIMAL_STACK_SIZE * 2)

// Interrupts
#define configLIBRARY_LOWEST_INTERRUPT_PRIORITY 0xf
#define configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY 5
```

#### Step 2: HardFault Handler (Production)
In `stm32f4xx_it.c`:
```c
void HardFault_Handler(void) {
    // 1. Capture Registers (R0-R3, R12, LR, PC, xPSR)
    // 2. Save to Backup Registers or Flash (Error Log)
    // 3. Auto Reset
    
    __asm volatile (
        " tst lr, #4                                                \n"
        " ite eq                                                    \n"
        " mrseq r0, msp                                             \n"
        " mrsne r0, psp                                             \n"
        " ldr r1, [r0, #24]                                         \n"
        " b PrvGetRegistersFromStack                                \n"
    );
}

void PrvGetRegistersFromStack(uint32_t *pulFaultStackAddress) {
    volatile uint32_t r0 = pulFaultStackAddress[0];
    volatile uint32_t r1 = pulFaultStackAddress[1];
    volatile uint32_t r2 = pulFaultStackAddress[2];
    volatile uint32_t r3 = pulFaultStackAddress[3];
    volatile uint32_t r12 = pulFaultStackAddress[4];
    volatile uint32_t lr = pulFaultStackAddress[5];
    volatile uint32_t pc = pulFaultStackAddress[6];
    volatile uint32_t psr = pulFaultStackAddress[7];
    
    printf("HardFault! PC: %08lx LR: %08lx\n", pc, lr);
    
    NVIC_SystemReset();
}
```

---

## 🔬 Lab Exercise: Lab 84.1 - The Stress Test

### 1. Lab Objectives
- Run the system for 1 hour.
- Flood it with Pings.
- Clap continuously.
- Reconnect Ethernet repeatedly.

### 2. Step-by-Step Guide

#### Phase A: Network Stress
1.  PC: `ping -t -l 1000 192.168.1.X`.
2.  While pinging, toggle User Button (Mute/Unmute).
3.  **Observation:** Ping should not drop. Audio should not glitch.

#### Phase B: Audio Stress
1.  Play a 1kHz tone.
2.  Disconnect Ethernet.
3.  **Observation:** Tone should remain stable. Network timeout logic should not block Audio Task.

#### Phase C: Memory Leak Check
1.  Monitor `xPortGetFreeHeapSize()` via CLI every 10 mins.
2.  **Observation:** Should stabilize. If it keeps dropping, you have a leak (check `cJSON_Delete` or `pbuf_free`).

### 3. Verification
If system freezes, attach debugger. Check if stuck in `HardFault_Handler` or `Error_Handler` (HAL).

---

## 🧪 Additional / Advanced Labs

### Lab 2: Power Profiling
- **Goal:** Battery Life.
- **Task:**
    1.  Measure current in Idle (no audio, net connected).
    2.  Implement `vApplicationIdleHook` -> `__WFI()`.
    3.  Measure savings.

### Lab 3: Boot Time Optimization
- **Goal:** Fast Startup.
- **Task:**
    1.  Measure time from Reset to "Ready".
    2.  Optimize: Remove `HAL_Delay` in Init. Use DMA for initial prints.
    3.  Target: < 2 seconds (DHCP is the bottleneck).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Optimization Bugs (-O3)
*   **Cause:** Compiler reordering code or removing "useless" loops.
*   **Solution:** Use `volatile` for hardware registers and shared variables.

#### 2. Race Conditions (Rare)
*   **Cause:** Only happens after hours of running.
*   **Solution:** Code Review. Ensure all shared data is protected by Queues or Mutexes.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Static Analysis:** Run `cppcheck` or `scan-build` on the entire project. Fix all warnings.
- **Documentation:** Generate Doxygen docs for your BSP and App APIs.

---

## 🧠 Assessment & Review

### Phase 1 Summary
Congratulations! You have completed the **Core Embedded Engineering Foundations**.
*   **Weeks 1-4:** Bare Metal (Registers, C, Assembly).
*   **Weeks 5-8:** Peripherals (UART, SPI, I2C, Timers).
*   **Weeks 9-10:** RTOS & IoT (FreeRTOS, LwIP, MQTT).
*   **Weeks 11-12:** DSP & Capstone (Audio, Architecture).

### Transition to Phase 2
Phase 2 focuses on **Embedded Linux**.
*   **Hardware:** Raspberry Pi (or QEMU).
*   **Topics:** Kernel Modules, Device Drivers, Yocto, Userspace programming.
*   **Mindset Shift:** From "I control every bit" to "I manage complex subsystems".

### Challenge Task
> **Task:** Archive your Phase 1 project. Create a `README.md` with photos, architecture diagrams, and build instructions. Push to GitHub. This is your portfolio piece!

---

## 📚 Further Reading & References
- [The Art of Unix Programming (Philosophy applies to Embedded)](http://www.catb.org/~esr/writings/taoup/)
- [Embedded Systems Architecture (Book)](https://www.amazon.com/Embedded-Systems-Architecture-Comprehensive-Guide/dp/1259076926)

---
