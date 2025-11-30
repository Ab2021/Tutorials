# Day 11: Interrupts and NVIC
## Phase 1: Core Embedded Engineering Foundations | Week 2: ARM Cortex-M Architecture

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
1.  **Explain** the Nested Vectored Interrupt Controller (NVIC) architecture in Cortex-M.
2.  **Configure** interrupt priorities and understand Preemption vs. Sub-priority.
3.  **Implement** a robust Interrupt Service Routine (ISR) in C.
4.  **Calculate** interrupt latency and understand the factors affecting it.
5.  **Debug** interrupt-related issues (race conditions, stuck interrupts).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 9 (Exception Model)
    *   Day 5 (Memory Map)
*   **Datasheets:**
    *   [STM32F407 Reference Manual (NVIC Section)](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)
    *   [Cortex-M4 Generic User Guide (NVIC Section)](https://developer.arm.com/documentation/dui0553/latest/)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Interrupt Fundamentals

#### 1.1 Polling vs. Interrupts
*   **Polling:** The CPU constantly checks a flag (`while(1) { if(flag) ... }`). Wastes CPU cycles and power. High latency if the loop is long.
*   **Interrupts:** The hardware signals the CPU (`IRQ`). The CPU pauses the main code, jumps to a specific function (ISR), handles the event, and resumes. Efficient and responsive.

#### 1.2 The NVIC (Nested Vectored Interrupt Controller)
The NVIC is a standard peripheral inside the Cortex-M core (not vendor-specific).
*   **Nested:** A higher priority interrupt can interrupt a lower priority one.
*   **Vectored:** The address of the ISR is fetched directly from a table (Vector Table) in memory. No software dispatching needed.
*   **Low Latency:** Hardware stacking and tail-chaining reduce overhead.

### 🔹 Part 2: Priorities and Preemption

#### 2.1 Priority Levels
Cortex-M4 supports up to 256 priority levels (0-255). Lower number = Higher Priority.
*   **STM32 Implementation:** STM32 only implements 4 bits of priority (16 levels, 0-15).

#### 2.2 Preemption vs. Sub-Priority
The priority bits can be split into two groups using the `AIRCR` register (Priority Grouping).
*   **Preemption Priority:** Determines if an interrupt can interrupt another active interrupt.
*   **Sub-Priority:** Used only to break ties when two interrupts happen *simultaneously*. Does NOT allow preemption.

**Example (Group 4 - 4 bits Preemption, 0 bits Sub):**
*   IRQ A (Prio 0) arrives.
*   IRQ B (Prio 1) arrives. IRQ A continues (0 > 1).
*   IRQ C (Prio 0) arrives while A is running. C waits (Equal priority doesn't preempt).

### 🔹 Part 3: The Vector Table

The Vector Table is an array of 32-bit addresses starting at `0x00000000` (or `0x08000000`).
*   Offset 0x00: Initial SP
*   Offset 0x04: Reset Handler
*   Offset 0x08: NMI Handler
*   Offset 0x0C: HardFault Handler
*   ...
*   Offset 0x40+: External Interrupts (WWDG, PVD, TAMP, etc.)

```mermaid
graph TD
    Event[Hardware Event (Button Press)] --> NVIC[NVIC Controller]
    NVIC -->|Check Priority| CPU[CPU Core]
    CPU -->|Push Context| Stack[Stack (RAM)]
    CPU -->|Fetch Vector| Vector[Vector Table (Flash)]
    Vector -->|Jump| ISR[ISR Code]
    ISR -->|Return| CPU
    CPU -->|Pop Context| Stack
```

---

## 💻 Implementation: External Interrupt (Button)

> **Instruction:** We will configure the User Button (PA0) to trigger an interrupt (EXTI0) on the rising edge.

### 🛠️ Hardware/System Configuration
*   **Button:** PA0 -> EXTI Line 0.
*   **LED:** PD12.

### 👨‍💻 Code Implementation

#### Step 1: Vector Table Setup (`startup.c`)
Ensure the vector table has an entry for `EXTI0_IRQHandler`.
```c
void EXTI0_IRQHandler(void) __attribute__((weak, alias("Default_Handler")));

// Inside vector_table array:
// ...
(uint32_t *)EXTI0_IRQHandler, // IRQ 6
// ...
```

#### Step 2: NVIC & EXTI Configuration (`main.c`)

```c
#include "stm32f4xx.h"

void EXTI0_Init(void) {
    // 1. Enable Clocks (GPIOA and SYSCFG)
    RCC->AHB1ENR |= (1 << 0); // GPIOA
    RCC->APB2ENR |= (1 << 14); // SYSCFG

    // 2. Configure PA0 as Input
    GPIOA->MODER &= ~(0x3 << 0);

    // 3. Connect EXTI0 to PA0
    // SYSCFG_EXTICR1: Bits 0-3 select the port for EXTI0. 0000 = PA.
    SYSCFG->EXTICR[0] &= ~(0xF << 0);

    // 4. Configure EXTI Line 0
    EXTI->IMR |= (1 << 0);  // Unmask Interrupt
    EXTI->RTSR |= (1 << 0); // Rising Edge Trigger

    // 5. Configure NVIC
    // EXTI0_IRQn is 6.
    NVIC_SetPriority(EXTI0_IRQn, 1); // Priority 1
    NVIC_EnableIRQ(EXTI0_IRQn);      // Enable in NVIC
}

// The ISR
void EXTI0_IRQHandler(void) {
    // 1. Check if the interrupt is pending
    if (EXTI->PR & (1 << 0)) {
        // 2. Clear the pending bit (Write 1 to clear)
        EXTI->PR |= (1 << 0);

        // 3. Handle the event (Toggle LED)
        GPIOD->ODR ^= (1 << 12);
    }
}

int main(void) {
    // Init LED (PD12)
    RCC->AHB1ENR |= (1 << 3);
    GPIOD->MODER |= (1 << 24);

    EXTI0_Init();

    while(1) {
        // Main loop does nothing!
        // CPU could sleep here.
    }
}
```

---

## 🔬 Lab Exercise: Lab 11.1 - Priority Preemption

### 1. Lab Objectives
- Verify that a higher priority interrupt preempts a lower priority one.

### 2. Step-by-Step Guide

#### Phase A: Setup
*   Configure **EXTI0** (Button) as Low Priority (2).
*   Configure **SysTick** as High Priority (1).

#### Phase B: Coding
1.  In `EXTI0_IRQHandler`:
    *   Turn ON Orange LED.
    *   Busy wait for 2 seconds (simulating long task).
    *   Turn OFF Orange LED.
2.  In `SysTick_Handler` (fires every 1ms):
    *   Toggle Green LED.

#### Phase C: Observation
*   **Scenario 1 (Preemption Enabled):** Press button. Orange LED turns ON. Green LED *continues blinking* rapidly (because SysTick preempts the busy wait).
*   **Scenario 2 (SysTick Priority Lowered to 3):** Press button. Orange LED turns ON. Green LED *stops blinking* for 2 seconds (because SysTick cannot preempt EXTI0).

### 3. Verification
Use an oscilloscope or logic analyzer to see the Green LED signal pausing in Scenario 2.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Software Interrupts
- **Goal:** Trigger an interrupt manually from software.
- **Task:** Use the `NVIC->STIR` register (Software Trigger Interrupt Register) to fire EXTI0 without pressing the button.

### Lab 3: Race Conditions
- **Goal:** Corrupt a shared variable.
- **Task:**
    1.  Global `volatile int counter = 0;`.
    2.  Main loop: `counter++`.
    3.  ISR: `counter++`.
    4.  Run for a while and check if `counter` value makes sense (it's hard to catch, but theoretically possible if `counter++` isn't atomic).
    5.  Fix: Disable interrupts around `counter++` in main loop.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. ISR Not Firing
*   **Cause:**
    *   Interrupt Masked in EXTI (`IMR`).
    *   Interrupt Not Enabled in NVIC (`ISER`).
    *   Global Interrupts Disabled (`PRIMASK`).
    *   Wrong Vector Name (Linker doesn't link `EXTI0_IRQHandler` to the vector table).
*   **Solution:** Check all enable bits. Verify function name spelling matches startup file.

#### 2. Stuck in ISR
*   **Cause:** Forgot to clear the Pending Flag (`EXTI->PR`).
*   **Result:** As soon as ISR returns, NVIC sees pending flag and re-enters ISR immediately. Infinite loop.
*   **Solution:** Always write 1 to the PR bit at the end of ISR.

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **Keep ISRs Short:** Do the minimum work (set a flag, copy data) and return. Offload processing to the main loop or a lower-priority task.
- **Tail Chaining:** Cortex-M automatically handles back-to-back interrupts without popping/pushing stack, saving cycles.

### Code Quality
- **Volatile:** Any variable shared between ISR and Main Loop MUST be `volatile`.
- **Atomic Access:** Protect shared data with Critical Sections (`__disable_irq()`).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `EXTI->PR` and `NVIC->ICPR`?
    *   **A:** `EXTI->PR` is the peripheral status flag (the source). `NVIC->ICPR` is the controller's pending status. You usually need to clear the peripheral flag.
2.  **Q:** Can a Priority 0 interrupt preempt a Priority 0 interrupt?
    *   **A:** No. Preemption requires strictly higher priority (lower number).

### Challenge Task
> **Task:** Implement a "Debounced Interrupt". In the ISR, disable the EXTI line. Start a Timer for 50ms. In the Timer ISR, check the button state. If still pressed, register the click and re-enable EXTI.

---

## 📚 Further Reading & References
- [A Beginner’s Guide on Interrupt Latency](https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/beginner-guide-on-interrupt-latency-and-interrupt-latency-of-the-cortex-m-processors)
- [STM32 NVIC Application Note](https://www.st.com/resource/en/application_note/dm00042778-stm32f405-415-stm32f407-417-stm32f427-437-and-stm32f429-439-interrupts-stmicroelectronics.pdf)

---
