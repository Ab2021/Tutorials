# Day 120: Phase 1 Final Exam & Graduation
## Phase 1: Core Embedded Engineering Foundations | Week 17: Final Project - The Smart Home Hub

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
1.  **Validate** their knowledge of Phase 1 concepts through a comprehensive exam.
2.  **Debug** complex embedded C scenarios involving pointers, interrupts, and concurrency.
3.  **Architect** a complete system solution given a high-level problem statement.
4.  **Audit** their own code portfolio for quality and completeness.
5.  **Prepare** for Phase 2 (Embedded Linux) by setting up the necessary environment.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board (for practical exam).
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Days 1-119.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Journey So Far
We have covered:
1.  **C Foundations:** Pointers, Structs, Bit Manipulation.
2.  **Architecture:** Cortex-M4, NVIC, Memory Map, Boot Process.
3.  **Peripherals:** GPIO, UART, SPI, I2C, ADC, DMA, Timers.
4.  **Protocols:** UART, I2C, SPI, MQTT, LoRa, BLE.
5.  **Systems:** RTOS Concepts, File Systems, Bootloaders, Security.
6.  **Tools:** GDB, OpenOCD, Logic Analyzer, Git, Doxygen.

### 🔹 Part 2: The Exam Structure
*   **Section A:** Theory (Multiple Choice / Short Answer).
*   **Section B:** Code Debugging (Spot the bug).
*   **Section C:** Implementation (Write a driver).
*   **Section D:** System Design (Architect a solution).

---

## 💻 Implementation: The Final Exam

> **Instruction:** Attempt these questions without looking up answers first.

### 📝 Section A: Theory

**Q1:** What is the difference between `AHB` and `APB` buses?
*   **A:** AHB (Advanced High-performance Bus) is for high-speed components (Core, DMA, Flash, RAM). APB (Advanced Peripheral Bus) is for lower-speed peripherals (UART, SPI, Timers).

**Q2:** Explain the role of the `SysTick` timer.
*   **A:** It is a core peripheral (part of Cortex-M) used to generate a periodic interrupt (usually 1ms) for the OS scheduler or `HAL_Delay`.

**Q3:** What happens if two interrupts with the *same* preemption priority occur simultaneously?
*   **A:** The one with the lower Exception Number (Vector Table index) runs first.

**Q4:** Why do we use `volatile` for memory-mapped I/O?
*   **A:** To prevent the compiler from optimizing away reads/writes, as the hardware state can change outside the program flow.

**Q5:** What is the difference between `Mode 0` and `Mode 3` in SPI?
*   **A:** Mode 0: CPOL=0, CPHA=0 (Clock Idle Low, Sample on Rising). Mode 3: CPOL=1, CPHA=1 (Clock Idle High, Sample on Rising).

---

### 📝 Section B: Code Debugging

**Bug 1: The Hanging UART**
```c
void UART_Send(char *data) {
    while(*data) {
        // Bug: Waiting for TC (Transmission Complete) instead of TXE (Transmit Data Register Empty)
        // TC is set AFTER the byte is fully sent. TXE is set when we can write the NEXT byte.
        // Using TC is slower, but works. However, if TC flag is not cleared, this might loop once.
        // REAL BUG: Not waiting BEFORE writing DR.
        USART2->DR = *data++;
        while(!(USART2->SR & USART_SR_TXE)); 
    }
}
```
*   **Fix:** Wait for TXE *before* writing to DR (or check TXE before writing).

**Bug 2: The ISR Race**
```c
volatile int counter = 0;
void EXTI0_IRQHandler(void) {
    counter++; // Not atomic!
    // Bug: Forgot to clear Pending Bit.
}
```
*   **Fix:** `EXTI->PR = (1 << 0);` at the end. Otherwise, ISR loops infinitely.

**Bug 3: The Stack Overflow**
```c
void Process_Data(void) {
    uint8_t buffer[4096]; // 4KB on Stack!
    // ...
}
```
*   **Fix:** STM32 stack is usually 1KB-2KB. Allocate large buffers globally (`static`) or use Heap (carefully).

---

### 📝 Section C: Implementation

**Task:** Write a function to read a register from an I2C Sensor.
*   **Address:** 0x68 (Sensor).
*   **Register:** 0x75 (Who Am I).
*   **Return:** Value.

```c
uint8_t I2C_ReadReg(I2C_TypeDef *I2Cx, uint8_t devAddr, uint8_t regAddr) {
    // 1. Start
    I2Cx->CR1 |= I2C_CR1_START;
    while(!(I2Cx->SR1 & I2C_SR1_SB));
    
    // 2. Send Address (Write)
    I2Cx->DR = (devAddr << 1);
    while(!(I2Cx->SR1 & I2C_SR1_ADDR));
    (void)I2Cx->SR2; // Clear ADDR
    
    // 3. Send Register Address
    I2Cx->DR = regAddr;
    while(!(I2Cx->SR1 & I2C_SR1_TXE));
    
    // 4. Restart
    I2Cx->CR1 |= I2C_CR1_START;
    while(!(I2Cx->SR1 & I2C_SR1_SB));
    
    // 5. Send Address (Read)
    I2Cx->DR = (devAddr << 1) | 1;
    while(!(I2Cx->SR1 & I2C_SR1_ADDR));
    
    // 6. Disable ACK & Stop (Single Byte Read)
    I2Cx->CR1 &= ~I2C_CR1_ACK;
    (void)I2Cx->SR2; // Clear ADDR
    I2Cx->CR1 |= I2C_CR1_STOP;
    
    // 7. Read Data
    while(!(I2Cx->SR1 & I2C_SR1_RXNE));
    return I2Cx->DR;
}
```

---

### 📝 Section D: System Design

**Scenario:** Design a "Black Box" for a Car.
*   **Inputs:** GPS (UART), Accelerometer (SPI), OBD-II (CAN).
*   **Storage:** SD Card (SPI).
*   **Power:** Car Battery (12V) -> 3.3V. Must handle "Cranking" (voltage drop) and "Load Dump" (voltage spike).
*   **Constraint:** Must save the last 10 seconds of data *after* a crash (power loss).

**Architecture:**
1.  **Power:** Supercapacitor backup to keep MCU alive for 10s after main power cut.
2.  **Memory:** Circular Buffer in RAM (10s history).
3.  **Trigger:** Accelerometer High-G interrupt.
4.  **Action:** On Trigger, flush RAM buffer to SD Card (or internal Flash if SD is too slow/vulnerable).
5.  **Protocol:** CAN Bus interrupt for OBD-II data. DMA for GPS/Accel to minimize CPU load.

---

## 🔬 Lab Exercise: Lab 120.1 - The Graduation Project

### 1. Lab Objectives
- Submit the "Smart Home Hub" (Week 17 Project).
- Ensure it meets all requirements (Day 113).
- Record a 2-minute demo video.

### 2. Step-by-Step Guide

#### Phase A: Final Polish
1.  Run `cppcheck` one last time.
2.  Generate Doxygen.
3.  Clean up code formatting.

#### Phase B: Documentation
1.  Update `README.md`.
2.  Include wiring diagram (Fritzing or Hand drawn).

#### Phase C: Submission
1.  Push to GitHub.
2.  Tag `v1.0`.

### 3. Verification
Does it compile? Does it run? Is the code readable?

---

## 🧪 Additional / Advanced Labs

### Lab 2: The "Impossible" Bug
- **Goal:** Fix a provided binary.
- **Task:**
    1.  Download `broken.elf`.
    2.  Flash it. It HardFaults.
    3.  Use GDB to find the line.
    4.  (It's a stack corruption caused by recursive `printf`).

### Lab 3: Phase 2 Prep
- **Goal:** Linux Environment.
- **Task:**
    1.  Install WSL2 (Windows Subsystem for Linux) or VirtualBox (Ubuntu 22.04).
    2.  Install `build-essential`, `git`, `vim`.
    3.  Compile "Hello World" in Linux.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Burnout
*   **Cause:** 120 Days is a long time.
*   **Solution:** Take a break! You have completed Phase 1. Celebrate.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Refactoring:** Now that you know more, look at your Day 1 code. It probably looks terrible. That's good! It means you learned.

---

## 🧠 Assessment & Review

### Graduation Checklist
- [ ] Completed all 120 Days.
- [ ] Built the Smart Home Hub.
- [ ] Created a GitHub Portfolio.
- [ ] Updated Resume.
- [ ] Ready for Phase 2 (Linux Kernel & Drivers).

### Challenge Task
> **Task:** "The Bridge". Write a C program on your PC (Linux) that talks to your STM32 over UART. This bridges the gap between Bare Metal (Phase 1) and Host Systems (Phase 2).

---

## 📚 Further Reading & References
- [Embedded Systems Roadmap](https://roadmap.sh/embedded)
- [Linux Device Drivers (LDD3)](https://lwn.net/Kernel/LDD3/)

---
