# Day 4: Bit Manipulation and Register Operations
## Phase 1: Core Embedded Engineering Foundations | Week 1: Embedded C Fundamentals

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
1.  **Perform** complex bitwise operations (AND, OR, XOR, NOT, Shift) with confidence.
2.  **Implement** Read-Modify-Write (RMW) sequences to safely update hardware registers.
3.  **Evaluate** the pros and cons of using C Bit Fields versus Bitwise Macros.
4.  **Understand** the concept of atomicity and why it matters in interrupt-driven systems.
5.  **Create** a reusable library for register abstraction.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
*   **Prior Knowledge:**
    *   Day 2 (Bitwise macros intro)
    *   Binary and Hexadecimal number systems

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Bitwise Operations Mastery

#### 1.1 The Operators
In embedded systems, we rarely work with the "value" of a register (e.g., 1,234,567). We work with individual bits or groups of bits (fields).

*   **`&` (AND):** Used to **CLEAR** bits or **MASK** (extract) bits.
    *   `x & 0` = `0` (Clears)
    *   `x & 1` = `x` (Keeps)
*   **`|` (OR):** Used to **SET** bits.
    *   `x | 1` = `1` (Sets)
    *   `x | 0` = `x` (Keeps)
*   **`^` (XOR):** Used to **TOGGLE** bits.
    *   `x ^ 1` = `~x` (Flips)
    *   `x ^ 0` = `x` (Keeps)
*   **`~` (NOT):** Inverts all bits (1's complement).
*   **`<<` (Left Shift):** Multiplies by 2^n. Moves bits to higher significance.
*   **`>>` (Right Shift):** Divides by 2^n.
    *   *Logical Shift:* Fills with 0 (for unsigned types).
    *   *Arithmetic Shift:* Fills with sign bit (for signed types). **Danger:** Avoid right shifting signed integers in portable code.

#### 1.2 Read-Modify-Write (RMW)
To change one bit in a 32-bit register without affecting others, we must:
1.  **Read** the current value.
2.  **Modify** the specific bit(s).
3.  **Write** the new value back.

```c
// Set Bit 5
reg = reg | (1 << 5);

// Clear Bit 5
reg = reg & ~(1 << 5);
```

### 🔹 Part 2: Bit Fields vs. Macros

#### 2.1 C Bit Fields
C allows defining structure members with specific bit widths.
```c
typedef struct {
    uint32_t ENABLE : 1;  // 1 bit
    uint32_t MODE   : 2;  // 2 bits
    uint32_t        : 5;  // 5 bits padding
    uint32_t DATA   : 8;  // 8 bits
} ControlReg_t;
```
**Pros:** Readable syntax (`reg.MODE = 2;`).
**Cons:**
*   **Portability:** The ordering of bits (LSB first vs MSB first) is compiler-dependent.
*   **Atomicity:** The compiler might generate multiple instructions to update a bit field, which is not thread-safe.
*   **Recommendation:** Avoid Bit Fields for hardware registers. Use them for software flags or network packets (if endianness is handled).

#### 2.2 The Macro Approach (Standard)
This is the industry standard (used in CMSIS, Linux Kernel).
```c
#define CR_ENABLE_POS   (0U)
#define CR_ENABLE_MSK   (0x1U << CR_ENABLE_POS)

#define CR_MODE_POS     (1U)
#define CR_MODE_MSK     (0x3U << CR_MODE_POS)

// Usage
REG = (REG & ~CR_MODE_MSK) | (2U << CR_MODE_POS);
```

### 🔹 Part 3: Atomic Operations

#### 3.1 The Race Condition
Imagine `REG |= (1 << 0)` compiles to:
1.  `LDR r0, [addr]`
2.  `ORR r0, r0, #1`
3.  `STR r0, [addr]`

If an interrupt occurs after step 1, modifies `REG`, and returns, the `STR` in step 3 will overwrite the interrupt's changes with the old value (plus the set bit). This is a **Race Condition**.

#### 3.2 Solutions
1.  **Disable Interrupts:** `__disable_irq(); ... __enable_irq();`. (Heavy handed).
2.  **Atomic Instructions:** Cortex-M3/M4 has `LDREX`/`STREX` (Load/Store Exclusive).
3.  **Bit-Banding:** (Cortex-M3/M4 specific) Alias regions where writing to a word atomically sets a bit in another region.
4.  **Hardware Support:** Registers like `BSRR` (Bit Set/Reset Register) in STM32 allow atomic setting/clearing by writing 1s to specific bits. Writing 0 does nothing.

---

## 💻 Implementation: Robust Register Access Library

> **Instruction:** We will create a safer way to handle register modifications using the "Clear-then-Set" pattern.

### 🛠️ Hardware/System Configuration
STM32F4 Discovery.

### 👨‍💻 Code Implementation

#### Step 1: Helper Macros (`reg_utils.h`)

```c
#ifndef REG_UTILS_H
#define REG_UTILS_H

#include <stdint.h>

// Mask generation
#define MASK_1BIT  (0x1U)
#define MASK_2BIT  (0x3U)
#define MASK_3BIT  (0x7U)
#define MASK_4BIT  (0xFU)

// Generic Modify Macro
// REG: Register to modify
// CLEAR_MASK: Bits to clear (set to 0)
// SET_MASK: Bits to set (set to 1)
#define MODIFY_REG(REG, CLEAR_MASK, SET_MASK) \
    ((REG) = (((REG) & (~(CLEAR_MASK))) | (SET_MASK)))

// Example for a specific peripheral register
// GPIO Mode Register
#define GPIO_MODER_PIN_POS(PIN)      ((PIN) * 2U)
#define GPIO_MODER_PIN_MASK(PIN)     (0x3U << GPIO_MODER_PIN_POS(PIN))

#define GPIO_MODE_INPUT              (0x0U)
#define GPIO_MODE_OUTPUT             (0x1U)
#define GPIO_MODE_ALT                (0x2U)
#define GPIO_MODE_ANALOG             (0x3U)

// Function to set pin mode safely
static inline void GPIO_SetMode(volatile uint32_t *MODER, uint8_t pin, uint8_t mode) {
    uint32_t pos = GPIO_MODER_PIN_POS(pin);
    uint32_t mask = GPIO_MODER_PIN_MASK(pin);
    uint32_t val = (mode << pos);
    
    // Read-Modify-Write
    uint32_t temp = *MODER;
    temp &= ~mask; // Clear old mode
    temp |= val;   // Set new mode
    *MODER = temp; // Write back
}

#endif
```

#### Step 2: Atomic Bit Setting (BSRR)
The STM32 GPIO `BSRR` register is 32-bits.
*   Bits 0-15: Set the corresponding ODR bit.
*   Bits 16-31: Reset the corresponding ODR bit.

```c
// Atomic Set
static inline void GPIO_SetPinAtomic(volatile uint32_t *BSRR, uint8_t pin) {
    *BSRR = (1U << pin);
}

// Atomic Reset
static inline void GPIO_ResetPinAtomic(volatile uint32_t *BSRR, uint8_t pin) {
    *BSRR = (1U << (pin + 16));
}
```

---

## 🔬 Lab Exercise: Lab 4.1 - LED Chaser with Bit Ops

### 1. Lab Objectives
- Use shift operators to create a moving light pattern.
- Implement a "Ping-Pong" effect (Left to Right, then Right to Left).

### 2. Step-by-Step Guide

#### Phase A: Setup
Use the 4 LEDs on the Discovery Board (PD12, PD13, PD14, PD15).

#### Phase B: Coding
```c
#include "stm32f4xx.h" // Assuming standard header is available or defined manually

void delay(int n) { while(n--) __asm("nop"); }

int main(void) {
    // Enable GPIOD Clock
    RCC->AHB1ENR |= (1 << 3);

    // Configure PD12-PD15 as Output
    // We can do this in a loop
    for (int i = 12; i <= 15; i++) {
        // Clear bits (2 bits per pin)
        GPIOD->MODER &= ~(0x3 << (i * 2));
        // Set to Output (01)
        GPIOD->MODER |= (0x1 << (i * 2));
    }

    uint8_t current_led = 0; // 0 to 3
    int8_t direction = 1;    // 1 for Up, -1 for Down

    while (1) {
        // Clear all LEDs first
        // We can use BSRR Reset bits (16-31)
        // PD12 is bit 12. Reset is bit 12+16 = 28.
        // Mask for 12,13,14,15 is 0xF000. Shifted by 16 is 0xF0000000.
        GPIOD->BSRR = (0xF << (12 + 16));

        // Set the current LED
        // PD12 + current_led
        GPIOD->BSRR = (1 << (12 + current_led));

        delay(500000);

        // Update logic
        current_led += direction;
        if (current_led >= 3) direction = -1;
        if (current_led <= 0) direction = 1;
    }
}
```

#### Phase C: Analysis
*   **Observation:** The LEDs should bounce back and forth.
*   **Optimization:** Instead of clearing all and setting one, can you calculate the exact BSRR value to set the new one and clear the old one in a single write?

### 3. Verification
Verify that no "ghosting" occurs (LEDs staying dimly lit).

---

## 🧪 Additional / Advanced Labs

### Lab 2: Bit Reversal (Software vs Hardware)
- **Goal:** Cortex-M4 has a `RBIT` instruction that reverses bits in a word.
- **Task:**
    1.  Implement a C function for bit reversal.
    2.  Use inline assembly `__asm("rbit %0, %1" : "=r"(out) : "r"(in));`.
    3.  Compare execution speed (cycles).

### Lab 3: Extracting Data from a Packed Stream
- **Scenario:** A sensor sends 3 values packed into 4 bytes:
    *   Val1: 10 bits
    *   Val2: 12 bits
    *   Val3: 10 bits
- **Task:** Write a function to unpack `uint32_t raw` into a struct `{ v1, v2, v3 }` using masking and shifting.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Off-by-One Shifting
*   **Symptom:** Wrong bit set.
*   **Cause:** `1 << 1` sets bit 1 (value 2), not bit 0 (value 1).
*   **Solution:** Remember `1 << N` sets the Nth bit (0-indexed).

#### 2. Signed Shift Disasters
*   **Symptom:** High bits become 1s unexpectedly.
*   **Cause:** Right shifting a negative signed integer (`int32_t x = -1; x >> 1` is still `-1` or `0xFFFFFFFF`).
*   **Solution:** Always cast to `uint32_t` before shifting if you want logical shift (0-fill).

#### 3. Operator Precedence
*   **Symptom:** Logic fails.
*   **Cause:** `x & 1 == 0` is parsed as `x & (1 == 0)` because `==` has higher precedence than `&`.
*   **Solution:** **ALWAYS** use parentheses. `(x & 1) == 0`.

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **Constant Folding:** `(1 << 5) | (1 << 6)` is calculated by the compiler at compile-time, not runtime. Don't be afraid to write verbose constant math for readability.
- **Bit-Banding:** Use it for single-bit flags in SRAM to avoid RMW cycles.

### Code Quality
- **Magic Numbers:** `REG |= 0x40;` is bad. `REG |= CR_TXE_MASK;` is good.
- **Consistency:** Stick to either `(1 << N)` or `BIT(N)` macro style throughout the project.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the result of `0x55 & 0xAA`?
    *   **A:** `0x00`. (01010101 & 10101010 have no common bits).
2.  **Q:** How do you clear the 3rd bit of `x`?
    *   **A:** `x &= ~(1 << 3);`

### Challenge Task
> **Task:** Implement a "Circular Shift" (Rotate) function for 8-bit value. `RotateLeft(0x80, 1)` should become `0x01`.
> **Hint:** `(x << n) | (x >> (8 - n))`

---

## 📚 Further Reading & References
- [Hacker's Delight (Book)](https://en.wikipedia.org/wiki/Hacker%27s_Delight) - The bible of bit twiddling.
- [Bit Twiddling Hacks (Stanford)](https://graphics.stanford.edu/~seander/bithacks.html)

---
