# Day 2: C Programming Review & Embedded Specifics
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
1.  **Differentiate** between standard C and Embedded C.
2.  **Select** the appropriate data types for specific embedded constraints (size, signedness).
3.  **Apply** the `volatile`, `const`, and `static` keywords correctly in hardware-facing code.
4.  **Understand** storage classes and their impact on memory location (Stack vs. Heap vs. Data vs. BSS).
5.  **Perform** bitwise operations efficiently for register manipulation.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board (for testing code snippets)
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   GDB for inspecting variable locations
*   **Prior Knowledge:**
    *   Day 1 concepts (Memory hierarchy, Compilation process)
*   **Datasheets:**
    *   [C Standard (C99 or C11) Reference](https://en.cppreference.com/w/c)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Embedded C vs. Standard C

#### 1.1 The Myth of "Embedded C"
Strictly speaking, "Embedded C" is not a separate language. It is standard C (ISO C) used with a specific set of constraints and coding styles tailored for non-hosted (bare-metal) environments.

**Key Differences:**
*   **Resource Constraints:** In desktop C, `malloc(1GB)` might work. In embedded, `malloc(1KB)` might crash the system.
*   **Hardware Access:** Embedded C relies heavily on direct memory access (pointers to fixed addresses) to control hardware.
*   **Real-Time Behavior:** Code must be deterministic. Recursion is often banned because stack usage is hard to predict.
*   **Startup:** Standard C starts at `main()`. Embedded C starts at the Reset Vector, initializes hardware, sets up the stack, and *then* calls `main()`.

#### 1.2 Fixed-Width Integer Types (`<stdint.h>`)
In the old days, `int` could be 16-bit or 32-bit depending on the architecture. This is dangerous in embedded systems where a register is exactly 32 bits.

**Always use `<stdint.h>` types:**
*   `uint8_t`: Unsigned 8-bit integer (0 to 255). Perfect for byte data.
*   `int8_t`: Signed 8-bit integer (-128 to 127).
*   `uint16_t`: Unsigned 16-bit (0 to 65535). Good for 10-bit/12-bit ADC values.
*   `uint32_t`: Unsigned 32-bit (0 to 4.29 Billion). Standard for ARM registers.
*   `uint64_t`: Unsigned 64-bit. Use sparingly (slow on 32-bit MCU).

**Why Unsigned?**
Hardware registers are collections of bits, not numbers. Bit 31 isn't a sign bit; it's just the 32nd switch. Using `int32_t` for a register can cause issues with right-shifting (arithmetic shift vs. logical shift).

### 🔹 Part 2: Critical Keywords in Embedded

#### 2.1 The `volatile` Keyword
This is the most important keyword for embedded engineers. It tells the compiler: *"Do not optimize this variable. Its value can change at any time, outside the control of this code."*

**Use Cases:**
1.  **Memory-Mapped Registers:** Hardware changes the value (e.g., a status flag).
2.  **Global Variables Shared with ISRs:** An interrupt updates a flag; the main loop reads it.
3.  **Memory-Mapped I/O:** Writing to a port (e.g., toggling a pin).

**Example of Optimization Failure:**
```c
// Without volatile
int *status_reg = (int *)0x40001000;
while (*status_reg == 0) {
    // Do nothing
}
```
*Compiler's View:* "The variable `*status_reg` is never written to inside this loop. So, I can just read it once, check if it's 0, and if so, loop forever."
*Result:* The code hangs forever, even if the hardware sets the register to 1.

**Correct Code:**
```c
volatile int *status_reg = (int *)0x40001000;
// Compiler generates a LOAD instruction in every iteration
```

#### 2.2 The `const` Keyword
`const` means "read-only". It protects you from accidentally modifying variables.
*   **Flash Storage:** In embedded, `const` global variables are often placed in Flash (ROM) to save precious RAM.
    ```c
    const uint8_t font_table[] = {0x00, 0xFF, ...}; // Goes to .rodata (Flash)
    ```
*   **Const Pointers:**
    *   `const int *ptr`: Pointer to constant data (cannot change value).
    *   `int * const ptr`: Constant pointer (cannot change address).
    *   `const int * const ptr`: Constant pointer to constant data (hardware register that is read-only).

#### 2.3 The `static` Keyword
*   **Inside Function:** Persists across function calls. Stored in `.data` or `.bss`, not Stack.
*   **Global Scope:** Limits visibility to the current file (private). Crucial for encapsulation in C.

### 🔹 Part 3: Storage Classes & Memory Layout

#### 3.1 Where do variables live?
| Variable Type | Segment | Location | Lifetime |
| :--- | :--- | :--- | :--- |
| `int global_var = 10;` | `.data` | RAM (init from Flash) | Program |
| `int global_zero;` | `.bss` | RAM (zeroed) | Program |
| `const int config = 5;` | `.rodata` | Flash | Program |
| `void func() { int loc; }` | Stack | RAM | Function |
| `malloc(10)` | Heap | RAM | Dynamic |

#### 3.2 Stack vs. Heap
*   **Stack:** Fast, deterministic, auto-managed. Used for local variables, return addresses. **Risk:** Stack Overflow (recursion, large arrays).
*   **Heap:** Flexible, manual (`malloc`/`free`). **Risk:** Fragmentation, Memory Leaks, nondeterministic allocation time.
*   **Rule of Thumb:** Avoid Heap in small embedded systems (bare-metal). Use static allocation pools instead.

---

## 💻 Implementation: Bit Manipulation Library

> **Instruction:** We will create a robust set of macros for bit manipulation, which is the bread and butter of driver development.

### 🛠️ Hardware/System Configuration
No specific hardware needed, but we will test this on the STM32.

### 👨‍💻 Code Implementation

#### Step 1: Macro Definitions (`bit_ops.h`)

```c
#ifndef BIT_OPS_H
#define BIT_OPS_H

#include <stdint.h>

// Set bit N
#define BIT_SET(REG, N)      ((REG) |= (1U << (N)))

// Clear bit N
#define BIT_CLR(REG, N)      ((REG) &= ~(1U << (N)))

// Toggle bit N
#define BIT_TGL(REG, N)      ((REG) ^= (1U << (N)))

// Check if bit N is set (returns non-zero if set)
#define BIT_CHECK(REG, N)    ((REG) & (1U << (N)))

// Modify a multi-bit field
// MASK: The bitmask for the field (e.g., 0x03 for 2 bits)
// POS: The starting position
// VAL: The value to write
#define FIELD_WRITE(REG, MASK, POS, VAL) \
    ((REG) = ((REG) & ~((MASK) << (POS))) | (((VAL) & (MASK)) << (POS)))

// Read a multi-bit field
#define FIELD_READ(REG, MASK, POS) \
    (((REG) >> (POS)) & (MASK))

#endif // BIT_OPS_H
```

#### Step 2: Testing with "Fake" Registers (`main.c`)

```c
#include <stdio.h>
#include "bit_ops.h"

// Simulate a 32-bit register
volatile uint32_t FAKE_REG = 0x00000000;

int main(void) {
    printf("Initial: 0x%08X\n", FAKE_REG);

    // 1. Set Bit 5
    BIT_SET(FAKE_REG, 5);
    printf("Set Bit 5: 0x%08X (Expected: 0x00000020)\n", FAKE_REG);

    // 2. Clear Bit 5
    BIT_CLR(FAKE_REG, 5);
    printf("Clr Bit 5: 0x%08X (Expected: 0x00000000)\n", FAKE_REG);

    // 3. Toggle Bit 31
    BIT_TGL(FAKE_REG, 31);
    printf("Tgl Bit 31: 0x%08X (Expected: 0x80000000)\n", FAKE_REG);

    // 4. Field Write: Write value 0x5 (101 binary) to bits [3:1]
    // Mask for 3 bits is 0x7 (111 binary)
    FIELD_WRITE(FAKE_REG, 0x7, 1, 0x5);
    printf("Field Write: 0x%08X (Expected: 0x8000000A)\n", FAKE_REG);
    // Explanation: 0x5 << 1 = 0xA. 0x80000000 | 0xA = 0x8000000A.

    return 0;
}
```

---

## 🔬 Lab Exercise: Lab 2.1 - Data Type Sizes

### 1. Lab Objectives
- Verify the size of standard C types on the ARM Cortex-M4 architecture.
- Observe memory alignment padding in structures.

### 2. Step-by-Step Guide

#### Phase A: Setup
1.  Use the project setup from Day 1.
2.  Create a new file `lab2_types.c`.

#### Phase B: Coding
```c
#include <stdint.h>
#include <stdio.h> // Requires semihosting or UART redirection

struct Aligned {
    uint8_t a;
    uint32_t b;
    uint8_t c;
};

struct Packed {
    uint8_t a;
    uint32_t b;
    uint8_t c;
} __attribute__((packed));

void lab2_main(void) {
    printf("Size of char: %d\n", sizeof(char));
    printf("Size of int: %d\n", sizeof(int));
    printf("Size of long: %d\n", sizeof(long));
    printf("Size of uint32_t: %d\n", sizeof(uint32_t));
    
    printf("Size of struct Aligned: %d\n", sizeof(struct Aligned));
    printf("Size of struct Packed: %d\n", sizeof(struct Packed));
}
```

#### Phase C: Analysis
1.  Run the code (using GDB or UART output).
2.  **Expected Output:**
    *   `char`: 1
    *   `int`: 4
    *   `long`: 4 (On ARM32, long is 32-bit. On x86_64, it's 64-bit!)
    *   `uint32_t`: 4
    *   `struct Aligned`: 12 bytes. (1 byte + 3 padding + 4 bytes + 1 byte + 3 padding).
    *   `struct Packed`: 6 bytes. (1 + 4 + 1).

### 3. Verification
- Use GDB to inspect the memory address of `struct Aligned` members.
- `print &s.a`, `print &s.b`. You will see `&s.b` is `&s.a + 4`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Volatile Optimization Check
- **Goal:** Prove that `volatile` matters.
- **Steps:**
    1.  Write a loop `while(flag);` where `flag` is a global `int` (not volatile).
    2.  Compile with optimization `-O2`.
    3.  Disassemble (`objdump -d`).
    4.  Observe that the compiler generates an infinite loop without checking memory.
    5.  Change to `volatile int flag`.
    6.  Observe the `LDR` instruction inside the loop.

### Lab 3: Endianness Test
- **Scenario:** You are receiving data from a network (Big Endian) but your ARM is Little Endian.
- **Task:** Write a function `uint32_t swap_endian(uint32_t val)` using bitwise operations. Verify it by storing `0x12345678` and checking memory to see `78 56 34 12`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Unaligned Access Usage Fault"
*   **Symptom:** The processor jumps to the UsageFault handler.
*   **Cause:** Trying to read a `uint32_t` from an address that is not divisible by 4 (e.g., `0x20000001`). This often happens when casting `uint8_t*` buffer to `uint32_t*`.
*   **Solution:** Ensure pointers are aligned. Use `memcpy` for unaligned copies.

#### 2. Structure Padding Surprises
*   **Symptom:** `sizeof(struct)` is larger than expected. Communication protocol fails because bytes are shifted.
*   **Solution:** Use `__attribute__((packed))` for protocol structures, but be aware of unaligned access penalties.

---

## ⚡ Optimization & Best Practices

### Performance Optimization
- **Alignment:** Always align data to word boundaries (4 bytes) for fastest access. Packed structures are slower because the CPU may need two fetch cycles to read one integer.
- **Bit Banding:** Cortex-M3/M4 supports "Bit Banding" which allows atomic access to single bits in SRAM/Peripheral regions via a special alias region.

### Code Quality
- **Typedefs:** Use `typedef` for structures to avoid typing `struct MyStruct` everywhere.
- **Naming:** Use `_t` suffix for typedefs (e.g., `gpio_config_t`).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the value of `sizeof(struct { uint8_t a; uint16_t b; })`?
    *   **A:** 4 bytes. (1 byte `a` + 1 byte padding + 2 bytes `b`).
2.  **Q:** Why shouldn't we use `int` for register definitions?
    *   **A:** The size of `int` is compiler/architecture dependent. Registers have fixed widths.

### Challenge Task
> **Task:** Implement a macro `BIT_REVERSE_32(x)` that reverses the order of bits in a 32-bit integer (Bit 0 becomes Bit 31).
> **Hint:** Use a divide-and-conquer approach (swap adjacent bits, then swap 2-bit pairs, then nibbles, bytes, and half-words).

---

## 📚 Further Reading & References
- [Embedded C Coding Standard (Barr Group)](https://barrgroup.com/embedded-systems/books/embedded-c-coding-standard)
- [C Traps and Pitfalls](https://www.amazon.com/Traps-Pitfalls-Andrew-Koenig/dp/0201179288)
- [Understanding volatile keyword](https://barrgroup.com/embedded-systems/how-to/c-volatile-keyword)

---
