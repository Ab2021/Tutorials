# Day 89: Unit Testing & TDD
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
1.  **Explain** the benefits of Unit Testing and Test-Driven Development (TDD) in embedded systems.
2.  **Integrate** the **Unity** test framework into a C project.
3.  **Write** unit tests for hardware-independent logic (e.g., Circular Buffers, Parsers).
4.  **Mock** hardware dependencies using function pointers or link-time substitution to test drivers on a PC.
5.  **Automate** test execution using a Makefile.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC (Host) - Primary testing environment.
    *   STM32F4 (Target) - Secondary verification.
*   **Software Required:**
    *   GCC (Native for PC, ARM for STM32).
    *   [Unity Test Framework](https://github.com/ThrowTheSwitch/Unity).
*   **Prior Knowledge:**
    *   C Function Pointers.
    *   Modular Design (Day 79 - BSP).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why TDD?
Embedded code is often "Spaghetti" mixed with hardware registers.
*   **Problem:** To test a logic bug in the "Clap Detector", you have to flash the board, clap your hands, and watch an LED. This takes 2 minutes per cycle.
*   **Solution:** TDD. Write a test that feeds an array of numbers (simulating audio) to the detector and checks the output. Runs in 10ms on your PC.
*   **Cycle:** Red (Write fail test) -> Green (Write code to pass) -> Refactor.

### 🔹 Part 2: Test Frameworks
*   **Unity:** Lightweight, C-only. Perfect for embedded.
*   **CppUTest:** C++, powerful mocking.
*   **GoogleTest:** Heavy, requires C++.

We will use **Unity**.
*   `TEST_ASSERT_EQUAL(expected, actual)`
*   `TEST_ASSERT_TRUE(condition)`
*   `RUN_TEST(func)`

### 🔹 Part 3: Mocking Hardware
How do we test `void LED_On() { HAL_GPIO_WritePin(...); }` on a PC? The PC doesn't have `HAL_GPIO_WritePin`.
*   **Technique 1: Link Seams.** Create a `mock_stm32_hal.c` that implements empty HAL functions or logs calls. Link this instead of the real HAL.
*   **Technique 2: Function Pointers.** Inject the hardware driver into the logic.

---

## 💻 Implementation: Setting up Unity

> **Instruction:** Create a simple test runner on the Host (PC).

### 👨‍💻 Code Implementation

#### Step 1: Directory Structure
```text
Project/
├── src/
│   ├── circular_buffer.c
│   └── circular_buffer.h
├── test/
│   ├── unity/ (Unity Source)
│   ├── test_circular_buffer.c
│   └── test_runner.c
└── Makefile
```

#### Step 2: The Logic (`circular_buffer.c`)
```c
#include "circular_buffer.h"

int CB_Push(CircularBuffer_t *cb, int data) {
    int next = (cb->head + 1) % cb->size;
    if (next == cb->tail) return -1; // Full
    cb->buffer[cb->head] = data;
    cb->head = next;
    return 0;
}

int CB_Pop(CircularBuffer_t *cb, int *data) {
    if (cb->head == cb->tail) return -1; // Empty
    *data = cb->buffer[cb->tail];
    cb->tail = (cb->tail + 1) % cb->size;
    return 0;
}
```

#### Step 3: The Test (`test_circular_buffer.c`)
```c
#include "unity.h"
#include "circular_buffer.h"

CircularBuffer_t cb;
int buffer[5];

void setUp(void) {
    // Run before each test
    cb.buffer = buffer;
    cb.size = 5;
    cb.head = 0;
    cb.tail = 0;
}

void tearDown(void) {}

void test_Push_Pop_Success(void) {
    TEST_ASSERT_EQUAL(0, CB_Push(&cb, 42));
    
    int val;
    TEST_ASSERT_EQUAL(0, CB_Pop(&cb, &val));
    TEST_ASSERT_EQUAL(42, val);
}

void test_Buffer_Full(void) {
    CB_Push(&cb, 1);
    CB_Push(&cb, 2);
    CB_Push(&cb, 3);
    CB_Push(&cb, 4); // Full (Size-1)
    
    TEST_ASSERT_EQUAL(-1, CB_Push(&cb, 5)); // Should fail
}
```

#### Step 4: The Runner (`test_runner.c`)
```c
#include "unity.h"

extern void test_Push_Pop_Success(void);
extern void test_Buffer_Full(void);

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_Push_Pop_Success);
    RUN_TEST(test_Buffer_Full);
    return UNITY_END();
}
```

---

## 💻 Implementation: Mocking Hardware

> **Instruction:** Test a driver that toggles an LED.

### 👨‍💻 Code Implementation

#### Step 1: The Driver (`led_driver.c`)
```c
#include "led_driver.h"
#include "stm32_hal.h" // We need to mock this

void LED_TurnOn(void) {
    HAL_GPIO_WritePin(GPIOD, GPIO_PIN_12, GPIO_PIN_SET);
}
```

#### Step 2: The Mock (`mock_stm32_hal.c`)
```c
#include "stm32_hal.h"

// Variables to capture state
int last_pin_state = -1;
uint16_t last_pin = 0;

void HAL_GPIO_WritePin(GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin, GPIO_PinState PinState) {
    last_pin = GPIO_Pin;
    last_pin_state = PinState;
}
```

#### Step 3: The Test (`test_led_driver.c`)
```c
#include "unity.h"
#include "led_driver.h"
#include "mock_stm32_hal.h" // Access to spy variables

void test_LED_TurnOn_Writes_High(void) {
    LED_TurnOn();
    
    TEST_ASSERT_EQUAL(GPIO_PIN_12, last_pin);
    TEST_ASSERT_EQUAL(GPIO_PIN_SET, last_pin_state);
}
```

---

## 🔬 Lab Exercise: Lab 89.1 - TDD Workflow

### 1. Lab Objectives
- Write a test *before* the code.
- Implement a "Packet Parser".

### 2. Step-by-Step Guide

#### Phase A: Requirement
Packet Format: `[START:0xAA] [LEN] [DATA...] [CHECKSUM]`

#### Phase B: Write Test
```c
void test_Parse_Valid_Packet(void) {
    uint8_t data[] = {0xAA, 0x01, 0x55, 0x00}; // Checksum 0x00 (Example)
    Packet_t pkt;
    TEST_ASSERT_EQUAL(PARSE_OK, Parse_Packet(data, 4, &pkt));
    TEST_ASSERT_EQUAL(0x55, pkt.payload[0]);
}
```

#### Phase C: Run & Fail
Compile. Error: `Parse_Packet` undefined.

#### Phase D: Implement
Write `Parse_Packet`. Compile. Run. Pass.

### 3. Verification
If test passes, change the input data to be invalid (e.g., wrong Start Byte) and ensure it returns `PARSE_ERROR`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: On-Target Testing
- **Goal:** Run Unity on STM32.
- **Task:**
    1.  Retarget `putchar` to UART (Day 86).
    2.  Include Unity sources in STM32 project.
    3.  Call `UNITY_BEGIN()` in `main()`.
    4.  View results in Serial Terminal.

### Lab 3: CMock (Automated Mocking)
- **Goal:** Generate mocks automatically.
- **Task:**
    1.  Install Ruby.
    2.  Use CMock to generate `Mockstm32_hal.c` from `stm32_hal.h`.
    3.  Use `Expect` calls: `HAL_GPIO_WritePin_Expect(GPIOD, GPIO_PIN_12, GPIO_PIN_SET);`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Segfault on PC
*   **Cause:** Dereferencing a pointer that would be valid on STM32 (e.g., `GPIOD` = `0x40020C00`) but is invalid virtual memory on PC.
*   **Solution:** In the Mock, define `GPIOD` as a pointer to a local struct, not a hardcoded address.

#### 2. Linker Errors
*   **Cause:** Multiple definitions of `HAL_GPIO_WritePin` (Real vs Mock).
*   **Solution:** Use separate Makefiles for "Release" (Real) and "Test" (Mock).

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Pure Functions:** Write logic as pure functions (Input -> Output) with no side effects (Globals/Hardware). These are easiest to test.
- **Dependency Injection:** Pass hardware handles into functions rather than using globals.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is a "Seam"?
    *   **A:** A place where you can alter the behavior of the program without editing the source code (e.g., Linker seam, Preprocessor seam).
2.  **Q:** Why run tests on PC if the target is ARM?
    *   **A:** Speed. PC tests run in milliseconds. Target tests take seconds/minutes to flash. Also, PC tools (Valgrind, Sanitizers) are better.

### Challenge Task
> **Task:** Integrate the tests into a CI pipeline (GitHub Actions). Every push should compile the tests and run them.

---

## 📚 Further Reading & References
- [Test Driven Development for Embedded C (Book)](https://pragprog.com/titles/jgade/test-driven-development-for-embedded-c/)
- [ThrowTheSwitch.org (Unity/CMock)](http://www.throwtheswitch.org/)

---
