# Day 117: Testing & Verification
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
1.  **Apply** the V-Model of testing to the Smart Home Hub project.
2.  **Write** Unit Tests for the Command Parser using the Unity framework.
3.  **Perform** Integration Testing to verify BSP and Middleware interaction.
4.  **Execute** System Testing (End-to-End) using a Python test harness.
5.  **Analyze** code quality using Static Analysis tools (Cppcheck).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   PC with Python.
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   [Unity Test Framework](http://www.throwtheswitch.org/unity)
    *   [Cppcheck](http://cppcheck.sourceforge.net/)
*   **Prior Knowledge:**
    *   Day 89 (Unit Testing)
    *   Day 116 (App Logic)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The V-Model
1.  **Requirements** <-> **Acceptance Testing:** Does it meet the user's needs? (Does it log data?)
2.  **System Design** <-> **System Testing:** Does the whole system work? (WiFi + MQTT + Sensors).
3.  **Module Design** <-> **Integration Testing:** Do modules talk correctly? (App -> Middleware -> BSP).
4.  **Implementation** <-> **Unit Testing:** Does the function work? (`Parse_JSON` returns correct struct).

### 🔹 Part 2: Test Coverage
*   **Statement Coverage:** Did every line run?
*   **Branch Coverage:** Did every `if/else` take both paths?
*   **Path Coverage:** Did every combination of paths run?

---

## 💻 Implementation: Unit Testing (Command Parser)

> **Instruction:** Test the logic that parses "RELAY=1".

### 👨‍💻 Code Implementation

#### Step 1: The Code to Test (`cmd_parser.c`)
```c
#include "cmd_parser.h"
#include <string.h>
#include <stdio.h>

int Parse_Command(const char *input, Command_t *out) {
    if (input == NULL || out == NULL) return -1;
    
    if (strncmp(input, "RELAY=", 6) == 0) {
        int val = input[6] - '0';
        if (val == 0 || val == 1) {
            out->type = CMD_RELAY;
            out->value = val;
            return 0;
        }
    }
    return -2; // Invalid Format
}
```

#### Step 2: The Test (`test_cmd.c`)
```c
#include "unity.h"
#include "cmd_parser.h"

void setUp(void) {}
void tearDown(void) {}

void test_Parse_RelayOn(void) {
    Command_t cmd;
    int res = Parse_Command("RELAY=1", &cmd);
    TEST_ASSERT_EQUAL(0, res);
    TEST_ASSERT_EQUAL(CMD_RELAY, cmd.type);
    TEST_ASSERT_EQUAL(1, cmd.value);
}

void test_Parse_Invalid(void) {
    Command_t cmd;
    int res = Parse_Command("RELAY=9", &cmd);
    TEST_ASSERT_EQUAL(-2, res);
}

void test_Parse_Null(void) {
    int res = Parse_Command(NULL, NULL);
    TEST_ASSERT_EQUAL(-1, res);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_Parse_RelayOn);
    RUN_TEST(test_Parse_Invalid);
    RUN_TEST(test_Parse_Null);
    return UNITY_END();
}
```

---

## 💻 Implementation: System Testing (Python Harness)

> **Instruction:** Automate the End-to-End test.

### 👨‍💻 Code Implementation

#### Step 1: Python Script (`system_test.py`)
```python
import serial
import time
import paho.mqtt.client as mqtt

# Config
SERIAL_PORT = "COM3"
MQTT_BROKER = "test.mosquitto.org"
TOPIC_CMD = "home/cmd"
TOPIC_STATUS = "home/status"

# Setup
ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
client = mqtt.Client()
client.connect(MQTT_BROKER)

def test_relay_control():
    print("Testing Relay Control...")
    
    # 1. Send Command via MQTT
    client.publish(TOPIC_CMD, "RELAY=1")
    
    # 2. Wait for Log on UART
    start = time.time()
    while time.time() - start < 5:
        line = ser.readline().decode('utf-8').strip()
        if "Relay ON" in line:
            print("PASS: Relay turned ON")
            return
            
    print("FAIL: Timeout waiting for Relay ON")

if __name__ == "__main__":
    test_relay_control()
```

---

## 🔬 Lab Exercise: Lab 117.1 - Static Analysis

### 1. Lab Objectives
- Run Cppcheck on the project.
- Identify potential bugs (Memory leaks, Uninitialized variables).

### 2. Step-by-Step Guide

#### Phase A: Run Cppcheck
Terminal:
```bash
cppcheck --enable=all --inconclusive --std=c99 Drivers/ App/ Middlewares/
```

#### Phase B: Analyze Output
*   `[App/app_main.c:45]: (error) Uninitialized variable: evt` -> Fix it!
*   `[Drivers/BSP/bsp_wifi.c:20]: (warning) Unused variable: ret` -> Remove it.

### 3. Verification
Fix all errors and warnings. Run again until clean.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Fuzz Testing
- **Goal:** Robustness.
- **Task:**
    1.  Send random garbage strings via UART/MQTT.
    2.  Ensure system does not HardFault.
    3.  Use a script to send 1000 random packets.

### Lab 3: Long Duration Test (Soak Test)
- **Goal:** Memory Leaks.
- **Task:**
    1.  Run the system for 24 hours.
    2.  Monitor Heap usage (if using malloc) or Stack usage.
    3.  Monitor WiFi reconnection count.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Unity Linker Error
*   **Cause:** Missing `unity.c` in build.
*   **Solution:** Add `unity.c` to Makefile sources.

#### 2. Python Serial Access Denied
*   **Cause:** Terminal app (Putty) still open.
*   **Solution:** Close other apps using the COM port.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Testable Code:** Write pure functions (no side effects, no globals) where possible. They are easiest to test.
- **Mocking:** Use `CMock` to mock hardware dependencies (e.g., mock `HAL_GPIO_WritePin` to verify it was called).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Regression Testing"?
    *   **A:** Re-running old tests after a code change to ensure you didn't break existing functionality.
2.  **Q:** Why is Static Analysis useful?
    *   **A:** It finds bugs *before* you even run the code (e.g., buffer overflows, logic errors).

### Challenge Task
> **Task:** "CI Pipeline". Set up a GitHub Actions workflow that runs:
> 1.  Cppcheck.
> 2.  Unit Tests (on host).
> 3.  Build Firmware (ARM GCC).
> On every push.

---

## 📚 Further Reading & References
- [Test Driven Development for Embedded C (Grenning)](https://pragprog.com/titles/jgade/test-driven-development-for-embedded-c/)

---
