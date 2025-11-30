# Day 90: Integration and System Testing
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
1.  **Distinguish** between Unit, Integration, and System Testing.
2.  **Develop** a Hardware-in-the-Loop (HIL) test harness using Python (PySerial).
3.  **Automate** end-to-end scenarios (e.g., "Send Command -> Verify LED -> Verify Response").
4.  **Implement** a basic Continuous Integration (CI) pipeline script.
5.  **Measure** Code Coverage to identify untested paths.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   PC with Python 3 installed.
*   **Software Required:**
    *   Python Libraries: `pyserial`, `pytest`.
    *   Firmware with CLI (from Day 83).
*   **Prior Knowledge:**
    *   Day 83 (CLI)
    *   Day 89 (Unit Testing)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Testing Pyramid
1.  **Unit Tests (Base):** Fast, isolated, software-only. (Day 89).
2.  **Integration Tests (Middle):** Combine modules (e.g., Driver + Middleware). Test interaction.
3.  **System Tests (Top):** Test the full product as a black box. Slow, requires hardware.

### 🔹 Part 2: Hardware-in-the-Loop (HIL)
HIL involves connecting the Device Under Test (DUT) to a "Test Harness" that simulates the environment.
*   **Stimulus:** PC sends UART command "LED ON".
*   **Response:** PC reads UART response "OK".
*   **Verification:** PC uses a webcam or light sensor to *physically* check if the LED is on (Advanced HIL). For now, we trust the firmware response + internal state query.

### 🔹 Part 3: Test Automation
Manual testing is error-prone. We use **Pytest**.
*   `def test_led_toggle():`
    *   `dut.write("led on")`
    *   `assert "OK" in dut.read()`

---

## 💻 Implementation: Python Test Harness

> **Instruction:** Create a Python script to test the CLI commands.

### 👨‍💻 Code Implementation

#### Step 1: `conftest.py` (Setup)
```python
import pytest
import serial
import time

@pytest.fixture(scope="session")
def dut():
    # Connect to Board
    ser = serial.Serial('COM3', 115200, timeout=1)
    ser.reset_input_buffer()
    
    # Reset Board (DTR toggle or just wait)
    ser.dtr = 0
    time.sleep(0.1)
    ser.dtr = 1
    time.sleep(1) # Wait for boot
    
    yield ser
    
    ser.close()

def send_cmd(ser, cmd):
    ser.write(f"{cmd}\n".encode())
    time.sleep(0.1)
    return ser.read_all().decode()
```

#### Step 2: `test_system.py` (The Tests)
```python
import time

def test_cli_ping(dut):
    """Verify CLI is responsive"""
    resp = send_cmd(dut, "status")
    assert "State:" in resp
    assert "Heap:" in resp

def test_volume_control(dut):
    """Verify Volume Command"""
    resp = send_cmd(dut, "vol 50")
    assert "Volume set to 50" in resp
    
    # Verify persistence (if implemented)
    resp = send_cmd(dut, "status")
    assert "Vol: 50" in resp

def test_invalid_command(dut):
    """Verify Error Handling"""
    resp = send_cmd(dut, "foobar")
    assert "Unknown command" in resp
```

#### Step 3: Running Tests
Terminal:
```bash
pytest -v
```
**Output:**
```text
test_system.py::test_cli_ping PASSED
test_system.py::test_volume_control PASSED
test_system.py::test_invalid_command PASSED
```

---

## 💻 Implementation: Integration Test (Firmware Side)

> **Instruction:** Add a "Self-Test" command to the firmware.

### 👨‍💻 Code Implementation

#### Step 1: Self-Test Logic
```c
bool Run_Self_Test(void) {
    bool pass = true;
    
    // 1. Check Sensors
    if (BSP_Temp_Read() == 0xFFFF) {
        printf("[FAIL] Temp Sensor\n");
        pass = false;
    }
    
    // 2. Check Flash
    if (BSP_Flash_Check() != FLASH_OK) {
        printf("[FAIL] Flash Integrity\n");
        pass = false;
    }
    
    // 3. Check Network
    if (!BSP_Net_IsLinkUp()) {
        printf("[WARN] No Ethernet\n");
        // Warning doesn't fail test
    }
    
    return pass;
}
```

#### Step 2: CLI Command
```c
else if (strcmp(cmd, "selftest") == 0) {
    if (Run_Self_Test()) printf("SELF-TEST PASSED\n");
    else printf("SELF-TEST FAILED\n");
}
```

---

## 🔬 Lab Exercise: Lab 90.1 - The Long Run

### 1. Lab Objectives
- Create a stability test script.
- Run it for 10 minutes.

### 2. Step-by-Step Guide

#### Phase A: The Script
```python
def test_stability(dut):
    start = time.time()
    while (time.time() - start) < 600: # 10 mins
        resp = send_cmd(dut, "status")
        assert "State: 1" in resp # Ensure not in Error state
        time.sleep(1)
```

#### Phase B: Execution
1.  Run `pytest`.
2.  While running, press buttons on the board.
3.  **Observation:** Test should pass. If board crashes (HardFault), the script will timeout or read garbage, failing the test.

### 3. Verification
If the test fails due to "Serial Timeout", it means the board stopped responding. Check the HardFault handler (Day 88).

---

## 🧪 Additional / Advanced Labs

### Lab 2: Fuzz Testing
- **Goal:** Break the parser.
- **Task:**
    1.  Python script sends random garbage bytes: `dut.write(os.urandom(100))`.
    2.  Then sends "status".
    3.  Board should recover and reply to "status". It should not hang.

### Lab 3: CI Pipeline (GitHub Actions)
- **Goal:** Automate.
- **Task:**
    1.  Create `.github/workflows/test.yml`.
    2.  Step 1: Install ARM GCC.
    3.  Step 2: Build Firmware (`make`).
    4.  Step 3: Run Unit Tests (Day 89).
    5.  (System Tests require physical runner or QEMU).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Serial Port Busy
*   **Cause:** Terminal open in VS Code while running Pytest.
*   **Solution:** Close all other terminal connections.

#### 2. Board Reset
*   **Cause:** Opening Serial port on some OS toggles DTR, resetting the board.
*   **Solution:** Handle this in the fixture (wait for boot).

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Testability:** Design your CLI to be machine-parsable.
    *   Bad: `Volume is now set to level 50.`
    *   Good: `VOL:50` (or JSON).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Flaky Test"?
    *   **A:** A test that sometimes passes and sometimes fails without code changes. Usually due to timing issues (e.g., `time.sleep(0.1)` is too short).
2.  **Q:** How do I test I2C failure without breaking hardware?
    *   **A:** Fault Injection. Modify the I2C driver to return `HAL_ERROR` when a specific flag is set, controllable via CLI.

### Challenge Task
> **Task:** Implement "Loopback Test". Connect UART TX to RX. Write a firmware test that sends a byte and verifies it receives it via DMA.

---

## 📚 Further Reading & References
- [Pytest Documentation](https://docs.pytest.org/en/7.1.x/)
- [Renode (Hardware Simulation for CI)](https://renode.io/)

---
