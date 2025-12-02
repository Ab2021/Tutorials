# Day 174: Week 27 Review & Project (The Validation Suite)
## Phase 3: Camera Systems & ISP | Week 27: Testing, Validation & Compliance

---

## 🎯 Learning Objectives
1.  **Synthesize** the V-Model of testing: Unit -> Integration -> System -> Acceptance.
2.  **Build** an Automated Validation Suite (Python + PyTest) that controls hardware (Power Supply, Relay, Camera).
3.  **Execute** a Long-Duration Reliability Run (The "Burn-In").
4.  **Generate** a Compliance Matrix (Requirements vs Test Results).
5.  **Prepare** for Week 50 (The Final Capstone).

---

## 📚 Week 27 Recap

### Topics Covered

**Day 168: Image Quality Testing**
- MTF50, SNR, Color Accuracy, Imatest.

**Day 169: Automotive Standards**
- AEC-Q100, ISO 16750, Load Dump, Environmental.

**Day 170: Functional Safety**
- ISO 26262, ASIL, Safety Mechanisms, FMEA.

**Day 171: EMC/EMI Testing**
- CISPR 25, Radiated Emissions, Shielding.

**Day 172: Production Testing**
- EOL, Cycle Time, Yield, Calibration.

**Day 173: Compliance**
- CE, FCC, RoHS, DoC.

---

## 💻 Week 27 Project: The Automated Validation Suite

### Objective
Create a Python framework that automates the "Stress Test" of the camera. It should cycle power, capture images, check for freezing, and log temperature.

### Hardware Setup
1.  **DUT (Device Under Test):** Your Camera System.
2.  **Programmable Power Supply:** To simulate voltage dips/spikes (e.g., via SCPI commands).
3.  **Relay Board:** To physically cut power (Simulate loose cable).
4.  **Host PC:** Runs the test script.

### Software Architecture (PyTest)
*   **Fixture:** `setup_camera()`, `teardown_camera()`.
*   **Tests:**
    *   `test_boot_time()`
    *   `test_image_sharpness()`
    *   `test_power_cycle_1000_times()`
    *   `test_thermal_throttling()`

### Implementation Snippet

```python
import pytest
import time
import cv2
import serial # For Power Supply

# Mock Power Supply Control
def set_voltage(volts):
    print(f"PSU: Setting {volts}V")
    time.sleep(0.1)

def power_cycle():
    set_voltage(0)
    time.sleep(1)
    set_voltage(12)
    time.sleep(5) # Wait for boot

@pytest.fixture
def camera():
    # Setup
    cap = cv2.VideoCapture(0)
    yield cap
    # Teardown
    cap.release()

def test_boot_reliability(camera):
    for i in range(10): # Run 10 times (In real life: 1000)
        print(f"Cycle {i}")
        power_cycle()
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), "Camera failed to open after reboot"
        
        ret, frame = cap.read()
        assert ret, "Failed to grab frame"
        assert frame.mean() > 10, "Image is black"
        
        cap.release()

def test_voltage_drop():
    # Simulate Brownout
    set_voltage(12)
    time.sleep(2)
    set_voltage(6) # Drop to 6V
    time.sleep(0.1)
    set_voltage(12) # Recover
    
    # Camera should either recover or reboot safely
    # It should NOT hang
    time.sleep(5)
    cap = cv2.VideoCapture(0)
    assert cap.isOpened()
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: The "Monkey Test"

**Objective:** Random inputs.

**Steps:**
1.  Write a script that randomly changes settings (Exposure, Gain, Resolution) as fast as possible.
2.  Run for 1 hour.
3.  **Goal:** Driver should not crash. Kernel should not panic.

### Lab 2: Thermal Chamber Simulation

**Objective:** Software-controlled heating.

**Steps:**
1.  Run `stress-ng` on the DUT to self-heat.
2.  Monitor temperature.
3.  When Temp > 80C, verify that `test_image_sharpness` still passes (or degrades gracefully).

### Lab 3: Report Generation

**Objective:** PDF Output.

**Steps:**
1.  Use `pytest-html` plugin.
2.  Run `pytest --html=report.html`.
3.  **Result:** A nice HTML report with Pass/Fail status, logs, and screenshots of failed frames.

---

## 📝 Assessment Questions

### Comprehensive Questions

1.  **Why is "Regression Testing" important?** (To ensure that fixing Bug A didn't break Feature B. Run the full suite after every code change).
2.  **What is "HIL" (Hardware-in-the-Loop)?** (Testing the embedded software with real hardware inputs/outputs simulated by a test rig).
3.  **Difference between Verification and Validation?** (Verification: "Are we building the product right?" (Does it meet spec?). Validation: "Are we building the right product?" (Does it meet user needs?)).

### Practical Challenges

1.  **Bug Tracker:** Log a bug in Jira/GitHub. "Camera hangs after 453 power cycles. Logs attached."
2.  **Traceability Matrix:** Link `test_boot_reliability` to Requirement `REQ-001: System shall boot in < 5s`.

---

## 📚 Resources & Next Steps

### Week 27 Summary

**Completed:**
- ✅ **IQ:** Sharpness, Color.
- ✅ **Standards:** Automotive, Industrial.
- ✅ **Safety:** ASIL, FMEA.
- ✅ **EMC:** Noise.
- ✅ **Production:** EOL.
- ✅ **Project:** Automated Validation.

### Week 50 Preview (The Final Capstone)

**Topics:**
- **The Masterpiece:** Building a complete Autonomous Mobile Robot (AMR) Vision System.
- **Integration:** Camera + AI + Control + Cloud.
- **Presentation:** Demo video and documentation.
- **Graduation:** The end of Phase 3.

---

**Day 174 Complete** | Phase 3: Camera Systems & ISP | Week 27: Testing, Validation & Compliance


