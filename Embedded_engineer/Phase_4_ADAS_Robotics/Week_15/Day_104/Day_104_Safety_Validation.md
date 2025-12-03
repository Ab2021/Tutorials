# Day 104: Safety Validation (X-in-the-Loop)
## Phase 4: ADAS & Robotics Systems | Week 15: Safety Standards (ISO 26262 & SOTIF)

---

> **📝 Day 104 Focus:**
> You designed a Watchdog (Day 103). But does it work? **Validation** is the process of proving it. We can't crash real cars to test airbags. Instead, we use **Fault Injection** in simulation to break things on purpose and measure the response.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Contrast** Verification ("Did we build the product right?") vs. Validation ("Did we build the right product?").
2.  **Perform** Fault Injection (e.g., cut sensor wire, freeze CPU) in simulation.
3.  **Measure** Diagnostic Coverage (DC) (Did we catch 99% of faults?).
4.  **Construct** a Safety Case Argument (GSN - Goal Structuring Notation).
5.  **Automate** a Safety Regression Test Suite.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 103:** Fault Tolerance.
-   **Day 98:** Validation Pipeline.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `unittest`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: V&V (Verification and Validation)

-   **Verification (Left side of V-Model):** Checking requirements.
    -   *Test:* Unit Test, Code Review.
    -   *Question:* "Does the code match the spec?"
-   **Validation (Right side of V-Model):** Checking the final product.
    -   *Test:* Vehicle Test, Fault Injection.
    -   *Question:* "Is the car safe to drive?"

### 🔹 Part 2: Fault Injection

To validate safety mechanisms, we must trigger the fault they protect against.
-   **Hardware Fault Injection:** Pin lifting, Short circuiting (on a test bench).
-   **Software Fault Injection:** Corrupting RAM, Delaying messages, Returning `NaN`.
-   **Simulation Fault Injection:** Disabling a sensor in CARLA.

### 🔹 Part 3: The Safety Case

A document that argues *why* the system is safe.
-   **Claim:** "The system is safe against single-point failures."
-   **Argument:** "We use a Watchdog and Dual-Core Lockstep."
-   **Evidence:** "Test Report #123 shows Watchdog catches 100% of CPU freezes."

---

## 💻 Implementation: Fault Injection Suite

**Scenario:**
-   **SUT (System Under Test):** The `CruiseControl` from Day 103.
-   **Injector:** A test script that mocks the Radar and injects faults.
-   **Goal:** Verify that the system enters `SAFE_STOP` within 1.0s of a fault.

### 🛠️ Setup
Create `week15_day104` and `test_safety.py`.

```bash
mkdir -p ~/ros2_ws/src/week15_day104
cd ~/ros2_ws/src/week15_day104
touch test_safety.py
```

### 👨‍💻 Code: Automated Fault Injection

```python
import unittest
import time
import threading

# --- Mock System (Simplified from Day 103) ---
class CruiseControl:
    def __init__(self):
        self.state = "ACTIVE"
        self.last_heartbeat = time.time()
        self.fault_detected = False
        
    def update_heartbeat(self):
        self.last_heartbeat = time.time()
        
    def run_cycle(self):
        # Safety Logic
        if time.time() - self.last_heartbeat > 0.5:
            self.fault_detected = True
            self.state = "SAFE_STOP"
        else:
            self.state = "ACTIVE"

# --- Test Suite ---
class TestSafetyMechanisms(unittest.TestCase):
    
    def setUp(self):
        self.cc = CruiseControl()
        
    def test_normal_operation(self):
        """Verify system stays ACTIVE if heartbeat is present."""
        print("\nTest: Normal Operation")
        for _ in range(5):
            self.cc.update_heartbeat()
            self.cc.run_cycle()
            time.sleep(0.1)
            self.assertEqual(self.cc.state, "ACTIVE")
            
    def test_loss_of_heartbeat(self):
        """Verify system enters SAFE_STOP on heartbeat loss."""
        print("\nTest: Fault Injection (Heartbeat Loss)")
        
        # 1. Run normal for a bit
        self.cc.update_heartbeat()
        self.cc.run_cycle()
        
        # 2. INJECT FAULT: Stop updating heartbeat
        print(">> Injecting Fault: Cutting Radar Wire...")
        time.sleep(0.6) # Wait longer than 0.5s timeout
        
        # 3. Run Cycle
        self.cc.run_cycle()
        
        # 4. Assert Safe State
        print(f"State: {self.cc.state}")
        self.assertEqual(self.cc.state, "SAFE_STOP")
        self.assertTrue(self.cc.fault_detected)

    def test_recovery_attempt(self):
        """Verify system does NOT auto-recover immediately (Latching)."""
        print("\nTest: Latching Fault")
        
        # 1. Trigger Fault
        time.sleep(0.6)
        self.cc.run_cycle()
        self.assertEqual(self.cc.state, "SAFE_STOP")
        
        # 2. Restore Heartbeat (Ghost returns)
        print(">> Removing Fault: Radar comes back...")
        self.cc.update_heartbeat()
        self.cc.run_cycle()
        
        # 3. Assert it stays in SAFE_STOP (Latching logic missing in Mock?)
        # Let's check our mock logic.
        # In `run_cycle`, `else: self.state = "ACTIVE"`.
        # Oops! Our mock auto-recovers. This test should FAIL.
        
        # Ideally, safety systems latch. Let's see if it fails.
        # If it fails, we found a bug in the design!
        if self.cc.state == "ACTIVE":
            print("FAIL: System auto-recovered! Safety violation.")
        else:
            print("PASS: System latched.")

if __name__ == "__main__":
    unittest.main()
```

---

## 🔬 Lab Exercise: The Bug Hunt

### Lab Objectives
1.  Run the test: `python3 test_safety.py`.
2.  **Observation:**
    -   `test_normal_operation`: PASS.
    -   `test_loss_of_heartbeat`: PASS.
    -   `test_recovery_attempt`: **FAIL** (or prints "FAIL").
3.  **Analysis:**
    -   The mock code `run_cycle` has a bug: `else: self.state = "ACTIVE"`.
    -   This means if the sensor flickers (Off -> On), the car goes (Stop -> Go). This is dangerous "Jerky" behavior.
4.  **Fix:**
    -   Modify `run_cycle`:
        ```python
        if self.fault_detected:
            self.state = "SAFE_STOP" # Latch forever
            return
        ```
    -   Re-run test. It should now pass the safety requirement.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Flaky Tests
**Symptom:** Test passes sometimes, fails others.
**Cause:** Using `time.sleep()` in tests relies on OS scheduling.
**Solution:** Mock the time. Use a virtual clock (`self.current_time`) that you advance manually in the test.

#### 2. Heisenbugs
**Symptom:** The bug disappears when you enable logging.
**Cause:** Logging slows down the code, masking race conditions.
**Solution:** Use lock-free logging or hardware tracing.

---

## ⚡ Optimization & Best Practices

### 1. Chaos Engineering
Netflix uses "Chaos Monkey" to kill servers.
Waymo uses "Chaos" in simulation:
-   Randomly delete cars.
-   Randomly flip bits in the map.
-   Randomly delay messages.
-   If the AV crashes, the safety case is invalid.

### 2. Code Coverage
For ASIL D, you need **100% MC/DC (Modified Condition/Decision Coverage)**.
-   Every `if (A or B)` must be tested with:
    -   A=T, B=F
    -   A=F, B=T
    -   A=F, B=F
-   This ensures every condition independently affects the outcome.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the purpose of Fault Injection?
    *   **A:** To validate that the safety mechanisms (which are rarely used) actually work when needed.
2.  **Q:** What is a "Latching Fault"?
    *   **A:** A fault that keeps the system in a safe state even if the condition clears. Requires a manual reset (Key Cycle).
3.  **Q:** What is GSN?
    *   **A:** Goal Structuring Notation. A graphical way to document the Safety Case (Claims -> Arguments -> Evidence).

### Challenge Task
**Task:** Latency Injection.
1.  Modify the test to inject a *delay* in heartbeat (0.4s) instead of total loss.
2.  The Watchdog (0.5s) should NOT trigger.
3.  If it does, your timeout is too tight (False Positive).

---

## 📚 Further Reading & References
-   [ISO 26262 Part 6: Software Testing](https://www.iso.org/standard/43464.html)
-   [Fault Injection Techniques](https://users.ece.cmu.edu/~koopman/des_s99/fault_injection/)

---

**Day 104 Complete** | Phase 4: ADAS & Robotics Systems | Week 15: Safety Standards (ISO 26262 & SOTIF)
