# Day 34: Safety & Verification (ISO 26262)
## Phase 4: ADAS & Robotics Systems | Week 5: Path Planning & Decision Making

---

> **📝 Day 34 Focus:**
> An autonomous car that crashes is useless, no matter how good its AI is. **Functional Safety** is the discipline of ensuring that systems fail safely. Today, we dive into the automotive bible: **ISO 26262**, and learn how to design systems that protect human life.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the core concept of Functional Safety: "Absence of unreasonable risk due to hazards caused by malfunctioning behavior."
2.  **Determine** the ASIL (Automotive Safety Integrity Level) of a function using HARA (Hazard Analysis and Risk Assessment).
3.  **Differentiate** between ISO 26262 (System Failures) and ISO 21448 (SOTIF - Performance Limitations).
4.  **Perform** a basic FMEA (Failure Mode and Effects Analysis) on a braking system.
5.  **Design** a Safety Monitor in Python that triggers a Safe State (Emergency Stop) upon fault detection.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Systems Engineering:** Requirements, V-Model.
-   **Probability:** Failure rates (FIT).

### Hardware Requirements
-   **None:** This is a theoretical and process-heavy day.

### Software Stack
-   **Python:** For simulation of safety logic.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: ISO 26262 Overview

ISO 26262 is the international standard for functional safety of electrical/electronic systems in production automobiles.

#### 1.1 The V-Model
-   **Left Side:** Requirements -> Architecture -> Design.
-   **Right Side:** Unit Test -> Integration Test -> System Test.
-   **Bottom:** Implementation.

#### 1.2 HARA (Hazard Analysis and Risk Assessment)
For every feature (e.g., Lane Keep Assist), we analyze potential hazards.
We score them on three dimensions:
1.  **Severity (S):** How bad is the injury? (S0: No injury -> S3: Fatal).
2.  **Exposure (E):** How often does this happen? (E0: Rare -> E4: Frequent).
3.  **Controllability (C):** Can the driver intervene? (C0: Easy -> C3: Impossible).

#### 1.3 ASIL (Automotive Safety Integrity Level)
Combination of S, E, C determines ASIL.
-   **QM (Quality Management):** Normal engineering is enough.
-   **ASIL A:** Low risk.
-   **ASIL D:** Highest risk (e.g., Airbag, Steering, Brakes). Requires redundancy and strict verification.

---

### 🔹 Part 2: SOTIF (ISO 21448)

ISO 26262 covers **Hardware/Software Failures** (Bugs, broken wires).
What if the system works *exactly as designed*, but still crashes because the camera was blinded by the sun?
This is **SOTIF (Safety Of The Intended Function)**.
-   Focuses on **Functional Insufficiencies** and **Unknown Unsafe Scenarios**.

---

### 🔹 Part 3: Safety Mechanisms

To achieve ASIL D, we need mechanisms to detect and handle faults.
1.  **Watchdog Timer:** Resets the CPU if code hangs.
2.  **CRC (Cyclic Redundancy Check):** Detects corrupted data on CAN bus.
3.  **Lockstep Cores:** Two CPUs run the same code. If outputs differ -> Fault.
4.  **Safe State:** The default fallback (e.g., Turn off motor, Apply brakes).

---

## 💻 Implementation: Safety Monitor

We will implement a `SafetyMonitor` class that supervises a `BrakeController`.
It simulates a "Heartbeat" check and a "Range Check".

### 🛠️ Setup
Create `week5_day34` and `safety_monitor.py`.

```bash
mkdir -p ~/ros2_ws/src/week5_day34
cd ~/ros2_ws/src/week5_day34
touch safety_monitor.py
```

### 👨‍💻 Code: Safety Monitor Simulation

```python
import time
import random
import threading

class BrakeController:
    def __init__(self):
        self.requested_pressure = 0.0
        self.actual_pressure = 0.0
        self.fault_injected = False
        self.alive = True

    def run(self):
        while self.alive:
            # Simulate processing loop
            time.sleep(0.1)
            
            # Simulate Actuator Physics
            if self.fault_injected:
                # Fault: Actuator stuck at 0
                self.actual_pressure = 0.0
            else:
                # Normal operation
                self.actual_pressure = self.requested_pressure + random.uniform(-0.5, 0.5)

    def set_request(self, pressure):
        self.requested_pressure = pressure

    def get_status(self):
        return {
            'requested': self.requested_pressure,
            'actual': self.actual_pressure,
            'timestamp': time.time()
        }

class SafetyMonitor:
    def __init__(self, controller):
        self.controller = controller
        self.safe_state_triggered = False
        self.tolerance = 2.0 # Max allowed deviation
        self.timeout = 0.5 # Max allowed silence

    def monitor(self):
        print("Safety Monitor Started.")
        last_heartbeat = time.time()
        
        while not self.safe_state_triggered:
            status = self.controller.get_status()
            now = time.time()
            
            # Check 1: Timing (Watchdog)
            # In a real system, the controller would send a 'heartbeat' signal.
            # Here we check timestamp freshness.
            if now - status['timestamp'] > self.timeout:
                print(f"[FAULT] Timeout! Controller hung.")
                self.trigger_safe_state()
                break
                
            # Check 2: Plausibility (Range Check)
            error = abs(status['requested'] - status['actual'])
            if error > self.tolerance:
                print(f"[FAULT] Deviation too high! Req: {status['requested']:.1f}, Act: {status['actual']:.1f}")
                self.trigger_safe_state()
                break
                
            time.sleep(0.05)

    def trigger_safe_state(self):
        print("!!! TRIGGERING SAFE STATE !!!")
        print("Action: Disabling Throttle. Applying Emergency Brake (Mechanical).")
        self.safe_state_triggered = True
        self.controller.alive = False # Kill controller

def run_simulation():
    # 1. Setup
    brake_ecu = BrakeController()
    monitor = SafetyMonitor(brake_ecu)
    
    # Start Controller Thread
    t_ecu = threading.Thread(target=brake_ecu.run)
    t_ecu.start()
    
    # Start Monitor Thread
    t_mon = threading.Thread(target=monitor.monitor)
    t_mon.start()
    
    # 2. Normal Operation
    print("--- Normal Operation ---")
    for i in range(5):
        req = random.uniform(10, 50)
        brake_ecu.set_request(req)
        print(f"Driver requests: {req:.1f}")
        time.sleep(0.5)
        if monitor.safe_state_triggered: break
        
    # 3. Inject Fault
    if not monitor.safe_state_triggered:
        print("\n--- INJECTING FAULT (Stuck Actuator) ---")
        brake_ecu.fault_injected = True
        brake_ecu.set_request(80.0) # Emergency brake request
        
        # Wait for monitor to catch it
        time.sleep(2.0)
        
    t_ecu.join()
    t_mon.join()
    print("Simulation End.")

if __name__ == "__main__":
    run_simulation()
```

---

## 🔬 Lab Exercise: FMEA

### Lab Objectives
1.  **Task:** Perform a mini-FMEA for an "Automatic Emergency Braking (AEB)" system.
2.  **Table:** Create a Markdown table with columns: `Component`, `Failure Mode`, `Effect`, `Severity`, `Detection`, `Action`.
3.  **Example Row:**
    -   **Component:** Lidar Sensor.
    -   **Failure Mode:** Covered by mud (No data).
    -   **Effect:** AEB does not trigger when needed.
    -   **Severity:** High (Collision).
    -   **Detection:** Low point count in scan.
    -   **Action:** Trigger "Sensor Blocked" warning, disable AEB, alert driver.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. False Positives
**Symptom:** Safety Monitor triggers Safe State during normal operation.
**Cause:** Tolerance too tight or noise too high.
**Solution:** Increase tolerance or use a debounce filter (Fault must persist for X ms).

#### 2. Nuisance Faults
**Symptom:** Driver ignores warnings because they happen too often.
**Cause:** Poor calibration of SOTIF limits.
**Solution:** Extensive real-world testing to tune thresholds.

---

## ⚡ Optimization & Best Practices

### 1. E-Gas Monitoring Concept
Standard architecture for Engine Control Units (ECU).
-   **Level 1:** Functional Code (Torque calculation).
-   **Level 2:** Monitoring (Redundant calculation).
-   **Level 3:** Hardware Monitoring (Watchdog).

### 2. Decomposition
If a function is ASIL D, it's expensive to build.
**ASIL Decomposition:**
-   Implement the function on two independent ASIL B units.
-   $B(D) + B(D) = D$.
-   Example: Main CPU (ASIL B) + Safety Checker Chip (ASIL B).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What does ASIL stand for?
    *   **A:** Automotive Safety Integrity Level.
2.  **Q:** What is the difference between ISO 26262 and SOTIF?
    *   **A:** ISO 26262 deals with *malfunctions* (broken parts). SOTIF deals with *limitations* (blind spots, bad weather).
3.  **Q:** What is a "Safe State"?
    *   **A:** A state where the risk is minimized (e.g., Stopped, Power Off, Limp Home Mode).

### Challenge Task
**Task:** Redundant Sensor Check.
1.  Simulate two sensors: `Speed_Wheel` and `Speed_GPS`.
2.  In the Safety Monitor, compare them.
3.  If `abs(Speed_Wheel - Speed_GPS) > 5 km/h`, trigger a fault "Speed Sensor Discrepancy".

---

## 📚 Further Reading & References
-   [ISO 26262 Wikipedia](https://en.wikipedia.org/wiki/ISO_26262)
-   [SOTIF Overview (NVIDIA)](https://developer.nvidia.com/blog/sotif-safety-of-the-intended-functionality/)

---

**Day 34 Complete** | Phase 4: ADAS & Robotics Systems | Week 5: Path Planning & Decision Making
