# Day 169: Regulation & Safety Standards
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 25: Final Integration & Graduation

---

> **📝 Content Creator Instructions:**
> Don't kill anyone. Prove it.
> - **Focus:** ISO 26262 (Functional Safety), ASIL Levels (A/B/C/D), SOTIF (ISO 21448), and Decomposition (Doer-Checker Architecture).
> - **Code:** A Python script `safety_decomposition.py`. Implement a "Main Controller" (Complex, calculates speed) and a "Safety Monitor" (Simple, checks limits). If Main > Limit, Monitor cuts power. (ASIL D decomposition).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Distinguish** between Functional Safety (ISO 26262) and SOTIF (ISO 21448).
2.  **Calculate** ASIL levels based on Severity, Exposure, and Controllability.
3.  **Implement** a Doer-Checker (Safety Bag) pattern.
4.  **Explain** the V-Model of Systems Engineering.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
# Standard Python
```

### Prior Knowledge
- Systems Engineering.
- Probability of Failure.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: ISO 26262 (Functional Safety)

Deals with **Malfunctions** (Bugs, Hardware failure).
*   **ASIL (Automotive Safety Integrity Level):**
    *   **QM:** Quality Management (Radio).
    *   **ASIL A:** Rear lights.
    *   **ASIL D:** Airbag, Steering, Braking (Most critical).
    *   **Risk:** Severity (S) x Exposure (E) x Controllability (C).

### 🔹 Part 2: SOTIF (ISO 21448)

Deals with **Functional Insufficiencies** (No bug, but System failed).
*   **Example:** Camera sees a white truck against a bright sky. Algorithm works "correctly" (no crash) but detection fails due to physics.
*   **Goal:** Reduce the "Unknown Unsafe" area.

### 🔹 Part 3: Decomposition

How to achieve ASIL D with cheap chips?
*   **Redundancy:** Use two ASIL B chips.
*   **Doer-Checker:**
    *   **Doer (Complex):** Neural Net Planner. Runs on GPU. (QM).
    *   **Checker (Simple):** Physics Validator. Runs on Safety MCU. (ASIL D).
    *   **Logic:** `if Doer.cmd > SafeLimit: E_Stop()`.

---

## 💻 Implementation: The Safety Bag

We simulate a Cruise Control system.
*   **Main:** Uses a "Complex" MPPI output (simulated).
*   **Monitor:** Checks current speed and distance.

### 🛠️ Project Structure
```text
day169_safety/
├── src/
│   ├── safety_decomposition.py
└── output/
    ├── safety_log.txt
```

### 👨‍💻 Doer-Checker (`src/safety_decomposition.py`)

```python
import time
import random

class VehicleState:
    def __init__(self):
        self.speed = 20.0 # m/s (approx 72 km/h)
        self.dist_to_lead = 50.0 
        self.max_decel = 5.0 # m/s^2 (Physical limit)

class MainController_QM:
    """
    The 'Complex' AI controller. Can have bugs.
    """
    def compute_cmd(self, state):
        # Simulate neural net noise or bug
        noise = random.uniform(-1, 1)
        
        # Bug: Occasionally outputs insane acceleration
        if random.random() < 0.1:
            print("   [QM] BUG: Determining FULL THROTTLE!")
            return 10.0 # Accel +10 m/s^2 (Unsafe!)
            
        # Normal logic: Keep distance
        error = state.dist_to_lead - 30.0
        cmd = 0.5 * error # P-controller
        return cmd

class SafetyMonitor_ASIL_D:
    """
    The 'Simple' Safety Checker. Verified Code.
    """
    def check_cmd(self, cmd, state):
        # Rule 1: Max Acceleration limit (Comfort/Traction)
        if cmd > 3.0:
            print(f"   [ASIL-D] REJECT: Accel {cmd:.1f} > Limit 3.0")
            return 0.0 # Clamp
            
        # Rule 2: TTC (Time To Collision)
        # If closing speed is high, do not allow accel
        # Simple check: If too close, force brake.
        if state.dist_to_lead < 20.0 and cmd > 0:
            print(f"   [ASIL-D] OVERRIDE: Dist {state.dist_to_lead:.1f} too small for accel.")
            return -2.0 # Force Brake
            
        return cmd # Pass through

def main():
    state = VehicleState()
    controller = MainController_QM()
    monitor = SafetyMonitor_ASIL_D()
    
    print("--- Safety Decomposition Test ---")
    
    for t in range(20):
        print(f"Time {t}: Dist={state.dist_to_lead:.1f}, Spd={state.speed:.1f}")
        
        # 1. Main Path
        raw_cmd = controller.compute_cmd(state)
        
        # 2. Safety Path
        safe_cmd = monitor.check_cmd(raw_cmd, state)
        
        if safe_cmd != raw_cmd:
            print(f"   -> Intervened! Raw: {raw_cmd:.2f} -> Safe: {safe_cmd:.2f}")
        
        # 3. Physics
        state.speed += safe_cmd * 0.1
        # Lead car moves at 20 m/s too, roughly constant dist unless we change speed
        # relative v = 20 - state.speed
        state.dist_to_lead += (20.0 - state.speed) * 0.1
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Heartbeat"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:** The Monitor catches the "+10.0" spike.
- **Modify:** What if Main freezes (stops sending)?
- **Feature:** Add a Watchdog Timer (WDT).
- **Logic:** `SafetyMonitor` expects a call every 100ms. If 200ms passes, `E_Stop()`.
- **Task:** Simulate a loop hang in `MainController` and verify Watchdog triggers.

---

## 🚀 Project: "HARA (Hazard Analysis and Risk Assessment)"

**Goal:** Create a Risk Table.
1.  **Function:** "Keep Lane".
2.  **Failure:** "Steer into opposing traffic".
3.  **S (Severity):** S3 (Life Threatening).
4.  **E (Exposure):** E4 (High probability of being on a road).
5.  **C (Controllability):** C3 (Driver cannot react in time).
6.  **ASIL:** D (Highest).
7.  **Requirement:** "Steering Actuator must have redundant windings."

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "False Positives"
*   **Cause:** Monitor is *too* conservative.
*   **Result:** Car keeps braking because Monitor thinks gap is unsafe, but Main knows lead car is accelerating.
*   **Fix:** Monitor needs State, not just Limits. But Keep It Simple!

#### 2. "Common Mode Failure"
*   **Cause:** Main and Monitor run on same CPU. CPU overheats. Both die.
*   **Fix:** Hardware Independence. Use a separate Watchdog chip (PMIC).

---

## ⚡ Optimization: Lockstep Cores

Hardware solution.
*   **CPU:** Two cores run the exact same instruction stream.
*   **Comparator:** Hardware compares output of execution units every cycle.
*   **Mismatch:** Instant Reset. (Detects cosmic ray bit flips).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** QM vs ASIL?
    *   **A:** QM = Standard Industry Practice (Infotainment). ASIL = Safety Critical (ABS).
2.  **Q:** What is the V-Model?
    *   **A:** Design down (Left side), Test up (Right side). Requirements $\to$ Arch $\to$ Code $\to$ Unit Test $\to$ Integration Test $\to$ Validation.
3.  **Q:** SOTIF example?
    *   **A:** Lidar works perfectly, but fog absorbs the beam. No bug, just limitation. Need to handle it.

### Challenge Task
> **Task:** E-Stop Logic.
> 1. Implement a 3-stage shutdown.
> 2. Stage 1: "Warn Driver".
> 3. Stage 2 (1s later): "Minimal Risk Maneuver" (Pull over).
> 4. Stage 3 (Critical): "Full Brake".

---

## 📚 Further Reading
- **ISO 26262 Part 6:** Software Development.
- **Microchip/Infineon:** Safety Manuals for microcontrollers (Aurix, Hercules).

---

**Day 169 Complete**
