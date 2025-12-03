# Day 161: Week 23 Review & Project (Test Automation)
## Phase 4: ADAS & Robotics Systems | Week 23: Testing & Validation

---

> **📝 Day 161 Focus:**
> We have climbed the V-Model. From Math (MIL) to Code (SIL) to Hardware (HIL) to Track (VIL) to Scenarios. Today, we build the **Quality Gate**. A unified automation framework that enforces quality at every step.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Architect** a Continuous Testing Pipeline integrating MIL, SIL, and HIL.
2.  **Generate** a Unified Quality Report (Coverage, Pass Rate, KPIs).
3.  **Implement** a "Quality Gate" script that blocks release if criteria aren't met.
4.  **Visualize** the V-Model traceability (Req $\leftrightarrow$ Test).
5.  **Critique** the cost vs. benefit of different testing stages.

---

## 📚 Week 23 Review

### 1. The V-Model
-   **MIL:** Fast, Math-based. Tuning logic.
-   **SIL:** Compiled code. Implementation bugs.
-   **HIL:** Real ECU. Timing/Hardware faults.
-   **VIL:** Real Vehicle. Dynamics/Reality.

### 2. Scenarios
-   **OpenSCENARIO:** Standard format for defining traffic situations.
-   **Regression:** Running everything to prevent backsliding.

---

## 🛠️ Capstone Project: The Quality Gate

**Goal:** Create a master script `quality_gate.py` that runs the full pipeline for a specific module (e.g., AEB).
**Criteria:**
-   MIL: Overshoot < 5%.
-   SIL: 100% Pass.
-   Coverage: > 90%.
-   Scenario: No Collisions.

### Package Structure
Create `week23_project` package.

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python week23_project
mkdir -p week23_project/reports
```

### 👨‍💻 Code: The Quality Gate Script

```python
import subprocess
import sys
import json
import os

class QualityGate:
    def __init__(self):
        self.results = {
            "MIL": "PENDING",
            "SIL": "PENDING",
            "COVERAGE": 0.0,
            "SCENARIO": "PENDING"
        }
        
    def run_mil(self):
        print("--- STAGE 1: MIL (Model-in-the-Loop) ---")
        # Mocking the MIL run from Day 155
        try:
            # subprocess.check_call(["python3", "src/week23_day155/mil_acc.py"])
            print("MIL Simulation: OK")
            print("Requirement Check: Overshoot < 2m: PASS")
            self.results["MIL"] = "PASS"
            return True
        except Exception as e:
            self.results["MIL"] = "FAIL"
            return False

    def run_sil(self):
        print("--- STAGE 2: SIL (Software-in-the-Loop) ---")
        try:
            # subprocess.check_call(["colcon", "test", "--packages-select", "week23_day156"])
            print("Unit Tests: 5/5 PASS")
            self.results["SIL"] = "PASS"
            return True
        except:
            self.results["SIL"] = "FAIL"
            return False

    def check_coverage(self):
        print("--- STAGE 3: Code Coverage ---")
        # Mocking lcov result
        coverage = 92.5
        print(f"Line Coverage: {coverage}%")
        self.results["COVERAGE"] = coverage
        if coverage > 90.0:
            return True
        return False

    def run_scenario(self):
        print("--- STAGE 4: Scenario Test (Cut-In) ---")
        try:
            # subprocess.check_call(["python3", "scenario_runner.py", ...])
            print("Scenario 'Cut-In': PASS")
            self.results["SCENARIO"] = "PASS"
            return True
        except:
            self.results["SCENARIO"] = "FAIL"
            return False

    def generate_report(self):
        print("\n=== QUALITY GATE REPORT ===")
        print(json.dumps(self.results, indent=4))
        
        with open("week23_project/reports/quality_report.json", "w") as f:
            json.dump(self.results, f, indent=4)
            
        if all(v == "PASS" for k, v in self.results.items() if k != "COVERAGE") and self.results["COVERAGE"] > 90:
            print("RESULT: 🟢 RELEASE APPROVED")
            return 0
        else:
            print("RESULT: 🔴 RELEASE BLOCKED")
            return 1

def main():
    gate = QualityGate()
    
    if not gate.run_mil():
        sys.exit(gate.generate_report())
        
    if not gate.run_sil():
        sys.exit(gate.generate_report())
        
    if not gate.check_coverage():
        print("WARNING: Coverage too low!")
        # sys.exit(gate.generate_report()) # Strict mode
        
    if not gate.run_scenario():
        sys.exit(gate.generate_report())
        
    sys.exit(gate.generate_report())

if __name__ == "__main__":
    main()
```

---

## 🧪 Verification & Testing

### 1. The Happy Path
-   Run `python3 quality_gate.py`.
-   **Output:** All stages PASS. "RELEASE APPROVED".

### 2. The Failure
-   Simulate a MIL failure (return False).
-   **Output:** Script stops at Stage 1. "RELEASE BLOCKED".
-   **Lesson:** Fail Fast. Don't waste time running slow scenarios if the math is wrong.

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Testing Strategy
1.  **Q:** Which test stage is the most expensive?
    *   **A:** VIL (Vehicle-in-the-Loop). Requires track time, fuel, safety driver, and prototype vehicle.
2.  **Q:** Which test stage finds the most bugs?
    *   **A:** Ideally MIL/SIL (Unit Tests). If you find most bugs in VIL, your process is broken.

### Section 2: Automation
3.  **Q:** Why do we need a "Quality Gate"?
    *   **A:** To prevent human error. "I thought I ran the tests" is not an acceptable excuse in safety-critical engineering.

---

## 🏆 Conclusion

You have completed Week 23.
You are now a **Validation Engineer**. You know how to prove that a robot is safe.
Next week, we start the **Capstone Project: Autonomous Valet Parking**. We will use everything (Perception, Planning, Control, Testing) to build a complete system.

---

**Day 161 Complete** | Phase 4: ADAS & Robotics Systems | Week 23: Testing & Validation
