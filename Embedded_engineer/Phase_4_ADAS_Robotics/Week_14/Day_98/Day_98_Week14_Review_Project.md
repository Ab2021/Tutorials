# Day 98: Week 14 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 14: Simulation (CARLA & Gazebo)

---

> **📝 Day 98 Focus:**
> We have learned to build worlds (Gazebo), drive cars (CARLA), script scenarios (OpenSCENARIO), and connect hardware (HIL). Now, we scale up. We don't run one test; we run **1000 tests**. Today's project is an **Automated Validation Pipeline**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Design** a Test Suite containing multiple OpenSCENARIO files.
2.  **Implement** a Python Test Runner that executes scenarios sequentially.
3.  **Log** metrics (Min Distance, Max Jerk, Collision Status) to a JSON report.
4.  **Analyze** the "Sim-to-Real Gap" (Why simulation isn't enough).
5.  **Generate** Corner Cases (e.g., "Cut-in while raining at night").

---

## 📚 Week 14 Review

### 1. Gazebo vs CARLA
-   **Gazebo:** Best for robotics, physics manipulation, custom sensors.
-   **CARLA:** Best for autonomous driving, large maps, realistic rendering.

### 2. Scenario Runner
-   **OpenSCENARIO:** The standard XML format for defining "Stories".
-   **Traffic Manager:** Fills the background with chaos.

### 3. HIL (Hardware-in-the-Loop)
-   **Concept:** Real ECU + Virtual World.
-   **Critical:** Real-time synchronization and low latency.

---

## 🛠️ Capstone Project: The Validation Pipeline

**Goal:** Validate an AV Stack (simple agent) against 3 scenarios.
**Scenarios:**
1.  **FollowLead:** Follow a car at safe distance.
2.  **CutIn:** React to a sudden lane change.
3.  **RedLight:** Stop at a red light.

**Output:** A `test_report.json` summarizing Pass/Fail.

### Package Structure
Create `week14_project` folder.

```bash
mkdir -p ~/ros2_ws/src/week14_project/scenarios
cd ~/ros2_ws/src/week14_project
touch test_runner.py
```

### 👨‍💻 Code: The Test Runner

```python
import os
import subprocess
import json
import time
import xml.etree.ElementTree as ET

# --- Configuration ---
SCENARIO_DIR = "./scenarios"
CARLA_ROOT = "/opt/carla-simulator"
SCENARIO_RUNNER_PATH = f"{CARLA_ROOT}/ScenarioRunner/scenario_runner.py"

class TestResult:
    def __init__(self, name):
        self.name = name
        self.status = "PENDING"
        self.duration = 0.0
        self.metrics = {} # e.g., {'min_distance': 5.0}

class ValidationPipeline:
    def __init__(self):
        self.results = []
        
    def run_scenario(self, scenario_file):
        print(f"\n>>> Running Scenario: {scenario_file} <<<")
        
        result = TestResult(scenario_file)
        start_time = time.time()
        
        # Command: python scenario_runner.py --openscenario <file> --json
        cmd = [
            "python3", SCENARIO_RUNNER_PATH,
            "--openscenario", f"{SCENARIO_DIR}/{scenario_file}",
            "--json" # Output results to JSON
        ]
        
        try:
            # Run Process
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Check Exit Code
            if process.returncode == 0:
                result.status = "PASS"
            else:
                result.status = "FAIL"
                print(f"Error Output:\n{process.stderr}")
                
        except subprocess.TimeoutExpired:
            result.status = "TIMEOUT"
            print("Scenario Timed Out!")
            
        result.duration = time.time() - start_time
        self.results.append(result)
        
        print(f"Result: {result.status} ({result.duration:.1f}s)")

    def generate_report(self):
        report = {
            "timestamp": time.time(),
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.status == "PASS"),
            "failed": sum(1 for r in self.results if r.status != "PASS"),
            "details": []
        }
        
        for r in self.results:
            report["details"].append({
                "name": r.name,
                "status": r.status,
                "duration": r.duration
            })
            
        with open("test_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        print("\n>>> Test Report Generated: test_report.json <<<")

def main():
    # 1. Discovery
    # For this demo, we assume files exist. In reality, use os.listdir()
    # You need to copy the .xosc files from Day 94 to ./scenarios/
    scenarios = ["CutIn.xosc", "FollowLead.xosc"] 
    
    pipeline = ValidationPipeline()
    
    # 2. Execution
    for s in scenarios:
        # Check if file exists (Mocking for this script if files missing)
        if not os.path.exists(f"{SCENARIO_DIR}/{s}"):
            print(f"Warning: {s} not found. Skipping.")
            continue
            
        pipeline.run_scenario(s)
        time.sleep(2) # Cooldown
        
    # 3. Reporting
    pipeline.generate_report()

if __name__ == "__main__":
    main()
```

---

## 🧪 Verification & Testing

### 1. The Happy Path
-   **Input:** Valid `CutIn.xosc`.
-   **Execution:** CARLA loads, car drives, cut-in happens, car brakes, scenario finishes.
-   **Result:** `PASS`.

### 2. The Failure
-   **Input:** A scenario where Ego is set to `speed=100` and `brake=0`.
-   **Execution:** Collision occurs.
-   **Result:** `FAIL`.
-   **Report:** `test_report.json` shows `failed: 1`.

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Simulation Basics
1.  **Q:** Why do we need "Corner Cases"?
    *   **A:** Because 99% of driving is boring. The AI fails in the 1% (weird weather, weird behavior). We must simulate these specifically.
2.  **Q:** What is "Sim-to-Real Gap"?
    *   **A:** The difference between simulation and reality (e.g., Sim friction is perfect, Real friction varies; Sim LiDAR has Gaussian noise, Real LiDAR has multipath).

### Section 2: Architecture
3.  **Q:** How does Scenario Runner know if a test passed?
    *   **A:** It checks the `<StopTrigger>`. If a `CollisionCondition` triggers, it marks Failure. If the `Story` completes without collision, it marks Success.
4.  **Q:** Can I run this in the cloud?
    *   **A:** Yes! AWS RoboMaker or Azure allow running CARLA in headless containers to run 10,000 scenarios in parallel.

---

## 🏆 Conclusion

Congratulations on completing Week 14!
-   You have mastered **Simulation**.
-   You can build worlds, script traffic, and validate your code automatically.

**Next Week:** We get serious about **Safety**. ISO 26262, ASIL levels, and how to prove your car won't kill anyone.

---

**Day 98 Complete** | Phase 4: ADAS & Robotics Systems | Week 14: Simulation (CARLA & Gazebo)
