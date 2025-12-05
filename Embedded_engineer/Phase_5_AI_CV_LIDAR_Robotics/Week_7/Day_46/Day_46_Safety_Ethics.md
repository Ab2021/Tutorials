# Day 46: Safety & Ethics (ISO 13482)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 7: Human-Robot Interaction (HRI)

---

> **📝 Content Creator Instructions:**
> A robot that hurts people is a brick.
> - **Focus:** Functional Safety (ISO 13482), Risk Assessment, and Asimov's Laws in Real Code.
> - **Code:** Integrating a certified Safety Monitor (Watchdog) that overrides AI decisions.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Interpret** ISO 13482 (Personal Care Robots) and ISO 26262 (Automotive).
2.  **Conduct** a Hazard and Operability Study (HAZOP) for a mobile robot.
3.  **Implement** a Safety Bubble Monitor that runs independently of the main navigation stack.
4.  **Discuss** Algorithmic Bias in Perception (e.g., detecting different skin tones) and Ethics.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- E-Stop Button (Simulated).

### Software Environment
```bash
# None specific.
```

### Prior Knowledge
- Fault Trees.
- Finite State Machines.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: ISO 13482 Standards

Safety is not "It works 99% of the time." Safety is "When it fails, it fails safely."
**ISO 13482 Categories:**
1.  **Mobile Servant Robots:** (Butlers, Delivery).
2.  **Physical Assistant Robots:** (Exoskeletons).
3.  **Person Carrier Robots:** (Wheelchairs).
**Key Requirement:** Performance Level d (PLd). System must detect faults and stop safely even if one component fails (Redundancy).

### 🔹 Part 2: The Safety Monitor Pattern

AI (Neural Net) is a "Black Box". We cannot certify it.
**Solution:** The "Doer-Checker" Architecture.
*   **Doer (AI):** "I want to drive at 5m/s." (Complex, Unpredictable).
*   **Checker (Safety Monitor):** "Is 5m/s safe given the Lidar readings? No. Clamp to 1m/s." (Simple, Deterministic, 100 lines of Code).
*   **Rule:** Checker > Doer.

### 🔹 Part 3: Algorithmic Ethics

*   **Trolley Problem:** Rare in robotics.
*   **Real Problem:** Bias.
    *   Does the pedestrian detector work equally well for children? Wheelchairs? Dark skin tones?
    *   If trained on "Adults Walking", it might crash into a "Child Crawling".
    *   **Mitigation:** Diverse Datasets and "Edge Case" testing.

---

## 💻 Implementation: The Safety Shield

A separate ROS node that subscribes to `cmd_vel` (from Nav) and `scan` (Lidar).
If `cmd_vel` is unsafe, it publishes `cmd_vel_safe`.

### 🛠️ Project Structure
```text
day46_safety/
├── src/
│   ├── unsafe_planner.py
│   └── safety_monitor.py
└── run_simulator.py
```

### 👨‍💻 Safety Monitor (`src/safety_monitor.py`)

```python
import numpy as np

class SafetyShield:
    def __init__(self):
        self.max_speed = 1.0
        self.stop_dist = 0.5 # Meters
        self.slow_dist = 1.0
        
    def filter_command(self, cmd_v, cmd_w, scan_ranges, angle_min, angle_increment):
        # 1. Check Forward Collision Capability
        # Only check sensors in direction of motion
        min_front_dist = float('inf')
        
        for i, r in enumerate(scan_ranges):
            if r == float('inf') or r == 0.0: continue
            
            # Calculate angle of this ray
            angle = angle_min + i * angle_increment
            
            # If ray is in front (-45 to +45 deg)
            if -0.78 < angle < 0.78:
                if r < min_front_dist:
                    min_front_dist = r
        
        # 2. Safety Logic (Determinism)
        safe_v = cmd_v
        
        if min_front_dist < self.stop_dist:
            print("🚨 E-STOP TRIGGERED! Obstacle too close.")
            safe_v = 0.0 # Hard Stop
            
        elif min_front_dist < self.slow_dist:
            print(f"⚠️ Slowing down. Dist: {min_front_dist:.2f}")
            # Linear scaling: 0.5m -> 0.0, 1.0m -> max_speed
            ratio = (min_front_dist - self.stop_dist) / (self.slow_dist - self.stop_dist)
            safe_v = min(cmd_v, self.max_speed * ratio)
            
        return safe_v, cmd_w
```

### 👨‍💻 Unsafe Logic (`src/unsafe_planner.py`)

A "Buggy" AI that tries to crash.

```python
import time

class CrazyAI:
    def get_command(self):
        # Always drive full speed forward
        return 2.0, 0.0
```

### 👨‍💻 Simulation (`run_simulator.py`)

```python
from src.safety_monitor import SafetyShield
from src.unsafe_planner import CrazyAI

shield = SafetyShield()
ai = CrazyAI()

# Simulate Lidar finding an object getting closer
distances = [2.0, 1.5, 1.2, 0.9, 0.6, 0.4, 0.2]

print("--- Start Safety Test ---")
for d in distances:
    # Fake Lidar Scan (Front ray = d)
    scan = [float('inf')]*100
    scan[50] = d # Center ray
    
    raw_v, raw_w = ai.get_command()
    safe_v, safe_w = shield.filter_command(raw_v, raw_w, scan, -1.0, 0.02)
    
    print(f"Dist: {d:.2f}m | AI Req: {raw_v:.1f} m/s | Shield Output: {safe_v:.1f} m/s")
    
    if safe_v == 0.0:
        print("Test Passed: Robot Stopped.")
```

### 3. Expected Output
*   Dist 2.0: Output 2.0 (Actually clamped to max_speed 1.0 if implemented? Code allows pass-through if safe).
*   Dist 0.9: Output < 1.0 (Slowing).
*   Dist 0.4: Output 0.0 (Stopped). AI still requested 2.0.

---

## 🔬 Lab Exercise: The Heartbeat

### 1. Lab Objectives
- Implement a "Dead Man's Switch".
- **Rule:** The AI must send a heartbeat every 100ms.
- **Scenario:** The AI process crashes (Segfault).
- **Monitor:** Detects missing heartbeat > 200ms.
- **Action:** Publish `cmd_vel = 0` immediately.
- **Why?** In ROS, if a publisher dies, the last message might persist or the subscriber might wait forever. We need active Stopping.

---

## 🚀 Project: "Safety Bubble Certification"

**Goal:** Define and Visualize the ISO Safety Zones.
1.  **Zone A (Red):** $T_{stop} \times V_{cur}$. Any object here = E-Stop.
2.  **Zone B (Yellow):** Warning / Slow down.
3.  **Dynamic Reconfiguration:** As velocity increases, the Red Zone must expand (Stopping distance $d = v^2 / 2\mu g$).
4.  **Visualize:** Draw these polygons in Rviz.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Phantom Stops"
*   **Symptom:** Robot stops in empty space.
*   **Cause:** Lidar noise (sunlight/dust) detected as obstacle.
*   **Fix:** **Temporal Filtering**. Only stop if obstacle persists for 3 frames. (Trade-off: latency increase).

#### 2. "Oscillation"
*   **Symptom:** Stop -> Object "far" -> Go -> Object "close" -> Stop.
*   **Fix:** **Hysteresis**. Stop at 0.5m. Resume only when clear > 0.6m.

---

## ⚡ Optimization: Hardware Safety

Code is never 100% safe (Operating System bugs).
*   **Safety PLC:** A dedicated micro-controller (SIL 3 rated) reading the E-Stop and Lidar directly.
*   It cuts power to the motors via a relay.
*   The PC (Main Robot) is completely bypassed.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is SIL?
    *   **A:** Safety Integrity Level (1 to 4). Probability of dangerous failure per hour. Autonomous cars need ASIL D (Highest).
2.  **Q:** Why separate Safety from AI?
    *   **A:** Certifiability. AI is probablistic/nondeterministic. Safety logic must be formally verifiable.
3.  **Q:** Is "Accuracy" a safety metric?
    *   **A:** No. "Recall" (Not missing an obstacle) is critical. "Precision" (Not having false alarms) is for usability.

### Challenge Task
> **Task:** Cliff Detection.
> 1. Safety isn't just obstacles. It's also Drop-offs (Stairs).
> 2. Implement logic using Depth Camera/Downward Lidar.
> 3. If floor is missing > 1m ahead, Stop.

---

## 📚 Further Reading
- **ISO 13482:** Robots and Robotic Devices.
- **Coded Bias:** Documentary on AI Bias.

---

**Day 46 Complete**
