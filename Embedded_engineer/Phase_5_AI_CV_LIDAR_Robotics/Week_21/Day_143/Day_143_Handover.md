# Day 143: Human-Robot Collaboration (Handover)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 21: Collaborative Robotics (Cobots)

---

> **📝 Content Creator Instructions:**
> Passing salt to a robot shouldn't be scary.
> - **Focus:** The Psychology and Physics of Handover, Phases (Pre-shape, Transport, Interaction, Release), Force Threshold triggering, and Gaze cues.
> - **Code:** A Python State Machine `handover_sm.py` that waits for a "Pull" force on the gripper before releasing the object.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deconstruct** the Human-Robot Handover interaction into discrete states.
2.  **Implement** a "Force-Triggered Release" (The robot feels you taking it).
3.  **Analyze** the safety risks (Dropping the object, pinching the human).
4.  **Design** visual or haptic cues to signal "I am ready to release".

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Robotiq Gripper (Ideal) or Simulation.
- F/T Sensor (Virtual).

### Software Environment
```bash
pip install numpy
```

### Prior Knowledge
- State Machines.
- Day 142 (Force Sensing).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Dance of Handover

HRC (Human-Robot Collaboration) is non-verbal.
1.  **Approach:** Robot moves object to transfer point. Decelerates smoothly (Confidence).
2.  **Signal:** Robot stops. Optional: LED turns Green. "I am ready."
3.  **Interaction:** Human grasps. Robot detects force change (Weight drops or Pull force increases).
4.  **Release:** Gripper opens.
5.  **Retract:** Robot moves away.

### 🔹 Part 2: Force Logic

*   **Load Load:** $F_z = m_{tool} + m_{object}$.
*   **Human Touch:** Disruption in $F_z$.
*   **Trigger:** If $F_{pull} > Threshold$, Open Gripper.
    *   *Warning:* If threshold is too low, it drops the object due to inertial vibration. If too high, human fights the robot.

---

## 💻 Implementation: The Polite Robot

We simulate the Finite State Machine (FSM) for a handover task.

### 🛠️ Project Structure
```text
day143_handover/
├── src/
│   ├── handover_sm.py
└── output/
    ├── handover_log.txt
```

### 👨‍💻 Handover State Machine (`src/handover_sm.py`)

```python
import time
import numpy as np

class RobotMock:
    def __init__(self):
        self.gripper_state = "CLOSED" # OPEN, CLOSED
        self.position = 0.0
        self.current_load = 0.5 # kg (Object mass)
        self.force_z = -4.9 # approx 0.5kg * 9.8
        
    def move_to(self, pos):
        self.position = pos
        print(f"Robot: Moving to {pos:.2f}...")
        
    def open_gripper(self):
        self.gripper_state = "OPEN"
        self.current_load = 0.0 # Dropped
        self.force_z = 0.0 # No load
        print("Robot: Gripper OPEN.")
        
    def read_force(self):
        # Simulate noisy sensor
        noise = np.random.normal(0, 0.1)
        return self.force_z + noise

class HandoverSM:
    def __init__(self, robot):
        self.robot = robot
        self.state = "IDLE"
        self.threshold = 2.0 # N (Pull force required)
        self.baseline_force = 0.0
        
    def step(self):
        f = self.robot.read_force()
        
        if self.state == "IDLE":
            # Start Process
            self.state = "APPROACH"
            
        elif self.state == "APPROACH":
            self.robot.move_to(1.0) # Transfer point
            self.state = "WAIT_FOR_GRASP"
            # Calibrate baseline load at rest
            self.baseline_force = np.mean([self.robot.read_force() for _ in range(10)])
            print(f"Robot: Baseline Load {self.baseline_force:.2f} N. Waiting for human...")
            
        elif self.state == "WAIT_FOR_GRASP":
            # Check for Pull
            # If Human pulls UP, Force Z becomes LESS negative (or positive)
            # Or Human pulls away.
            # Let's assume Human Pulls UP lifting the object weight off the robot.
            # Delta = Current - Baseline
            
            delta = f - self.baseline_force
            # If delta is Positive (Load lightening), human is lifting.
            
            if delta > self.threshold:
                print(f"Robot: Pull Detected (Delta={delta:.2f}N). Releasing...")
                self.robot.open_gripper()
                self.state = "RETRACT"
            else:
                # print(f"Wait... Delta={delta:.2f}")
                time.sleep(0.1)
                
        elif self.state == "RETRACT":
            self.robot.move_to(0.0)
            self.state = "DONE"
            
        elif self.state == "DONE":
            pass

def main():
    robot = RobotMock()
    sm = HandoverSM(robot)
    
    # Simulation Loop
    for t in range(20):
        print(f"Time {t}: State={sm.state}")
        sm.step()
        
        # Scenario: At t=10, Human grabs object and lifts (Applies 3N upward force)
        if t == 10:
            print(">>> EVENT: Human lifts object! <<<")
            # Human takes 0.3kg of the weight
            robot.force_z += 3.0 
            
        time.sleep(0.2)

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The False Release"

### 1. Lab Objectives
- **Run:** Sim. Works.
- **Fail:** Run again. At `t=5`, inject a "Bump" (Robot stops quickly, inertial jerk).
- **Modify:** `robot.force_z += 2.5` (Jerk).
- **Result:** Robot releases purely due to inertia. Object falls on floor.
- **Fix:** Add a timer. "Force must be > Threshold for 0.5 seconds continuously." (Debouncing).

$$ F_{triggerValid} \iff F > F_{thresh} \quad \forall t \in [t_0, t_0 + 0.5] $$

---

## 🚀 Project: "Multimodal Handover"

**Goal:** Use Vision + Force.
1.  **Vision:** Camera detects Human Hand approach (Bounding Box).
2.  **Safety:** If Hand velocity > 1m/s (Aggressive), Robot retracts (Scared).
3.  **Ready:** If Hand is static near object (> 2s), Robot loosens gripper slightly (compliance).
4.  **Confirm:** Force Pull triggers complete open.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Robot holds too long"
*   **Cause:** Threshold too high. Human has to yank the arm.
*   **Fix:** Adaptive Threshold. $10\%$ of Payload Weight.

#### 2. "Premature Open"
*   **Cause:** Sensor drift or vibration.
*   **Fix:** Zero the sensor immediately after reaching the Transfer Point (Tare).

---

## ⚡ Optimization: Minimum Jerk Trajectories

For Approach (State 2), use Minimum Jerk profile ($t^5$ polynomial).
*   **Why?** Human brains predict biological motion.
*   **Benefit:** Jerky moves scare humans. Smooth moves encourage interaction.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not just use voice? "Robot, give me the wrench."
    *   **A:** Latency and Noise. Factories are loud. Force is instant and unambiguous.
2.  **Q:** Safety hazard of Gripper?
    *   **A:** Pinch points. If the robot closes the gripper while you are waiting, it hurts. Use torque-limited closing.
3.  **Q:** Where should the robot look?
    *   **A:** At the object. Humans use "Joint Attention". If the robot looks at the object, the human knows to look there too.

### Challenge Task
> **Task:** Bottle Handover.
> 1. Robot holds a water bottle (Fluid sloshing changes dynamics!).
> 2. Filter the force signal to ignore sloshing freq (~2-5Hz).
> 3. Detect steady pull.

---

## 📚 Further Reading
- **Dragan, A. et al:** "Legibility and Predictability of Robot Motion".
- **HRI Conferences:** Papers on "Fluency" in collaboration.

---

**Day 143 Complete**
