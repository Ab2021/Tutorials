# Day 174: Capstone: Simulation & Testing
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 25: Final Integration & Graduation

---

> **📝 Content Creator Instructions:**
> If it isn't tested, it doesn't work.
> - **Focus:** Simulation-based Verification. SIL (Software-in-the-Loop). Creating Test Scenarios (Cut-in, Pedestrian crossing). Metric evaluation (Jerk, Distance to obstacle).
> - **Code:** `capstone_sim.py`. A Scenario Engine. It loads a test case (e.g., "Stop Sign"), runs the robot logic in the loop, and asserts whether the robot stopped within the legal distance.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Construct** complex driving scenarios using a Scripted Scenario Engine.
2.  **Define** Pass/Fail criteria for autonomous behaviors (ISO standards).
3.  **Execute** a Regression Test Suite to ensure new code doesn't break old features.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None. (CARLA/Gazebo typically used, we simulate the physics engine in Python).

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Unit Testing vs System Testing.
- Kinematics.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The X-in-the-Loop

*   **MIL (Model):** Matlab/Simulink logic.
*   **SIL (Software):** The actual C++/Python code running on PC.
*   **HIL (Hardware):** Except the Car is fake, the ECU is real.
*   **VIL (Vehicle):** Real car on a test track.

### 🔹 Part 2: Scenario Based Testing

Driving 1 million miles is inefficient (mostly boring highway).
*   **Idea:** Focus on **Critical Scenarios**.
*   **Ex:** "Left Turn at Intersection with oncoming traffic".
*   **OpenSCENARIO:** XML standard for defining these events.

### 🔹 Part 3: KPIs (Key Performance Indicators)

How do you grade a robot?
1.  **Safety:** Min Distance to obstacle > 0.5m.
2.  **Comfort:** Max Jerk < $2m/s^3$.
3.  **Legality:** Stop Line violation < 0.0m.
4.  **Efficiency:** Time to destination < Limit.

---

## 💻 Implementation: The Scenario Runner

We build a mini-simulator that runs the Robot Code against a scripted world.

### 🛠️ Project Structure
```text
day174_sim/
├── src/
│   ├── capstone_sim.py
│   └── scenarios/
│       ├── stop_sign.json
│       └── cut_in.json
└── output/
    ├── test_report.txt
```

### 👨‍💻 Scenario Engine (`src/capstone_sim.py`)

```python
import numpy as np
import json
import matplotlib.pyplot as plt

# --- ROBOT STACK (Simplified) ---
class RobotStack:
    def __init__(self):
        self.pos = 0.0
        self.vel = 10.0
        self.state = "CRUISE"
        
    def plan_and_control(self, dt, perception_data):
        # Perception Data: {'stop_sign_dist': 50.0} or {'obs_dist': 20.0}
        
        cmd_accel = 0.0
        
        # 1. Stop Sign Logic
        sign_dist = perception_data.get('stop_sign_dist')
        if sign_dist is not None:
             # Simple P-control to stop at 0
             target_dist = 5.0 # Stop 5m before sign (virtual line)
             error = sign_dist - target_dist
             
             if error < 0: # We passed it?
                 cmd_accel = -2.0 # Brake hard
             elif error < 50.0:
                 # Smooth braking v^2 = 2ax -> a = v^2 / 2d
                 req_accel = -(self.vel**2) / (2 * error + 0.1)
                 cmd_accel = np.clip(req_accel, -3.0, 1.0)
                 
        # 2. Obstacle Logic
        obs_dist = perception_data.get('obs_dist')
        if obs_dist is not None:
            if obs_dist < 20.0:
                cmd_accel = -4.0 # AEB
                
        return cmd_accel

# --- SIMULATOR ---
class Simulator:
    def __init__(self, scenario_file):
        with open(scenario_file, 'r') as f:
            self.config = json.load(f)
        
        self.robot = RobotStack()
        self.time = 0.0
        self.dt = 0.1
        self.logs = {'t': [], 'pos': [], 'vel': [], 'acc': []}
        
    def run(self):
        print(f"--- Running Scenario: {self.config['name']} ---")
        duration = self.config['duration']
        
        # Scenario Objects
        stop_sign_pos = self.config.get('stop_sign_pos') # e.g. 100m
        obs_spawn_time = self.config.get('obs_spawn_time', 999)
        
        passed = True
        comments = []
        
        for i in range(int(duration / self.dt)):
            # 1. Generate Sensor Data
            sensors = {}
            if stop_sign_pos:
                dist = stop_sign_pos - self.robot.pos
                if dist > -10 and dist < 150: # Sensor Range
                    sensors['stop_sign_dist'] = dist
                    
            if self.time > obs_spawn_time:
                 # Cut-in at 20m ahead relative
                 sensors['obs_dist'] = 15.0 # Sudden appearance
            
            # 2. Run Robot
            acc = self.robot.plan_and_control(self.dt, sensors)
            
            # 3. Physics Update
            self.robot.vel += acc * self.dt
            if self.robot.vel < 0: self.robot.vel = 0 # No reverse
            self.robot.pos += self.robot.vel * self.dt
            
            # 4. Log
            self.logs['t'].append(self.time)
            self.logs['pos'].append(self.robot.pos)
            self.logs['vel'].append(self.robot.vel)
            self.logs['acc'].append(acc)
            
            self.time += self.dt
            
        # 5. Evaluate Criteria
        print("Evaluating Criteria...")
        
        # Crit 1: Did we stop for Stop Sign?
        if stop_sign_pos:
            final_dx = stop_sign_pos - self.robot.pos
            # Allow stopping 0 to 10m before
            if 0.0 <= final_dx <= 10.0 and self.robot.vel < 0.1:
                comments.append("Stopped correctly at sign.")
            else:
                 passed = False
                 comments.append(f"FAIL: Stop Sign Logic. Final dx={final_dx:.2f}, Vel={self.robot.vel:.1f}")

        # Crit 2: Collision?
        # (Simplified)
        
        result = "PASS" if passed else "FAIL"
        print(f"Result: {result}")
        for c in comments: print(f"  - {c}")
        
        return self.logs

def main():
    # 1. Create Scenarios on the fly (Mocking file read)
    scen1 = {
        "name": "Stop Sign Test",
        "duration": 20.0,
        "stop_sign_pos": 100.0
    }
    with open('src/scenarios/stop_sign.json', 'w') as f:
        json.dump(scen1, f)
        
    # 2. Run Sim
    sim = Simulator('src/scenarios/stop_sign.json')
    logs = sim.run()
    
    # 3. Plot
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(logs['t'], logs['pos'])
    plt.ylabel('Pos (m)')
    plt.grid()
    
    plt.subplot(3,1,2)
    plt.plot(logs['t'], logs['vel'])
    plt.ylabel('Vel (m/s)')
    plt.grid()
    
    plt.subplot(3,1,3)
    plt.plot(logs['t'], logs['acc'])
    plt.ylabel('Cmd Acc (m/s^2)')
    plt.xlabel('Time (s)')
    plt.grid()
    
    plt.savefig('output/test_plot.png')

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The AEB Test"

### 1. Lab Objectives
- **Run:** Stop Sign Test.
- **Result:** Passes (Vel goes to 0 near Pos 100).
- **Create:** `cut_in.json`.
- **Config:** `obs_spawn_time = 2.0`, `stop_sign_pos` = None.
- **Run:** Verify AEB triggers (Acc = -4.0).
- **Challenge:** Tune AEB gains. If Acc < -4.0, fail comfort test? No, safety overrides comfort.

---

## 🚀 Project: "Continuous Integration (CI)"

**Goal:** GitHub Actions.
1.  **Commit:** Student pushes code.
2.  **Server:** Builds Docker Container.
3.  **Run:** Executes `run_tests.py` (The Simulator).
4.  **Bad:** If any test fails, Reject Commit.
5.  **Good:** If all pass, Deploy to Cloud Fleet.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Physics Mismatch"
*   **Cause:** Simulator assumes perfect brakes. Real car slides.
*   **Fix:** Add "Tire Model" or at least a latency (0.2s) + slew rate limit to `vel` update in simulator.

#### 2. "Zenos Paradox"
*   **Cause:** P-Controller never actually reaches 0 velocity error. Stays at 0.001 m/s.
*   **Fix:** "Standstill Manager". If v < 0.1 and brake > 0, set v = 0 hard.

---

## ⚡ Optimization: Parallel Testing

Running 1000 scenarios takes hours.
*   **Cloud:** Spin up 100 EC2 instances.
*   **Sharding:** Instance 1 runs tests 1-10, Instance 2 runs 11-20.
*   **Result:** 10 mins turn-around time.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is OpenSCENARIO?
    *   **A:** An XML standard to describe dynamic events (Actor A moves here, then Actor B cuts in).
2.  **Q:** Unit Test vs Regression Test?
    *   **A:** Unit = Test one function. Regression = Test *everything* to make sure new changes didn't break old stuff.
3.  **Q:** Why is HIL important?
    *   **A:** Because software timing depends on real hardware clocks/interrupts.

### Challenge Task
> **Task:** Monte Carlo Simulation.
> 1. Take "Stop Sign" Scenario.
> 2. Vary friction $\mu \in [0.4, 1.0]$.
> 3. Vary Sensor Noise $\sigma \in [0.1, 1.0]$.
> 4. Run 1000 times.
> 5. Plot "Probability of Collision".

---

## 📚 Further Reading
- **CARLA / LGSVL:** Open Source Simulators.
- **OpenSCENARIO 1.0:** Specification.

---

**Day 174 Complete**
