# Day 158: Vehicle-in-the-Loop (VIL)
## Phase 4: ADAS & Robotics Systems | Week 23: Testing & Validation

---

> **📝 Day 158 Focus:**
> HIL is great, but it doesn't simulate G-forces, tire slip, or the smell of burning rubber. **Vehicle-in-the-Loop (VIL)** puts the real car on a real track (Proving Ground), but feeds it **Virtual Obstacles**. It's the "Holodeck" for cars.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** VIL and its advantages over HIL and Public Road Testing.
2.  **Design** a Proving Ground Test Plan (Cone layout, Safety zones).
3.  **Explain** how to inject virtual objects into a real car's sensor stream.
4.  **Implement** a Safety Driver Protocol.
5.  **Analyze** VIL data for Vehicle Dynamics validation.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Vehicle Dynamics:** Tire models, Suspension.
-   **Safety:** ISO 26262 Functional Safety.

### Hardware Requirements
-   **None:** Conceptual / Simulation based. (Real VIL requires a test track).

### Software Stack
-   **ROS 2:** Visualization (Rviz) to see the "Virtual" world.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is VIL?

-   **The Setup:** A real car drives on a large empty asphalt pad.
-   **The Trick:** The car's computer thinks it's in downtown Tokyo.
-   **Sensors:**
    -   **Real:** IMU, Wheel Speed, GPS (Vehicle Dynamics).
    -   **Virtual:** Camera, Lidar, Radar (injected from a simulator running on the back seat).
-   **Benefit:** You can test "Emergency Braking for a Pedestrian" without risking a real pedestrian.

### 🔹 Part 2: The Proving Ground

-   **Controlled Environment:** No public traffic. Flat surface.
-   **Safety Driver:** Always behind the wheel, hands hovering, ready to override.
-   **Abort Criteria:** "If lateral error > 2m, Abort." "If GPS accuracy < 10cm, Abort."

### 🔹 Part 3: Augmented Reality for Cars

-   The car "sees" a virtual car cutting in front.
-   The car brakes hard (Real physics).
-   The Safety Driver feels the deceleration.
-   The Virtual Car drives away.

---

## 💻 Implementation: VIL Test Planner

**Scenario:**
-   Design a test where the car must overtake a virtual slow vehicle.
-   We need to define the **Trigger Points** based on GPS.

### 🛠️ Setup
Create `week23_day158` and `vil_planner.py`.

```bash
mkdir -p ~/ros2_ws/src/week23_day158
cd ~/ros2_ws/src/week23_day158
touch vil_planner.py
```

### 👨‍💻 Code: Virtual Object Injection

```python
import numpy as np
import matplotlib.pyplot as plt

class VIL_Scenario:
    def __init__(self):
        # Track: 500m straight line
        self.track_length = 500.0
        
        # Virtual Object: Slow Car
        self.obj_start_s = 200.0
        self.obj_speed = 10.0 # m/s
        self.obj_s = self.obj_start_s
        self.triggered = False
        
    def update(self, ego_s, dt):
        # Trigger Logic
        # Start moving the virtual object when Ego is 50m away
        if not self.triggered and (self.obj_s - ego_s) < 50.0:
            self.triggered = True
            print(f"TRIGGER: Virtual Car Activated at Ego S={ego_s:.1f}")
            
        if self.triggered:
            self.obj_s += self.obj_speed * dt
            
        return self.obj_s

def main():
    scenario = VIL_Scenario()
    
    # Ego Vehicle Simulation (Simple)
    ego_s = 0.0
    ego_v = 20.0 # 20 m/s (Approaching fast)
    dt = 0.1
    
    history_ego = []
    history_obj = []
    time = []
    
    print("Starting VIL Test Run...")
    for t in np.arange(0, 20, dt):
        # 1. Update Ego
        ego_s += ego_v * dt
        
        # 2. Update Scenario (Virtual World)
        obj_s = scenario.update(ego_s, dt)
        
        # 3. Check Collision (Virtual)
        dist = obj_s - ego_s
        if dist < 0:
            print("CRASH: Ego hit Virtual Object!")
            break
            
        # 4. Ego Reaction (Mock ACC)
        if dist < 30:
            ego_v = max(ego_v - 5.0 * dt, 0) # Brake hard
            
        history_ego.append(ego_s)
        history_obj.append(obj_s)
        time.append(t)
        
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(time, history_ego, label='Real Ego Car')
    plt.plot(time, history_obj, 'r--', label='Virtual Target Car')
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("VIL Scenario: Virtual Overtake/Brake")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: Test Plan Design

### Lab Objectives
1.  **Draw the Track:**
    -   Sketch a 100m x 50m area.
    -   Mark the "Start Zone", "Test Zone", and "Run-off Zone".
2.  **Define Safety Constraints:**
    -   Max Speed: 40 km/h.
    -   Max Lat Accel: 0.3g.
    -   Safety Driver: Must have foot on brake.
3.  **Scenario:** "Cut-in".
    -   Virtual car merges from left at $T=5s$.
    -   Ego must brake.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. GPS Drift
**Symptom:** The virtual object jumps around.
**Cause:** Standard GPS has 2-5m error.
**Solution:** **RTK-GPS** (Real-Time Kinematic) is mandatory for VIL. Accuracy < 2cm.

#### 2. Latency
**Symptom:** Virtual object appears "late" on the screen.
**Cause:** Simulator processing delay.
**Solution:** **Prediction**. The simulator must predict where the Ego will be in 100ms and render the frame for *that* time.

---

## ⚡ Optimization & Best Practices

### 1. Mixed Reality
Use AR Goggles (HoloLens) or a Dashboard Screen for the Safety Driver.
-   The driver needs to see the virtual object too, so they don't panic when the car brakes for "nothing".

### 2. Robot Targets
For physical contact testing (e.g., bumper tap).
-   Use **Soft Targets** (Balloon cars) mounted on flat robotic platforms (e.g., AB Dynamics).
-   If the AV fails and hits it, the car is undamaged.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is VIL better than HIL?
    *   **A:** It validates the real vehicle dynamics (suspension, tires, brake fade) which are hard to model perfectly in HIL.
2.  **Q:** Why is VIL better than Public Roads?
    *   **A:** Repeatability. You can run the exact same "Cut-in" scenario 100 times to statistically prove safety. You can't ask a stranger on the highway to cut you off 100 times.
3.  **Q:** What is the role of the Safety Driver?
    *   **A:** To monitor the system and take control immediately if the AV behaves unpredictably.

### Challenge Task
**Task:** Latency Compensation.
1.  In `vil_planner.py`, add a delay to the Ego position update.
2.  Implement a Kalman Filter to predict `ego_s` at `t + delay`.
3.  Use the predicted position to update the scenario.

---

## 📚 Further Reading & References
-   [AVL Vehicle-in-the-Loop](https://www.avl.com/en/testing-solutions/vehicle-testing/vehicle-in-the-loop)
-   [NHTSA Test Track Procedures](https://www.nhtsa.gov/)

---

**Day 158 Complete** | Phase 4: ADAS & Robotics Systems | Week 23: Testing & Validation
