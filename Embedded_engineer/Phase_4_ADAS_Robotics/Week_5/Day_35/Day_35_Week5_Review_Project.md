# Day 35: Week 5 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 5: Path Planning & Decision Making

---

> **📝 Day 35 Focus:**
> We have built the Brain. It can find a path, dodge obstacles, decide when to pass, and execute the maneuver smoothly and safely. Today, we put it all together in the **Highway Autopilot** project, simulating a Level 3 autonomous system.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Integrate** Global, Behavior, and Local planners into a hierarchical stack.
2.  **Implement** a complete Highway Autopilot simulation in Python.
3.  **Simulate** complex traffic scenarios (cut-ins, slow trucks, blocked lanes).
4.  **Validate** the system against safety metrics (Time to Collision, Jerk).
5.  **Evaluate** your mastery of Week 5 concepts through a comprehensive assessment.

---

## 📚 Week 5 Review

### 1. Global Planning (The Map)
-   **A*:** Finds the shortest path on a grid.
-   **RRT:** Finds a path in continuous space (good for parking).
-   **Output:** A list of waypoints $(x, y)$.

### 2. Behavior Planning (The Decision)
-   **FSM:** Discrete states (`LaneKeep`, `LaneChange`).
-   **Logic:** "If car ahead is slow AND left lane is free -> Change Left."
-   **Output:** A "Target" for the local planner (e.g., Target Lane, Target Speed).

### 3. Local Planning (The Trajectory)
-   **Frenet Frame:** Simplifies road coordinates to $(s, d)$.
-   **Quintic Polynomials:** Generate smooth, jerk-minimized curves.
-   **Cost Function:** Selects the best curve based on safety and comfort.
-   **Output:** A trajectory $(x, y, v, t)$.

### 4. Control (The Execution)
-   **PID:** Reactive error correction.
-   **MPC:** Predictive optimization.
-   **Output:** Steering angle $\delta$ and Throttle $a$.

### 5. Safety (The Guard)
-   **ISO 26262:** Process for ensuring safety.
-   **Safety Monitor:** Independent check that overrides the planner if limits are exceeded.

---

## 🛠️ Capstone Project: Highway Autopilot

**Goal:** Build a simulation where an Ego Car navigates a 3-lane highway with traffic.
**Features:**
-   **Sensors:** "Perfect" perception of surrounding cars.
-   **Planner:** Frenet-based Local Planner + FSM.
-   **Traffic:** Other cars move at constant speeds but different lanes.

### Architecture
1.  **World:** Manages the road and all vehicles.
2.  **EgoVehicle:** Contains the Planner and Controller.
3.  **TrafficVehicle:** Simple physics.

### Package Structure
Create `week5_project` folder.

```bash
mkdir -p ~/ros2_ws/src/week5_project
cd ~/ros2_ws/src/week5_project
touch highway_autopilot.py
```

### 👨‍💻 Code: The Simulation

```python
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import time

# --- Constants ---
MAX_SPEED = 50.0 / 3.6  # 50 km/h
TARGET_SPEED = 30.0 / 3.6 # 30 km/h
LANE_WIDTH = 4.0
DT = 0.2

# --- Helper Classes ---

class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        
        A = np.array([[time**3, time**4, time**5],
                      [3 * time**2, 4 * time**3, 5 * time**4],
                      [6 * time, 12 * time**2, 20 * time**3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time**2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        return self.a0 + self.a1 * t + self.a2 * t**2 + \
               self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

    def calc_first_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + \
               3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

    def calc_third_derivative(self, t):
        return 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

class FrenetPath:
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0
        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []

def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, target_speed, target_d):
    frenet_paths = []
    
    # Generate path to target lane
    # Lateral motion planning
    for Ti in np.arange(3.0, 5.0, 0.5): # Time horizon
        fp = FrenetPath()
        lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, target_d, 0.0, 0.0, Ti)
        
        fp.t = [t for t in np.arange(0.0, Ti, DT)]
        fp.d = [lat_qp.calc_point(t) for t in fp.t]
        fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
        fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
        fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]
        
        # Longitudinal motion planning (Velocity keeping)
        # Sample target speeds around the desired target speed
        for tv in np.arange(target_speed - 2.0, target_speed + 2.0, 1.0):
            tfp = copy.deepcopy(fp)
            lon_qp = QuinticPolynomial(s0, c_speed, 0.0, s0 + tv * Ti, tv, 0.0, Ti)
            
            tfp.s = [lon_qp.calc_point(t) for t in fp.t]
            tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
            tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
            tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
            
            # Cost
            J_p = sum(np.power(tfp.d_ddd, 2))  # square of jerk
            J_s = sum(np.power(tfp.s_ddd, 2))  # square of jerk
            
            # Weights
            KJ = 0.1
            KT = 0.1
            KD = 1.0
            KV = 1.0
            KLAT = 1.0
            
            tfp.cd = KJ * J_p + KT * Ti + KD * tfp.d[-1]**2
            tfp.cv = KJ * J_s + KT * Ti + KV * (target_speed - tfp.s_d[-1])**2
            tfp.cf = KLAT * tfp.cd + tfp.cv
            
            frenet_paths.append(tfp)
            
    return frenet_paths

def check_collision(fp, obstacles):
    # obstacles: [[s, d], ...]
    # Simple check: if any point in path is close to any obstacle
    for i in range(len(fp.s)):
        for obs in obstacles:
            d_s = fp.s[i] - obs[0]
            d_d = fp.d[i] - obs[1]
            if abs(d_s) < 5.0 and abs(d_d) < 2.0: # Collision box 10m x 4m
                return True
    return False

# --- Main Simulation ---

def main():
    # Ego State
    c_speed = 10.0 / 3.6
    c_d = 0.0 # Center lane (Lanes: -4, 0, 4)
    c_d_d = 0.0
    c_d_dd = 0.0
    s0 = 0.0
    
    target_d = 0.0 # Target Lane
    
    # Traffic (s, d, v)
    traffic = [
        [50, 0, 15/3.6], # Slow car ahead in center
        [50, 4, 30/3.6], # Fast car in right
        [100, -4, 20/3.6] # Car in left
    ]
    
    area = 200.0 # Simulation length
    
    plt.figure(figsize=(10, 5))
    
    for i in range(500):
        # 1. Behavior Planning (FSM Logic)
        # Find nearest car ahead
        min_dist = 999
        front_car_speed = TARGET_SPEED
        
        for obs in traffic:
            if obs[1] == target_d and obs[0] > s0:
                dist = obs[0] - s0
                if dist < min_dist:
                    min_dist = dist
                    front_car_speed = obs[2]
        
        # Logic
        desired_speed = TARGET_SPEED
        if min_dist < 20:
            desired_speed = front_car_speed # Match speed
            
            # Try to change lane?
            if target_d == 0: # If in center
                # Check Left
                left_free = True
                for obs in traffic:
                    if obs[1] == -4 and abs(obs[0] - s0) < 20: left_free = False
                
                if left_free:
                    print("Overtaking Left!")
                    target_d = -4
        
        # 2. Local Planning (Trajectory Generation)
        paths = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, desired_speed, target_d)
        
        # 3. Collision Check & Selection
        best_path = None
        min_cost = float('inf')
        
        for fp in paths:
            if not check_collision(fp, traffic):
                if fp.cf < min_cost:
                    min_cost = fp.cf
                    best_path = fp
        
        if best_path is None:
            print("EMERGENCY STOP! No path found.")
            break
            
        # 4. Update State (Simulate execution)
        # We assume perfect tracking, so we just move to the next point in the path
        s0 = best_path.s[1]
        c_d = best_path.d[1]
        c_d_d = best_path.d_d[1]
        c_d_dd = best_path.d_dd[1]
        c_speed = best_path.s_d[1]
        
        # Update Traffic
        for obs in traffic:
            obs[0] += obs[2] * DT
            
        # 5. Visualization
        if i % 2 == 0:
            plt.cla()
            # Draw Lanes
            plt.plot([s0-10, s0+50], [-2, -2], 'k--')
            plt.plot([s0-10, s0+50], [2, 2], 'k--')
            
            # Draw Ego
            plt.plot(s0, c_d, 'bo', label='Ego')
            plt.plot(best_path.s, best_path.d, 'b-')
            
            # Draw Traffic
            for obs in traffic:
                plt.plot(obs[0], obs[1], 'rs', label='Traffic')
                
            plt.xlim(s0-10, s0+50)
            plt.ylim(-6, 6)
            plt.title(f"Speed: {c_speed*3.6:.1f} km/h | Target Lane: {target_d}")
            plt.grid()
            plt.pause(0.01)

    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🧪 Verification & Testing

### 1. Overtaking Logic
**Scenario:** Slow car in center lane.
**Observation:** Ego car approaches, slows down to match speed, checks left lane. If free, it generates a trajectory to $d=-4$ and accelerates.

### 2. Blocked Overtake
**Scenario:** Slow car in center, AND car in left lane.
**Observation:** Ego car stays in center lane and matches speed (ACC mode).

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Planning
1.  **Q:** Why is A* not suitable for local trajectory generation?
    *   **A:** A* works on a grid and produces non-smooth paths. Vehicles need smooth derivatives (velocity, accel) which polynomials provide.
2.  **Q:** What is the "Dynamic Window Approach" (DWA)?
    *   **A:** A local planner that samples velocities $(v, \omega)$ reachable in the next time step, simulates them, and picks the best one.

### Section 2: Control
3.  **Q:** Explain the difference between Lateral and Longitudinal control.
    *   **A:** Lateral = Steering (Lane keeping). Longitudinal = Throttle/Brake (Speed keeping).
4.  **Q:** Why does High Gain in PID cause oscillation?
    *   **A:** The controller overreacts to error, pushing the system past the setpoint, then overreacts the other way.

### Section 3: Safety
5.  **Q:** What is HARA?
    *   **A:** Hazard Analysis and Risk Assessment. The process of identifying hazards and assigning ASIL levels.
6.  **Q:** If a planner fails to find a path, what should the system do?
    *   **A:** Trigger a Safe State (e.g., Emergency Brake or Minimum Risk Maneuver).

---

## 🏆 Conclusion

Congratulations on completing Week 5!
-   You have moved from Perception (Seeing) to Localization (Positioning) to Planning (Thinking).
-   You have built a complete Autopilot simulation.
-   You understand the safety-critical nature of these systems.

**Next Week:** We enter the world of **Embedded Systems & ROS 2**. We will take these algorithms out of Python simulation and put them onto real hardware (or high-fidelity simulation like CARLA/Gazebo) using C++.

---

**Day 35 Complete** | Phase 4: ADAS & Robotics Systems | Week 5: Path Planning & Decision Making
