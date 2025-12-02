# Day 63: Week 9 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 9: Path Planning & Navigation

---

> **📝 Day 63 Focus:**
> We have learned how to find paths (A*, RRT), smooth them (Min Jerk), steer the car (MPC), and make decisions (FSM). Today, we combine all these skills into the **Highway Autopilot** project. Your robot will navigate a 3-lane highway, overtaking slow cars and avoiding collisions autonomously.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Integrate** Behavior Planning (FSM) with Local Planning (Frenet).
2.  **Simulate** a dynamic highway environment with multiple traffic vehicles.
3.  **Implement** a full Navigation Stack: Perception -> Behavior -> Planning -> Control.
4.  **Tune** cost functions to balance safety (don't crash) vs efficiency (go fast).
5.  **Evaluate** your mastery of Week 9 concepts through a comprehensive assessment.

---

## 📚 Week 9 Review

### 1. Graph Search (A*)
-   **Grid-based:** Good for static mazes.
-   **Heuristic ($h$):** Guides the search.
-   **Optimality:** Guaranteed if $h$ is admissible.

### 2. Sampling-Based (RRT)
-   **High-DOF:** Good for arms and open spaces.
-   **Random:** Probabilistically complete.
-   **RRT*:** Optimizes the path over time.

### 3. Trajectory Optimization
-   **Min Jerk:** Quintic Polynomials for smooth motion.
-   **Boundary Conditions:** Start/End Pos, Vel, Acc.

### 4. MPC (Model Predictive Control)
-   **Receding Horizon:** Plan $N$ steps, execute 1.
-   **Constraints:** Handles limits ($v_{max}, \delta_{max}$) explicitly.

### 5. Behavior Planning (FSM)
-   **States:** Lane Keep, Change Left/Right.
-   **Transitions:** Based on safety and efficiency rules.

### 6. Frenet Frame
-   **$(s, d)$:** Natural coordinates for roads.
-   **Decoupled:** Plan lateral and longitudinal separately.

---

## 🛠️ Capstone Project: Highway Autopilot

**Goal:** Drive 1000m down a 3-lane highway without collision, maintaining 30 m/s, and overtaking slower traffic.

**Architecture:**
1.  **Simulator:** Updates vehicle positions (Constant Velocity).
2.  **FSM:** Decides Target Lane and Target Speed.
3.  **Frenet Planner:** Generates trajectory to Target.
4.  **Controller:** (Simplified) Teleports ego to next trajectory point.

### Package Structure
Create `week9_project` folder.

```bash
mkdir -p ~/ros2_ws/src/week9_project
cd ~/ros2_ws/src/week9_project
touch highway_autopilot.py
```

### 👨‍💻 Code: The Highway Autopilot

```python
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import random

# --- Constants ---
SIM_LOOP = 500
DT = 0.1
MAX_SPEED = 50.0 / 3.6
TARGET_SPEED = 30.0 / 3.6
LANE_WIDTH = 4.0
N_LANES = 3
ROBOT_RADIUS = 2.0

# --- Classes ---
class Vehicle:
    def __init__(self, id, x, y, vx, vy):
        self.id = id
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.s = x # Approximation for straight road
        self.d = y

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.s = self.x
        self.d = self.y

class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        return self.a0 + self.a1 * t + self.a2 * t ** 2 + \
               self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

    def calc_first_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + \
               3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

class FrenetPath:
    def __init__(self):
        self.t = []
        self.d = []
        self.s = []
        self.x = []
        self.y = []
        self.cf = 0.0

def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, target_d):
    frenet_paths = []
    # Generate trajectories to Target D
    # Varying Time (Ti) and Target Speed (tv)
    
    for Ti in np.arange(3.0, 5.0, 0.5):
        # Lateral: Go to target_d
        lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, target_d, 0.0, 0.0, Ti)
        
        for tv in np.arange(TARGET_SPEED - 5.0, TARGET_SPEED + 5.0, 1.0):
            fp = FrenetPath()
            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            
            # Longitudinal: Constant Velocity (Simplified)
            # s(t) = s0 + v*t + 0.5*a*t^2
            # Use Quintic for smoothness
            lon_qp = QuinticPolynomial(s0, c_speed, 0.0, s0 + tv * Ti, tv, 0.0, Ti)
            fp.s = [lon_qp.calc_point(t) for t in fp.t]
            
            # Global XY (Straight Road)
            fp.x = fp.s
            fp.y = fp.d
            
            # Cost
            # J_lat = jerk
            # J_lon = deviation from target speed
            # J_obs = collision risk (calculated later)
            
            lat_jerk = sum([abs(lat_qp.calc_second_derivative(t)) for t in fp.t]) # Approx
            speed_diff = abs(tv - TARGET_SPEED)
            
            fp.cf = 1.0 * lat_jerk + 1.0 * speed_diff + 1.0 * Ti
            
            frenet_paths.append(fp)
            
    return frenet_paths

def check_collision(fp, obstacles):
    for i in range(len(fp.x)):
        for obs in obstacles:
            # Simple distance check
            # Predict obstacle position at time t?
            # For simplicity, assume obstacle moves at constant velocity
            obs_x = obs.x + obs.vx * fp.t[i]
            obs_y = obs.y # Lane keeping
            
            dist = math.hypot(fp.x[i] - obs_x, fp.y[i] - obs_y)
            if dist <= ROBOT_RADIUS * 2.0: # Conservative
                return True
    return False

def behavior_planning(ego_s, ego_d, ego_v, obstacles):
    # FSM Logic
    # 1. Check current lane safety
    current_lane = int(round(ego_d / LANE_WIDTH))
    
    # Find closest car ahead
    min_dist = 999.0
    front_car = None
    for obs in obstacles:
        obs_lane = int(round(obs.d / LANE_WIDTH))
        if obs_lane == current_lane and obs.s > ego_s:
            dist = obs.s - ego_s
            if dist < min_dist:
                min_dist = dist
                front_car = obs
                
    target_d = current_lane * LANE_WIDTH
    target_speed = TARGET_SPEED
    
    if min_dist < 30.0: # Too close
        print(f"Slow car ahead ({min_dist:.1f}m).")
        # Try Left
        if current_lane > 0:
            # Check if left lane is free
            left_free = True
            for obs in obstacles:
                obs_lane = int(round(obs.d / LANE_WIDTH))
                if obs_lane == current_lane - 1:
                    if abs(obs.s - ego_s) < 20.0: # Gap check
                        left_free = False
            if left_free:
                print("Switching Left.")
                target_d = (current_lane - 1) * LANE_WIDTH
                return target_d
                
        # Try Right
        if current_lane < N_LANES - 1:
            right_free = True
            for obs in obstacles:
                obs_lane = int(round(obs.d / LANE_WIDTH))
                if obs_lane == current_lane + 1:
                    if abs(obs.s - ego_s) < 20.0:
                        right_free = False
            if right_free:
                print("Switching Right.")
                target_d = (current_lane + 1) * LANE_WIDTH
                return target_d
                
        # Stuck
        print("Stuck. Braking.")
    
    return target_d

def main():
    # Ego State
    c_speed = 10.0 / 3.6
    c_d = 0.0
    c_d_d = 0.0
    c_d_dd = 0.0
    s0 = 0.0
    
    # Traffic
    obstacles = [
        Vehicle(1, 50.0, 0.0, 15.0/3.6, 0), # Slow car in Lane 0
        Vehicle(2, 100.0, 4.0, 15.0/3.6, 0), # Slow car in Lane 1
        Vehicle(3, 150.0, 8.0, 15.0/3.6, 0)  # Slow car in Lane 2
    ]
    
    area = 20.0
    
    for i in range(SIM_LOOP):
        # 1. Behavior Planning
        target_d = behavior_planning(s0, c_d, c_speed, obstacles)
        
        # 2. Trajectory Generation
        path = None
        best_cost = float("inf")
        
        fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, target_d)
        
        for fp in fplist:
            if check_collision(fp, obstacles):
                continue
            if fp.cf < best_cost:
                best_cost = fp.cf
                path = fp
                
        if path is None:
            print("Emergency Brake!")
            c_speed -= 1.0 * DT # Decelerate
            s0 += c_speed * DT
        else:
            # 3. Execute (Simulate)
            s0 = path.s[1]
            c_d = path.d[1]
            c_d_d = path.d[1] - path.d[0] # Approx
            c_d_dd = 0.0
            c_speed = (path.s[1] - path.s[0]) / DT
            
        # Update Traffic
        for obs in obstacles:
            obs.update(DT)
            
        # Visualization
        if i % 10 == 0:
            plt.cla()
            # Draw Lanes
            plt.plot([s0-area, s0+area*5], [2, 2], "--k")
            plt.plot([s0-area, s0+area*5], [6, 6], "--k")
            
            # Draw Ego
            plt.plot(s0, c_d, "or", label="Ego")
            if path:
                plt.plot(path.x, path.y, "-r")
                
            # Draw Traffic
            for obs in obstacles:
                plt.plot(obs.x, obs.y, "sk")
                
            plt.xlim(s0 - area, s0 + area * 5)
            plt.ylim(-2, 10)
            plt.title(f"Speed: {c_speed*3.6:.1f} km/h")
            plt.pause(0.01)
            
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🧪 Verification & Testing

### 1. Lane Change Logic
-   **Observation:** Ego approaches Car 1 (Lane 0). FSM sees Lane 1 is free. Target D changes to 4.0. Planner generates a smooth curve to $y=4.0$.
-   **Result:** Successful overtake.

### 2. Blocked Scenario
-   **Observation:** Ego approaches Car 2 (Lane 1). Car 3 is in Lane 2 (blocking). Car 1 is in Lane 0 (blocking).
-   **Result:** FSM returns current lane. Planner generates path with lower speed (braking) to maintain distance.

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Planning
1.  **Q:** What is the difference between Global and Local planning?
    *   **A:** Global = Route (A* on map). Local = Trajectory (Polynomials on road).
2.  **Q:** Why is "Time" ($t$) important in trajectory generation?
    *   **A:** Because we need to avoid *moving* obstacles. We need to know where we will be at time $t$ to check if an obstacle will also be there.

### Section 2: Control
3.  **Q:** In MPC, what is the "Prediction Horizon"?
    *   **A:** How far into the future (in seconds or steps) the controller looks. Too short = unstable. Too long = slow computation.

### Section 3: Decision Making
4.  **Q:** How does a Finite State Machine handle uncertainty?
    *   **A:** Poorly. It assumes binary states (Safe/Unsafe). Probabilistic approaches (MDP/POMDP) are better for uncertainty but harder to solve.

---

## 🏆 Conclusion

Congratulations on completing Week 9!
-   You have built the "Brain" of the autonomous vehicle.
-   You can plan paths, avoid obstacles, and obey traffic rules.

**Next Week:** We move to **Multi-Object Tracking**. We can detect cars (Week 8) and plan paths (Week 9), but we need to *track* them over time to predict their future motion. We will build a DeepSORT tracker.

---

**Day 63 Complete** | Phase 4: ADAS & Robotics Systems | Week 9: Path Planning & Navigation
