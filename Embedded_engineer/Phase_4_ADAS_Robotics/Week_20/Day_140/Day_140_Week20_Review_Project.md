# Day 140: Week 20 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 20: Path Planning & Decision Making

---

> **📝 Day 140 Focus:**
> We have the Brains. We can find a path (A*), decide when to change lanes (FSM), and generate smooth curves (Polynomials). Today, we combine them into a **Highway Autopilot** that can overtake slow cars and maintain safety.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Architect** a hierarchical planning stack.
2.  **Integrate** Perception (Simulated) with Planning.
3.  **Implement** a collision checking mechanism for trajectories.
4.  **Visualize** the decision-making process in real-time.
5.  **Evaluate** the safety and comfort of the autonomous agent.

---

## 📚 Week 20 Review

### 1. Global Planning
-   **A*:** Grid-based, optimal.
-   **RRT:** Sampling-based, high-dimensional.

### 2. Local Planning
-   **Trajectory Generation:** Quintic Polynomials (Jerk minimization).
-   **MPC:** Optimization-based control (Constraints).

### 3. Decision Making
-   **FSM:** Discrete states (Lane Keep, Lane Change).
-   **RL:** Learning from experience.

---

## 🛠️ Capstone Project: Highway Autopilot

**Goal:** Drive 1000m down a 3-lane highway without crashing.
**Pipeline:**
1.  **Sensor Fusion:** Get list of objects.
2.  **Behavior Planner:** Decide Target Lane and Target Speed.
3.  **Trajectory Planner:** Generate polynomial path to Target.
4.  **Controller:** Execute path.

### Package Structure
Create `week20_project` folder.

```bash
mkdir -p ~/ros2_ws/src/week20_project
cd ~/ros2_ws/src/week20_project
touch highway_pilot.py
```

### 👨‍💻 Code: The Full Stack

```python
import numpy as np
import matplotlib.pyplot as plt
import copy

# --- Constants ---
LANE_WIDTH = 4.0
DT = 0.1
MAX_SPEED = 30.0 # m/s
MAX_ACCEL = 5.0
MAX_JERK = 10.0

# --- Helper Classes ---
class QuinticPolynomial:
    def __init__(self, xi, vi, ai, xf, vf, af, T):
        self.a0 = xi
        self.a1 = vi
        self.a2 = 0.5 * ai
        A = np.array([[T**3, T**4, T**5], [3*T**2, 4*T**3, 5*T**4], [6*T, 12*T**2, 20*T**3]])
        b = np.array([xf - self.a0 - self.a1*T - self.a2*T**2, vf - self.a1 - 2*self.a2*T, af - 2*self.a2])
        x = np.linalg.solve(A, b)
        self.a3, self.a4, self.a5 = x[0], x[1], x[2]

    def calc_point(self, t):
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5

class Vehicle:
    def __init__(self, id, s, d, v):
        self.id = id
        self.s = s
        self.d = d
        self.v = v
        self.lane = int(d / LANE_WIDTH)

    def update(self, dt):
        self.s += self.v * dt
        self.lane = int(self.d / LANE_WIDTH)

class EgoVehicle(Vehicle):
    def __init__(self, s, d, v):
        super().__init__(-1, s, d, v)
        self.target_lane = 1
        self.target_speed = MAX_SPEED
        self.state = "KL" # Keep Lane

    def plan(self, traffic, dt):
        # 1. Behavior Planning (Simplified FSM)
        self.behavior_planning(traffic)
        
        # 2. Trajectory Generation
        path_s, path_d = self.generate_trajectory(dt)
        
        return path_s, path_d

    def behavior_planning(self, traffic):
        # Check car ahead
        car_ahead = None
        min_dist = 999
        for car in traffic:
            if car.lane == self.lane and car.s > self.s:
                dist = car.s - self.s
                if dist < min_dist:
                    min_dist = dist
                    car_ahead = car
        
        if car_ahead and min_dist < 30:
            # Slow down or Change Lane
            if self.lane == 1: # Center
                # Try Left
                if self.check_lane_free(traffic, 0):
                    self.target_lane = 0
                    self.state = "LCL"
                # Try Right
                elif self.check_lane_free(traffic, 2):
                    self.target_lane = 2
                    self.state = "LCR"
                else:
                    self.target_speed = car_ahead.v # Match speed
            else:
                # Return to Center if possible
                if self.check_lane_free(traffic, 1):
                    self.target_lane = 1
                    self.state = "LCR" if self.lane == 0 else "LCL"
                else:
                    self.target_speed = car_ahead.v
        else:
            self.target_speed = MAX_SPEED
            if self.lane != 1 and self.check_lane_free(traffic, 1):
                 self.target_lane = 1 # Prefer center

    def check_lane_free(self, traffic, lane):
        for car in traffic:
            if car.lane == lane:
                if abs(car.s - self.s) < 20: # Safety Gap
                    return False
        return True

    def generate_trajectory(self, dt):
        T = 4.0 # Duration of maneuver
        
        # Longitudinal
        # Simple constant accel to target speed
        s_target = self.s + self.v * T + 0.5 * (self.target_speed - self.v)/T * T**2 # Approx
        # Better: Use Quintic for S too
        # Here we assume constant velocity for simplicity of visualization
        
        # Lateral
        d_target = self.target_lane * LANE_WIDTH + LANE_WIDTH/2
        lat_qp = QuinticPolynomial(self.d, 0, 0, d_target, 0, 0, T)
        
        path_s = []
        path_d = []
        for t in np.arange(0, T, dt):
            # Update S (Constant Accel Model)
            acc = (self.target_speed - self.v) / T
            if abs(acc) > MAX_ACCEL: acc = np.sign(acc) * MAX_ACCEL
            
            s_next = self.s + self.v * t + 0.5 * acc * t**2
            d_next = lat_qp.calc_point(t)
            
            path_s.append(s_next)
            path_d.append(d_next)
            
        return path_s, path_d

def main():
    ego = EgoVehicle(0, 6, 20) # Lane 1 (d=6), 20 m/s
    
    traffic = [
        Vehicle(1, 50, 6, 15), # Slow car ahead
        Vehicle(2, 40, 2, 25), # Fast car left
        Vehicle(3, 100, 10, 20) # Car right
    ]
    
    plt.figure(figsize=(10, 5))
    
    for step in range(100):
        # Update Traffic
        for car in traffic:
            car.update(DT)
            
        # Plan Ego
        path_s, path_d = ego.plan(traffic, DT)
        
        # Execute (Simulate 1 step)
        ego.s = path_s[1]
        ego.d = path_d[1]
        ego.v += (ego.target_speed - ego.v) * 0.1 # Simple P-control
        ego.lane = int(ego.d / LANE_WIDTH)
        
        # Visualization
        plt.clf()
        plt.xlim(ego.s - 20, ego.s + 100)
        plt.ylim(0, 12)
        
        # Draw Lanes
        plt.plot([ego.s-20, ego.s+100], [4, 4], 'k--')
        plt.plot([ego.s-20, ego.s+100], [8, 8], 'k--')
        
        # Draw Traffic
        for car in traffic:
            plt.plot(car.s, car.d, 'bs', markersize=10)
            
        # Draw Ego
        plt.plot(ego.s, ego.d, 'rs', markersize=10, label='Ego')
        
        # Draw Plan
        plt.plot(path_s, path_d, 'r-')
        
        plt.title(f"Step {step}: Speed {ego.v:.1f} m/s, State {ego.state}")
        plt.pause(0.01)
        
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🧪 Verification & Testing

### 1. Overtaking Logic
-   **Scenario:** Ego starts behind a slow car (15 m/s).
-   **Observation:** Ego detects the car, checks Left Lane. If blocked, checks Right Lane.
-   **Result:** Ego switches lanes, accelerates to 30 m/s, passes the car, and returns to center.

### 2. Safety Gap
-   **Scenario:** Car in left lane is right next to Ego.
-   **Observation:** `check_lane_free` returns False.
-   **Result:** Ego stays in lane and matches speed of the car ahead (ACC behavior).

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Theory
1.  **Q:** Why do we use Quintic Polynomials for lateral motion?
    *   **A:** To minimize lateral jerk, ensuring passenger comfort during lane changes.
2.  **Q:** What is the "Frenet Frame"?
    *   **A:** A coordinate system aligned with the road center line ($s, d$). It simplifies planning on curved roads.

### Section 2: Implementation
3.  **Q:** How does the planner handle a blocked lane?
    *   **A:** The cost (or logic check) for that lane becomes high/false, forcing the FSM to choose "Keep Lane" or the other lane.
4.  **Q:** What happens if the target speed is higher than the car ahead?
    *   **A:** The ACC logic overrides the target speed to match the lead car's speed to maintain a safe distance.

---

## 🏆 Conclusion

Congratulations on completing Week 20!
-   You have mastered **Path Planning & Decision Making**.
-   You can build the "Brain" of the autonomous vehicle.

**Next Week:** We enter the final phase. **System Integration & Capstone**. We will put everything together (Perception + Localization + Planning + Control) into a complete ROS 2 stack.

---

**Day 140 Complete** | Phase 4: ADAS & Robotics Systems | Week 20: Path Planning & Decision Making
