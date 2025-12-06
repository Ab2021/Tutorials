# Day 153: Motion Planning (Frenet Frame)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 22: Autonomous Driving Stack

---

> **📝 Content Creator Instructions:**
> Solve the maze in $s$ and $d$.
> - **Focus:** Frenet Coordinates ($s$: Longitudinal, $d$: Lateral), Quintic Polynomials (Jerk Minimization), Cost Functions (Safety, Comfort, Efficiency), and Local Trajectory Generation.
> - **Code:** A Python script `frenet_planner.py` that generates optimal paths to overtake a slow obstacle by sampling target states ($d_{target}$, $v_{target}$) in the Frenet frame and converting back to Global XY.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Transform** Cartesian coordinates $(x, y, \theta)$ to Frenet coordinates $(s, d)$.
2.  **Generate** Quintic Polynomials $d(t) = a_0 + a_1 t + \dots + a_5 t^5$ to ensure minimized jerk.
3.  **Evaluate** candidate trajectories based on weighted costs (Similarity to collision, Exceeding speed limit).
4.  **Implement** a Highway Overtaking planner.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Polynomials.
- Optimization Cost Functions.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Frenet?

Driving on a curvy road in $(x,y)$ is hard.
*   **Frenet Frame:** Unroll the road.
    *   **$s$:** Distance along the centerline (Arc length).
    *   **$d$:** Lateral offset from centerline (Left is positive).
*   **Result:** A curved lane change becomes a simple movement from $d=0$ to $d=4$.

### 🔹 Part 2: Jerk Minimization

To be comfortable (no coffee spills), we minimize Jerk ($\dddot{x}$).
Using the Euler-Lagrange equation, the functional that minimizes squared jerk integral is a **Quintic Polynomial** (5th order).
$$ d(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5 $$
*   6 Unknowns ($a_0 \dots a_5$).
*   6 Constraints: Start ($d_0, \dot{d}_0, \ddot{d}_0$) and End ($d_T, \dot{d}_T, \ddot{d}_T$).
*   System is solvable.

### 🔹 Part 3: The Sampling Methods

We don't know the exact best path. We **Sample**:
1.  **Target Speeds:** $V_{cur}, V_{cur}+5, V_{cur}-5$.
2.  **Target Lanes:** $d=0, d=4, d=8$.
3.  **Time Horizons:** $T=3s, 4s, 5s$.
Result: ~50 Trajectories.
Remove those that crash. Pick the one with lowest Cost.

---

## 💻 Implementation: The Highway Planner

We implement the Werling et al. (2010) approach.

### 🛠️ Project Structure
```text
day153_planning/
├── src/
│   ├── frenet_planner.py
└── output/
    ├── optimal_traj.png
```

### 👨‍💻 Frenet Planner (`src/frenet_planner.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # Calc coefficients of a0 + a1*t + ... + a5*t^5
        # Start conditions
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        # End conditions matrix inverse solution (simplified)
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

    def calc_first_deriv(self, t):
        return self.a1 + 2 * self.a2 * t + \
               3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

    def calc_second_deriv(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3
    
    def calc_third_deriv(self, t):
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
        self.c = 0.0 # Cost

def frenet_optimal_planning(c_s, c_s_d, c_s_dd, c_d, c_d_d, c_d_dd, obstacles):
    
    frenet_paths = []
    
    # 1. Sample Target States
    # Lateral (Lanes at 0, 4, 8)
    # Assuming road width 8m (2 lanes, center at 2 and 6? No, let's say center at 0)
    # Let's say we are on a 1-lane road d=0, want to overtake?
    # Simple Highway: Lane Width 4m. Lanes at 0, 4, -4.
    
    # Target d
    for di in [-4, 0, 4]:
        # Target Time
        for Ti in [4.0, 5.0]:
            # Target Speed (Longitudinal)
            # Sample around current speed + some acceleration
            target_v = 30.0 / 3.6 # 30 km/h
            
            # Use Quartic for Longitudinal (End Position not fixed, Velocity Fixed)
            # Actually standard is usually Quintic for d, Quartic for s (Minimizing Jerk to reach Velocity)
            
            # Simple approach: Assume constant speed s_d for now? No, use polynomial.
            # Start s, s_d, s_dd
            # End s_d = target_v, s_dd = 0
            
            fp = FrenetPath()
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            # Long_qp (Quartic: start s, s_d, s_dd | end s_d, s_dd) - Simplification: Quintic with dummy s_end
            # Let's estimate s_end based on speed
            s_end_approx = c_s + target_v * Ti
            lon_qp = QuinticPolynomial(c_s, c_s_d, c_s_dd, s_end_approx, target_v, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, 0.1)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_deriv(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_deriv(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_deriv(t) for t in fp.t]
            
            fp.s = [lon_qp.calc_point(t) for t in fp.t]
            fp.s_d = [lon_qp.calc_first_deriv(t) for t in fp.t]
            fp.s_dd = [lon_qp.calc_second_deriv(t) for t in fp.t]
            fp.s_ddd = [lon_qp.calc_third_deriv(t) for t in fp.t]

            # Cost Function
            J_lat = sum(np.power(fp.d_ddd, 2))  # Lateral Jerk
            J_long = sum(np.power(fp.s_ddd, 2)) # Longitudinal Jerk
            
            # Tracking Cost (Deviation from Target d)
            diff_d = fp.d[-1] - di
            # Speed Cost
            diff_v = fp.s_d[-1] - target_v
            
            fp.c = 1.0 * J_lat + 1.0 * J_long + 1.0 * np.abs(diff_d) + 1.0 * np.abs(diff_v)
            
            # Collision Check
            collision = False
            # Convert s,d to x,y (Simplified: straight road along X axis)
            fp.x = list(fp.s)
            fp.y = list(fp.d)
            
            # Check obstacles (Simple Box)
            for obs in obstacles:
                # Obs: [s, d, radius]
                for i in range(len(fp.x)):
                    dist = math.sqrt((fp.x[i] - obs[0])**2 + (fp.y[i] - obs[1])**2)
                    if dist <= obs[2] + 2.0: # Robot radius 2.0
                        collision = True
                        break
            
            if not collision:
                frenet_paths.append(fp)

    return frenet_paths

def main():
    # Current State
    c_s = 0.0
    c_s_d = 10.0 / 3.6 # 10 km/h
    c_s_dd = 0.0
    
    c_d = 0.0
    c_d_d = 0.0
    c_d_dd = 0.0
    
    # Obstacles: [s, d, radius]
    obstacles = [
        [30.0, 0.0, 2.0] # Blocking center lane
    ]
    
    paths = frenet_optimal_planning(c_s, c_s_d, c_s_dd, c_d, c_d_d, c_d_dd, obstacles)
    
    # Select Best
    best_path = min(paths, key=lambda p: p.c)
    
    print(f"Generated {len(paths)} candidates.")
    print(f"Best Path Cost: {best_path.c:.2f}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot Candidates
    for p in paths:
        plt.plot(p.x, p.y, "-", color="gray", alpha=0.3)
        
    # Plot Best
    plt.plot(best_path.x, best_path.y, "-r", linewidth=2, label="Optimal")
    
    # Plot Obstacles
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='black')
        plt.gca().add_patch(circle)
        
    plt.axhline(4, color='b', linestyle='--')
    plt.axhline(0, color='b', linestyle='--')
    plt.axhline(-4, color='b', linestyle='--')
    
    plt.title("Frenet Frame Local Planner")
    plt.xlabel("S (Longitudinal)")
    plt.ylabel("D (Lateral)")
    plt.grid()
    plt.axis("equal")
    plt.legend()
    plt.savefig("output/optimal_traj.png")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Speeding Ticket"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:** Car swerves to $d=4$ or $d=-4$ to avoid obstacle at $s=30$.
- **Modify:** Change obstacle to $s=100$.
- **Result:** Car stays in $d=0$ (Lowest jerk cost) until close.
- **Task:** Add `Cost_Speed_Limit`. If $v > 30$, Cost += 1000.
- **Result:** Planner chooses trajectories that respect the limit, even if it means arriving later.

---

## 🚀 Project: "Curved Road Transform"

**Goal:** Handle non-straight roads.
1.  **Reference Line:** Cubic Spline through waypoints `(wx, wy)`.
2.  **Transform:**
    *   Find nearest point on Spline.
    *   $s$ = arc length to that point.
    *   $d$ = dist to that point.
    *   $\vec{d}$ is normal vector.
3.  **Inverse Transform:**
    *   $x = x_{ref}(s) + d \cdot \cos(\theta_{ref}(s) + \pi/2)$
    *   $y = y_{ref}(s) + d \cdot \sin(\theta_{ref}(s) + \pi/2)$

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Infeasible Trajectory"
*   **Cause:** Asking to move lateral 4m in 0.1s.
*   **Fix:** Validate Constraints (Max curvature, Max accel) *after* generation. Discard invalid ones.

#### 2. "Oscillation"
*   **Cause:** Cost function not consistent. Frame 1 picks Left lane. Frame 2 picks Right lane.
*   **Fix:** Add `Cost_Consistency`. Cost += 10 * |Target_Lane_New - Target_Lane_Old|.

---

## ⚡ Optimization: Lookup Tables

Calculating 50 polynomials every 100ms is cheap (CPU).
Calculating 5000 is expensive.
*   **Solution:** Pre-compute standard maneuvers (Lane change) into a Lookup Table. Select closest valid one and warp it.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why Quintic (5th order)?
    *   **A:** To control Start Position, Velocity, Acceleration and End Position, Velocity, Acceleration (6 constraints). If we only cared about Vel, Cubic (3rd order) would work.
2.  **Q:** What is the "Reference Line"?
    *   **A:** Usually the center of the lane provided by the HD Map or Global Planner (A* on road graph).
3.  **Q:** Static vs Dynamic Obstacles?
    *   **A:** For static, check box overlap. For dynamic, check overlap in $(s, t)$ space (Path-Time Diagram).

### Challenge Task
> **Task:** Follow Car.
> 1. Set $s_{target} = s_{lead\_car} - 10$ (Safety gap).
> 2. Set $s\_d_{target} = s\_d_{lead\_car}$ (Match speed).
> 3. Generate polynomial to match position AND speed.

---

## 📚 Further Reading
- **Werling et al. (2010):** "Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame".
- **Apollo Auto:** Planning Module documentation.

---

**Day 153 Complete**
