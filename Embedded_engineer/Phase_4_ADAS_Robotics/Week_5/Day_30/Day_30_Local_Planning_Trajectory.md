# Day 30: Local Planning (Trajectory Generation)
## Phase 4: ADAS & Robotics Systems | Week 5: Path Planning & Decision Making

---

> **📝 Day 30 Focus:**
> Global planners (A*) give us a jagged line of waypoints. But cars don't drive in jagged lines. They need smooth curves that respect velocity, acceleration, and jerk limits. Today, we master **Local Planning** using **Frenet Coordinates** and **Quintic Polynomials** to generate silky smooth trajectories.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the Frenet Frame ($s, d$) and why it simplifies highway planning.
2.  **Derive** Quintic Polynomials for jerk-minimizing trajectories.
3.  **Implement** a conversion function between Cartesian ($x, y$) and Frenet ($s, d$) coordinates.
4.  **Generate** a set of candidate trajectories (Lane keeping, Lane change, Emergency stop).
5.  **Select** the best trajectory based on a cost function (Safety, Comfort, Efficiency).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Calculus:** Derivatives (Velocity, Accel, Jerk).
-   **Geometry:** Curvature, Tangent vectors.

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS (or Windows/Mac with Python).

### Software Stack
-   **Python Libraries:** `numpy`, `matplotlib`, `scipy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Frenet Frame

In Cartesian space ($x, y$), a curved road makes math hard.
In **Frenet Space** ($s, d$), the road is a straight line.
-   **$s$ (Longitudinal):** Distance along the center of the lane.
-   **$d$ (Lateral):** Perpendicular offset from the center line.
    -   $d = 0$: Center of lane.
    -   $d = -4$: Left lane.
    -   $d = +4$: Right lane.

**Benefit:** Lane keeping becomes "Keep $d=0$". Lane changing becomes "Go from $d=0$ to $d=4$".

### 🔹 Part 2: Jerk Minimization

Passengers feel **Acceleration** (Force). They feel changes in acceleration (**Jerk**) even more (Nausea).
To minimize Jerk ($J(t) = \dddot{x}(t)$), the optimal trajectory is a **Quintic Polynomial** (Degree 5).

$$ s(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5 $$

We have 6 unknowns ($a_0 \dots a_5$). We need 6 boundary conditions:
-   Start ($t_0$): $s_0, \dot{s}_0, \ddot{s}_0$ (Position, Velocity, Accel).
-   End ($t_1$): $s_1, \dot{s}_1, \ddot{s}_1$.

Solving this system gives the smoothest path connecting start and end.

---

### 🔹 Part 3: Trajectory Generation Algorithm

1.  **Sample Targets:**
    -   Target speeds: $v_{target} \in [20, 30, 40]$ mph.
    -   Target lanes: $d_{target} \in [-4, 0, 4]$.
    -   Target times: $T \in [2, 4, 6]$ seconds.
2.  **Generate Polynomials:**
    -   Calculate $s(t)$ and $d(t)$ polynomials for each combination.
3.  **Convert to Cartesian:**
    -   Map $(s, d)$ back to $(x, y)$ using the map's reference line (Spline).
4.  **Evaluate Costs:**
    -   $C_{total} = w_1 C_{jerk} + w_2 C_{collision} + w_3 C_{efficiency} + w_4 C_{lane}$.
5.  **Pick Best:** Execute the lowest cost trajectory.

---

## 💻 Implementation: Frenet Optimal Trajectory

We will implement a script `frenet_planner.py`.

### 🛠️ Setup
Create `week5_day30` and `frenet_planner.py`.

```bash
mkdir -p ~/ros2_ws/src/week5_day30
cd ~/ros2_ws/src/week5_day30
touch frenet_planner.py
```

### 👨‍💻 Code: Quintic Polynomial Solver

```python
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # xs: start pos, vxs: start vel, axs: start accel
        # xe: end pos, vxe: end vel, axe: end accel
        # time: duration
        
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

def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []
    
    # Parameters
    MAX_SPEED = 50.0 / 3.6  # m/s
    MAX_ACCEL = 2.0  # m/s^2
    MAX_CURVATURE = 1.0  # 1/m
    MAX_ROAD_WIDTH = 7.0  # m
    D_ROAD_W = 1.0  # m
    DT = 0.2  # s
    MAX_T = 5.0  # s
    MIN_T = 4.0  # s
    TARGET_SPEED = 30.0 / 3.6  # m/s
    D_T_S = 5.0 / 3.6  # target speed sampling length
    N_S_SAMPLE = 1  # sampling number of target speed
    
    # Generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):
        
        # Lateral motion planning (d)
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()
            
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            
            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]
            
            # Longitudinal motion planning (s) (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuinticPolynomial(s0, c_speed, 0.0, 
                                           s0 + tv * Ti, tv, 0.0, Ti) # Quartic is enough for velocity keeping
                
                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
                
                # Cost Function
                J_p = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                J_s = sum(np.power(tfp.s_ddd, 2))  # square of jerk
                
                # Weights
                KJ = 0.1
                KT = 0.1
                KD = 1.0
                KV = 1.0
                KLAT = 1.0
                
                tfp.cd = KJ * J_p + KT * Ti + KD * tfp.d[-1]**2
                tfp.cv = KJ * J_s + KT * Ti + KV * (TARGET_SPEED - tfp.s_d[-1])**2
                tfp.cf = KLAT * tfp.cd + tfp.cv
                
                frenet_paths.append(tfp)
                
    return frenet_paths

def main():
    # Initial State
    c_speed = 10.0 / 3.6  # current speed [m/s]
    c_d = 2.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position
    
    paths = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
    
    # Visualization
    plt.figure(figsize=(10, 5))
    
    # Plot all candidates
    for fp in paths:
        plt.plot(fp.s, fp.d, "-g", alpha=0.3)
        
    # Find best path
    best_path = min(paths, key=lambda x: x.cf)
    plt.plot(best_path.s, best_path.d, "-r", linewidth=3, label="Optimal Trajectory")
    
    plt.title("Frenet Optimal Trajectory Generation")
    plt.xlabel("S (Longitudinal)")
    plt.ylabel("D (Lateral)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: Lane Change

### Lab Objectives
1.  Run the script.
2.  **Scenario:** Start at $d=2$ (Right lane). Set target to $d=-2$ (Left lane).
3.  **Observation:** The planner generates a smooth S-curve to move from right to left.
4.  **Cost Tuning:** Increase $K_J$ (Jerk weight).
    -   *Result:* The lane change takes longer (smoother).
    -   Decrease $K_T$ (Time weight).
    -   *Result:* The lane change is aggressive (faster).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Trajectory Overshoot
**Symptom:** Car swings wide before settling in lane.
**Cause:** Boundary conditions (Velocity/Accel) at end point are too strict or time $T$ is too short.
**Solution:** Increase min $T$ or relax end constraints.

#### 2. High Curvature
**Symptom:** Path is physically impossible (turn radius too small).
**Cause:** Converting $(s, d)$ to $(x, y)$ on a sharp curve results in singularities.
**Solution:** Check curvature $\kappa$ of the generated path. Discard if $\kappa > \kappa_{max}$.

---

## ⚡ Optimization & Best Practices

### 1. Spatial vs Temporal
We planned in Time ($s(t), d(t)$).
Alternatively, plan in Space ($d(s)$).
-   **Pros:** Independent of speed profile.
-   **Cons:** Harder to optimize for dynamic obstacles.

### 2. Collision Checking
For each candidate trajectory:
-   Convert to $(x, y)$.
-   Check overlap with static map and *predicted* positions of other cars.
-   If collision, set cost = Infinity.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why do we use Quintic (5th order) polynomials?
    *   **A:** To control Start/End Position, Velocity, AND Acceleration. This ensures continuity of Jerk.
2.  **Q:** What is the advantage of the Frenet Frame?
    *   **A:** It decouples longitudinal speed control from lateral steering control, simplifying the math for road following.
3.  **Q:** How do we handle a slow car in front?
    *   **A:** The "Velocity Keeping" cost will be high (collision). The planner will pick a "Lane Change" trajectory (if valid) or a "Follow" trajectory (match speed).

### Challenge Task
**Task:** Obstacle Avoidance.
1.  Add a static obstacle at $(s=30, d=2)$.
2.  In the cost function, check distance to obstacle.
3.  If distance < 2m, add huge cost.
4.  Observe the planner pick a path that swerves around it (or changes lanes).

---

## 📚 Further Reading & References
-   [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame (Werling et al.)](https://ieeexplore.ieee.org/document/5509799) - The seminal paper.

---

**Day 30 Complete** | Phase 4: ADAS & Robotics Systems | Week 5: Path Planning & Decision Making
