# Day 62: Frenet Frame Trajectories
## Phase 4: ADAS & Robotics Systems | Week 9: Path Planning & Navigation

---

> **📝 Day 62 Focus:**
> Driving on a winding road using $(x, y)$ coordinates is hard. The math gets messy. Drivers think in terms of "Lane Center" and "Distance along road". This is the **Frenet Frame** $(s, d)$. Today, we implement the industry-standard planner for highway driving.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Transform** coordinates between Cartesian $(x, y)$ and Frenet $(s, d)$.
2.  **Decouple** planning into Longitudinal ($s(t)$) and Lateral ($d(t)$) components.
3.  **Generate** Quintic Polynomials in the Frenet Frame to change lanes or keep speed.
4.  **Construct** a candidate set of trajectories and select the best one using a Cost Function.
5.  **Implement** a Frenet Optimal Trajectory Planner in Python.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 59:** Minimum Jerk Trajectories (Quintic Polynomials).
-   **Geometry:** Curvature, Tangent vectors.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `matplotlib`, `scipy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Frenet-Serret Frame

Instead of Global X-Y, we use:
-   **$s$ (Longitudinal):** Arc length along the road center line.
-   **$d$ (Lateral):** Perpendicular offset from the center line. (+Left, -Right).

**Advantages:**
-   Lane keeping becomes $d(t) \to 0$.
-   Lane changing becomes $d(t) \to \pm \text{LaneWidth}$.
-   Curved roads become straight lines in $(s, d)$ space.

### 🔹 Part 2: Decoupled Planning

We solve two 1D problems instead of one 2D problem.

1.  **Lateral Planning ($d$ vs $t$):**
    -   Start: $d_0, \dot{d}_0, \ddot{d}_0$.
    -   End: $d_f, \dot{d}_f=0, \ddot{d}_f=0$.
    -   Use Quintic Polynomial.

2.  **Longitudinal Planning ($s$ vs $t$):**
    -   Start: $s_0, \dot{s}_0, \ddot{s}_0$.
    -   End: Target Velocity ($\dot{s}_f$) or Target Position ($s_f$).
    -   Use Quartic (4th) or Quintic (5th) Polynomial.

### 🔹 Part 3: Transformation (Cartesian $\leftrightarrow$ Frenet)

To convert $(s, d)$ back to $(x, y)$ for the controller:
1.  Find the reference point $P_r$ on the centerline at arc length $s$.
2.  Find the heading $\theta_r$ of the centerline at $s$.
3.  $x = x_r - d \sin(\theta_r)$.
4.  $y = y_r + d \cos(\theta_r)$.

---

## 💻 Implementation: Frenet Optimal Planner

**Scenario:**
-   **Road:** A cubic spline representing the centerline.
-   **Obstacles:** Static objects on the road.
-   **Task:** Generate optimal path to avoid obstacles and maintain speed.

### 🛠️ Setup
Create `week9_day62` and `frenet_planner.py`.

```bash
mkdir -p ~/ros2_ws/src/week9_day62
cd ~/ros2_ws/src/week9_day62
touch frenet_planner.py
```

### 👨‍💻 Code: Frenet Planner

```python
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import sys

# --- Constants ---
MAX_SPEED = 50.0 / 3.6  # Maximum speed [m/s]
MAX_ACCEL = 2.0  # Maximum acceleration [m/ss]
MAX_CURVATURE = 1.0  # Maximum curvature [1/m]
MAX_ROAD_WIDTH = 7.0  # Maximum road width [m]
D_ROAD_W = 1.0  # Road width sampling length [m]
DT = 0.2  # Time tick [s]
MAX_T = 5.0  # Max prediction time [m]
MIN_T = 4.0  # Min prediction time [m]
TARGET_SPEED = 30.0 / 3.6  # Target speed [m/s]
D_T_S = 5.0 / 3.6  # Target speed sampling length [m/s]
N_S_SAMPLE = 1  # Sampling number of target speed
ROBOT_RADIUS = 2.0  # Robot radius [m]

# Cost Weights
KJ = 0.1
KT = 0.1
KD = 1.0
KLAT = 1.0
KLON = 1.0

class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
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

    def calc_third_derivative(self, t):
        return 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

class QuarticPolynomial:
    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        return self.a0 + self.a1 * t + self.a2 * t ** 2 + \
               self.a3 * t ** 3 + self.a4 * t ** 4

    def calc_first_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + \
               3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

    def calc_third_derivative(self, t):
        return 6 * self.a3 + 24 * self.a4 * t

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

    # Lateral Sampling (Lane Offset)
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):
        # Time Sampling
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal Sampling (Target Speed)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                # Cost Function
                Jp = sum(np.power(tfp.d_ddd, 2))  # Lateral Jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # Longitudinal Jerk
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2  # Speed deviation

                tfp.cd = KJ * Jp + KT * Ti + KD * tfp.d[-1] ** 2
                tfp.cv = KJ * Js + KT * Ti + KD * ds
                tfp.cf = KLAT * tfp.cd + KLON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths

def calc_global_paths(fplist, csp):
    for fp in fplist:
        # Transform Frenet to Global
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None: break
            iyaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(iyaw + math.pi / 2.0)
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # Calc Yaw and Curvature
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))
        
        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # Curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist

def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]
        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])
        if collision:
            return False
    return True

def check_paths(fplist, ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        if any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
            continue
        if any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
            continue
        if not check_collision(fplist[i], ob):
            continue
        ok_ind.append(i)
    return [fplist[i] for i in ok_ind]

def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob):
    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    # Find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path

# --- Spline Helper (Simplified) ---
class CubicSpline2D:
    def __init__(self, x, y):
        self.s = np.linspace(0, 100, 1000) # Dummy s
        self.x = x
        self.y = y
        
    def calc_position(self, s):
        # Linear interpolation for simplicity in this snippet
        # Real implementation uses Cubic Spline
        if s >= 100: return None, None
        idx = int(s / 100.0 * len(self.x))
        idx = min(idx, len(self.x)-1)
        return self.x[idx], self.y[idx]
        
    def calc_yaw(self, s):
        return 0.0 # Straight road

def main():
    # Waypoints
    wx = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    wy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ob = np.array([[30.0, 0.0]]) # Obstacle on road

    csp = CubicSpline2D(wx, wy)
    
    # Initial State
    c_speed = 10.0 / 3.6
    c_d = 0.0
    c_d_d = 0.0
    c_d_dd = 0.0
    s0 = 0.0
    
    path = frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob)
    
    if path:
        plt.plot(wx, wy, "r--", label="Ref")
        plt.plot(ob[:, 0], ob[:, 1], "xk", label="Obs")
        plt.plot(path.x, path.y, "-b", label="Traj")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()
    else:
        print("No path found!")

if __name__ == '__main__':
    main()
```

---

## 🔬 Lab Exercise: The Overtake

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   The robot starts at $(0, 0)$.
    -   Obstacle at $(30, 0)$.
    -   The planner generates many paths (Blue lines).
    -   The "Best" path swerves to $d=2.0$ (Left) or $d=-2.0$ (Right) to avoid the obstacle, then returns to $d=0$.
3.  **Experiment:**
    -   Increase `KJ` (Jerk Cost). The path will be smoother but might start turning earlier.
    -   Increase `KD` (Deviation Cost). The robot will try harder to stay close to $d=0$, potentially getting dangerously close to the obstacle before swerving.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. No Path Found
**Symptom:** Robot stops.
**Cause:** All trajectories collide or violate constraints ($a_{max}$).
**Solution:** Reduce speed, increase `MAX_ACCEL`, or allow wider road.

#### 2. Oscillations
**Symptom:** Robot wiggles left/right.
**Cause:** Cost function is unstable or `DT` is too large.
**Solution:** Tune weights ($K_{lat}$ vs $K_{lon}$). Add hysteresis to the chosen path (stick to current decision unless new one is significantly better).

---

## ⚡ Optimization & Best Practices

### 1. Spatial Lookup
Checking collision against *all* obstacles for *all* paths is slow.
-   Use a **Spatial Hash Map** or **Quadtree** to only check nearby obstacles.

### 2. Parallelization
Generating 1000 trajectories is "Embarrassingly Parallel".
-   Use GPU (CUDA) or Multi-threading to generate and evaluate paths in parallel.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why do we use Quartic (4th) for Longitudinal and Quintic (5th) for Lateral?
    *   **A:** For Lateral, we want start and end state to be stable ($d, \dot{d}, \ddot{d}$ all defined). For Longitudinal, we usually just care about reaching a target *velocity* ($\dot{s}$), not a specific *position* ($s$), so we have one less constraint -> Quartic.
2.  **Q:** What happens if the road curves sharply?
    *   **A:** The Frenet transformation breaks down if the radius of curvature is smaller than the road width (Singularity).
3.  **Q:** How does this handle dynamic obstacles?
    *   **A:** By predicting the obstacle's position at time $t$ ($s_{obs}(t), d_{obs}(t)$) and checking collision in the $(s, d)$ domain.

### Challenge Task
**Task:** Dynamic Overtake.
1.  Make the obstacle move at $v=5$ m/s.
2.  Update the collision check to use `ob_s + ob_v * t`.
3.  Observe the robot planning a path *behind* or *in front* of the moving obstacle.

---

## 📚 Further Reading & References
-   [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame (Werling et al.)](https://www.researchgate.net/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame)
-   [PythonRobotics Frenet Planner](https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning/FrenetOptimalTrajectory)

---

**Day 62 Complete** | Phase 4: ADAS & Robotics Systems | Week 9: Path Planning & Navigation
