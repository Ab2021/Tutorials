# Day 59: Trajectory Optimization (Minimum Jerk)
## Phase 4: ADAS & Robotics Systems | Week 9: Path Planning & Navigation

---

> **📝 Day 59 Focus:**
> A* gives you a jagged line. RRT gives you a lightning bolt. Robots can't follow these. They have mass and inertia. We need smooth curves with continuous velocity and acceleration. Today, we turn "Paths" into "Trajectories" using **Minimum Jerk Optimization**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Distinguish** between a Path (Geometry) and a Trajectory (Geometry + Time).
2.  **Explain** why minimizing Jerk (derivative of acceleration) is crucial for passenger comfort and robot longevity.
3.  **Derive** the Quintic Polynomial (5th Order) equations for point-to-point motion.
4.  **Solve** the linear system $Ax=b$ to find the polynomial coefficients.
5.  **Implement** a Python script to generate smooth trajectories between waypoints.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Calculus:** Derivatives ($x, \dot{x}, \ddot{x}, \dddot{x}$).
-   **Linear Algebra:** Matrix Inversion.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Path vs Trajectory

-   **Path:** Sequence of points $(x, y)$. No timing info. "Go here, then here."
-   **Trajectory:** Function of time $x(t), y(t)$. "Be at $x=10$ at $t=5.0s$."
-   **Constraints:**
    -   $v < v_{max}$ (Speed limit).
    -   $a < a_{max}$ (Motor torque limit).
    -   $j < j_{max}$ (Comfort limit).

### 🔹 Part 2: Why Minimum Jerk?

-   **Position ($x$):** Where you are.
-   **Velocity ($\dot{x}$):** How fast.
-   **Acceleration ($\ddot{x}$):** Force ($F=ma$).
-   **Jerk ($\dddot{x}$):** Change in Force. High jerk = Shaking/Vibration.
-   **Snap ($\dddot{x}$):** Change in Jerk.

Minimizing the squared jerk integral over time results in the smoothest possible motion for humans and motors.
$$ J = \int_0^T (\dddot{x}(t))^2 dt $$
The solution to this Euler-Lagrange equation is a **5th Order (Quintic) Polynomial**.

### 🔹 Part 3: The Quintic Polynomial

$$ x(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5 $$

We have 6 unknown coefficients ($a_0 \dots a_5$).
We need 6 Boundary Conditions (BCs):
1.  Start Pos: $x(0) = x_s$
2.  Start Vel: $\dot{x}(0) = v_s$
3.  Start Acc: $\ddot{x}(0) = a_s$
4.  End Pos: $x(T) = x_e$
5.  End Vel: $\dot{x}(T) = v_e$
6.  End Acc: $\ddot{x}(T) = a_e$

We set up a linear system $Ax=b$ and solve for $a$.

---

## 💻 Implementation: Min Jerk Generator

**Scenario:**
-   Robot starts at $x=0$ with $v=0, a=0$.
-   Robot must reach $x=10$ with $v=0, a=0$ in $T=5$ seconds.
-   We calculate the trajectory for 1D (can be applied to X and Y independently).

### 🛠️ Setup
Create `week9_day59` and `min_jerk.py`.

```bash
mkdir -p ~/ros2_ws/src/week9_day59
cd ~/ros2_ws/src/week9_day59
touch min_jerk.py
```

### 👨‍💻 Code: Quintic Polynomial Solver

```python
import numpy as np
import matplotlib.pyplot as plt

class QuinticPolynomial:
    def __init__(self, xs, vs, as_, xe, ve, ae, T):
        # xs: Start Position
        # vs: Start Velocity
        # as_: Start Acceleration
        # xe: End Position
        # ve: End Velocity
        # ae: End Acceleration
        # T: Duration
        
        # Coefficients: a0, a1, a2
        self.a0 = xs
        self.a1 = vs
        self.a2 = as_ / 2.0
        
        # Solve for a3, a4, a5
        # System of equations derived from boundary conditions at t=T
        A = np.array([
            [T**3, T**4, T**5],
            [3*T**2, 4*T**3, 5*T**4],
            [6*T, 12*T**2, 20*T**3]
        ])
        
        b = np.array([
            xe - self.a0 - self.a1*T - self.a2*T**2,
            ve - self.a1 - 2*self.a2*T,
            ae - 2*self.a2
        ])
        
        x = np.linalg.solve(A, b)
        
        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1*t + self.a2*t**2 + \
             self.a3*t**3 + self.a4*t**4 + self.a5*t**5
        return xt

    def calc_vel(self, t):
        vt = self.a1 + 2*self.a2*t + \
             3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4
        return vt

    def calc_acc(self, t):
        at = 2*self.a2 + \
             6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3
        return at

    def calc_jerk(self, t):
        jt = 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2
        return jt

class MinJerkTrajectory:
    def __init__(self, start_pose, end_pose, T):
        # start_pose: [x, y, yaw] (We ignore yaw for simple XY independent)
        # end_pose: [x, y, yaw]
        # Assume start/end vel/acc are 0 for simplicity
        
        self.qp_x = QuinticPolynomial(start_pose[0], 0, 0, end_pose[0], 0, 0, T)
        self.qp_y = QuinticPolynomial(start_pose[1], 0, 0, end_pose[1], 0, 0, T)
        self.T = T

    def generate(self, dt=0.1):
        time = np.arange(0.0, self.T + dt, dt)
        rx, ry, ryaw, rv, ra, rj = [], [], [], [], [], []
        
        for t in time:
            rx.append(self.qp_x.calc_point(t))
            ry.append(self.qp_y.calc_point(t))
            
            vx = self.qp_x.calc_vel(t)
            vy = self.qp_y.calc_vel(t)
            v = np.hypot(vx, vy)
            rv.append(v)
            
            ax = self.qp_x.calc_acc(t)
            ay = self.qp_y.calc_acc(t)
            a = np.hypot(ax, ay)
            ra.append(a)
            
            jx = self.qp_x.calc_jerk(t)
            jy = self.qp_y.calc_jerk(t)
            j = np.hypot(jx, jy)
            rj.append(j)
            
            ryaw.append(np.arctan2(vy, vx))
            
        return time, rx, ry, ryaw, rv, ra, rj

def main():
    # Start: (0, 0), End: (10, 5), Time: 5s
    traj = MinJerkTrajectory([0, 0, 0], [10, 5, 0], 5.0)
    t, x, y, yaw, v, a, j = traj.generate()
    
    # Plotting
    plt.figure(figsize=(10, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, y, '-r')
    plt.plot(0, 0, 'og')
    plt.plot(10, 5, 'xb')
    plt.title("Path (XY)")
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(t, v, '-b')
    plt.title("Velocity Profile")
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(t, a, '-g')
    plt.title("Acceleration Profile")
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(t, j, '-k')
    plt.title("Jerk Profile")
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Bang-Bang Comparison

### Lab Objectives
1.  Run the Min Jerk code.
2.  **Observation:**
    -   Velocity is a smooth bell shape.
    -   Acceleration is a smooth sine-like wave (Start +, End -).
    -   Jerk is continuous.
3.  **Comparison:**
    -   Imagine a "Trapezoidal Velocity" profile (Constant Accel -> Constant Vel -> Constant Decel).
    -   The Acceleration graph would be square waves.
    -   The Jerk graph would be infinite spikes (Dirac Delta) at the transitions.
    -   **Result:** Min Jerk is much better for motors.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Overshoot
**Symptom:** Robot goes past the goal and comes back.
**Cause:** $T$ (Duration) is too short for the distance. The math forces the polynomial to oscillate to meet the boundary conditions.
**Solution:** Increase $T$ or check feasibility ($v_{max}, a_{max}$).

#### 2. Singular Matrix
**Symptom:** `LinAlgError`.
**Cause:** $T=0$.
**Solution:** Ensure $T > 0$.

---

## ⚡ Optimization & Best Practices

### 1. Time Optimization
How do we pick $T$?
-   We usually want the *fastest* time that satisfies constraints.
-   **Algorithm:**
    1.  Guess $T$.
    2.  Calculate Trajectory.
    3.  Check if max($v$) < limit and max($a$) < limit.
    4.  If valid, decrease $T$. If invalid, increase $T$. (Binary Search).

### 2. Waypoints
For multiple waypoints ($P_1 \to P_2 \to P_3$):
-   Generate Min Jerk for $P_1 \to P_2$.
-   Generate Min Jerk for $P_2 \to P_3$.
-   **Constraint:** Velocity/Accel at $P_2$ must match for both segments (Continuity).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why 5th order (Quintic)? Why not 3rd order (Cubic)?
    *   **A:** Cubic allows setting Position and Velocity (4 BCs). Quintic allows Position, Velocity, AND Acceleration (6 BCs). If we don't constrain acceleration to 0 at start/end, we get a step change in force (infinite jerk).
2.  **Q:** What is the difference between this and a Spline?
    *   **A:** A Spline usually connects many points. This Quintic Polynomial connects exactly two states. (Though Splines are often made of piecewise polynomials).
3.  **Q:** Does this guarantee collision avoidance?
    *   **A:** No. This is a *local* trajectory generator. It assumes the path is clear. You need a path planner (A*/RRT) to find the waypoints first.

### Challenge Task
**Task:** 2D Waypoint Follower.
1.  Define 3 points: (0,0), (5,5), (10,0).
2.  Generate two trajectories: (0,0)->(5,5) and (5,5)->(10,0).
3.  Set the velocity at (5,5) to be non-zero (e.g., vector pointing to (10,0)).
4.  Stitch them together.

---

## 📚 Further Reading & References
-   [Minimum Jerk Trajectory Generation](https://courses.shadmehrlab.org/Shortcourse/minimumjerk.pdf)
-   [PythonRobotics Quintic Polynomials](https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning/QuinticPolynomialsPlanner)

---

**Day 59 Complete** | Phase 4: ADAS & Robotics Systems | Week 9: Path Planning & Navigation
