# Day 136: Trajectory Generation (Polynomials/Splines)
## Phase 4: ADAS & Robotics Systems | Week 20: Path Planning & Decision Making

---

> **📝 Day 136 Focus:**
> A path is just geometry ($x, y$). A **Trajectory** includes time ($x, y, t$). To drive a car smoothly, we need to control Position, Velocity, Acceleration, and **Jerk** (Change in acceleration). We use **Quintic Polynomials** to generate jerk-minimized paths.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Differentiate** between Path Planning and Trajectory Generation.
2.  **Derive** the Quintic Polynomial equation for jerk minimization.
3.  **Implement** a 1D Quintic Polynomial solver.
4.  **Generate** a 2D Lane Change trajectory.
5.  **Visualize** Position, Velocity, Acceleration, and Jerk profiles.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Calculus:** Derivatives ($x, \dot{x}, \ddot{x}, \dddot{x}$).
-   **Linear Algebra:** Solving $Ax = B$.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Quintic?

We want to move from $x_0$ to $x_1$ in time $T$.
Constraints:
1.  Start Pos: $x(0) = x_0$
2.  Start Vel: $\dot{x}(0) = v_0$
3.  Start Acc: $\ddot{x}(0) = a_0$
4.  End Pos: $x(T) = x_1$
5.  End Vel: $\dot{x}(T) = v_1$
6.  End Acc: $\ddot{x}(T) = a_1$

6 Constraints $\implies$ We need a polynomial with 6 coefficients ($a_0 \dots a_5$).
$$ x(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5 $$
This is a **Quintic (5th order) Polynomial**. It minimizes the squared jerk integral $\int J^2 dt$.

### 🔹 Part 2: Solving for Coefficients

$a_0, a_1, a_2$ are determined directly by start conditions.
$a_3, a_4, a_5$ are solved by a linear system $Ax=B$ using the end conditions.

### 🔹 Part 3: Frenet Frame

For cars, it's easier to plan in **Frenet Coordinates** $(s, d)$.
-   $s$: Longitudinal distance along the road.
-   $d$: Lateral offset from the center line.
-   Lane Change: $d(0) = 0 \to d(T) = 4.0$. $s(t)$ increases with constant velocity.

---

## 💻 Implementation: Quintic Polynomial Solver

**Scenario:**
-   Car at $x=0$, $v=10 m/s$, $a=0$.
-   Target: $x=100$, $v=0 m/s$, $a=0$ (Stop at stop sign).
-   Time: $T=10s$.

### 🛠️ Setup
Create `week20_day136` and `quintic_poly.py`.

```bash
mkdir -p ~/ros2_ws/src/week20_day136
cd ~/ros2_ws/src/week20_day136
touch quintic_poly.py
```

### 👨‍💻 Code: Jerk Minimization

```python
import numpy as np
import matplotlib.pyplot as plt

class QuinticPolynomial:
    def __init__(self, xi, vi, ai, xf, vf, af, T):
        # Calculate Coefficients
        self.a0 = xi
        self.a1 = vi
        self.a2 = 0.5 * ai
        
        A = np.array([
            [T**3, T**4, T**5],
            [3*T**2, 4*T**3, 5*T**4],
            [6*T, 12*T**2, 20*T**3]
        ])
        
        b = np.array([
            xf - self.a0 - self.a1*T - self.a2*T**2,
            vf - self.a1 - 2*self.a2*T,
            af - 2*self.a2
        ])
        
        x = np.linalg.solve(A, b)
        
        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5

    def calc_vel(self, t):
        return self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4

    def calc_acc(self, t):
        return 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3

    def calc_jerk(self, t):
        return 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2

def main():
    print("Generating Trajectory...")
    
    # 1. Longitudinal (Stop Sign)
    # Start: x=0, v=10, a=0
    # End: x=50, v=0, a=0
    # Time: 10s
    lon_qp = QuinticPolynomial(0, 10, 0, 50, 0, 0, 10)
    
    # 2. Lateral (Lane Change)
    # Start: d=0, v=0, a=0
    # End: d=4, v=0, a=0
    # Time: 10s
    lat_qp = QuinticPolynomial(0, 0, 0, 4, 0, 0, 10)
    
    t = np.arange(0, 10.1, 0.1)
    
    x = [lon_qp.calc_point(i) for i in t]
    v = [lon_qp.calc_vel(i) for i in t]
    a = [lon_qp.calc_acc(i) for i in t]
    j = [lon_qp.calc_jerk(i) for i in t]
    
    y = [lat_qp.calc_point(i) for i in t]
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(t, x, label='Longitudinal (s)')
    plt.plot(t, y, label='Lateral (d)')
    plt.title("Position")
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 2, 2)
    plt.plot(t, v)
    plt.title("Velocity (m/s)")
    plt.grid()
    
    plt.subplot(2, 2, 3)
    plt.plot(t, a)
    plt.title("Acceleration (m/s^2)")
    plt.grid()
    
    plt.subplot(2, 2, 4)
    plt.plot(t, j)
    plt.title("Jerk (m/s^3)")
    plt.grid()
    
    plt.figure()
    plt.plot(x, y)
    plt.title("XY Trajectory (Lane Change while Stopping)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.grid()
    
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Comfort Check

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   Velocity decreases smoothly from 10 to 0.
    -   Acceleration is continuous (no jumps).
    -   Jerk is continuous.
3.  **Experiment:**
    -   Change $T$ to 2.0s (Aggressive braking).
    -   **Result:** Acceleration spikes to $-7.5 m/s^2$.
    -   **Analysis:** This exceeds comfort limits (usually $\pm 2 m/s^2$).
    -   **Lesson:** Trajectory Generation must check constraints ($|a| < a_{max}$). If violated, increase $T$.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Singular Matrix
**Symptom:** `LinAlgError: Singular matrix`.
**Cause:** $T=0$.
**Solution:** Ensure $T > 0$.

#### 2. Overshoot
**Symptom:** Car goes backwards before going forwards.
**Cause:** Boundary conditions are physically impossible for the given $T$. (e.g., Stop from 100m/s in 1s).
**Solution:** Feasibility check.

---

## ⚡ Optimization & Best Practices

### 1. Cubic Splines
For long paths (GPS waypoints), Quintic is bad (oscillates).
-   Use **Cubic Splines**: Piecewise cubic polynomials connecting points.
-   Ensures continuity of $x, \dot{x}, \ddot{x}$ at join points.

### 2. Trapezoidal Velocity Profile
For simple robots, we don't need jerk minimization.
-   Accel -> Constant Vel -> Decel.
-   Simpler to compute, but has infinite jerk at transitions.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why do we minimize Jerk?
    *   **A:** Passenger comfort (prevents coffee spilling) and mechanical wear.
2.  **Q:** What is the difference between Cubic and Quintic?
    *   **A:** Cubic (4 coeffs) can satisfy Position and Velocity. Quintic (6 coeffs) can satisfy Position, Velocity, and Acceleration.
3.  **Q:** How do we handle obstacles?
    *   **A:** We generate multiple candidate trajectories (different $d$ offsets and $T$ durations) and pick the one that doesn't hit obstacles and has lowest cost. (This is the **Frenet Frame Planner**).

### Challenge Task
**Task:** 3D Trajectory.
1.  Add a Z-axis (Altitude).
2.  Generate a drone trajectory taking off ($z=0 \to 10$) while moving forward.
3.  Plot in 3D.

---

## 📚 Further Reading & References
-   [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame](https://www.researchgate.net/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame)
-   [PythonRobotics Cubic Spline](https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning/CubicSpline)

---

**Day 136 Complete** | Phase 4: ADAS & Robotics Systems | Week 20: Path Planning & Decision Making
