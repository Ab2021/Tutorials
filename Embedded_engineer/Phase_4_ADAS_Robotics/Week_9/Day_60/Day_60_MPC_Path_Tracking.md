# Day 60: Model Predictive Control (MPC) for Path Tracking
## Phase 4: ADAS & Robotics Systems | Week 9: Path Planning & Navigation

---

> **📝 Day 60 Focus:**
> PID is reactive (looks at the past). Pure Pursuit is geometric (looks at one point). **Model Predictive Control (MPC)** is predictive (looks into the future). It solves an optimization problem *every single time step* to find the best sequence of controls. It is the gold standard for self-driving cars.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the MPC problem: Cost Function, Constraints, and Prediction Horizon.
2.  **Formulate** the Kinematic Bicycle Model as a discrete state-space system.
3.  **Construct** the Quadratic Programming (QP) matrices ($H, f$) for Linear MPC.
4.  **Implement** an MPC controller in Python using `cvxpy` to track a reference path.
5.  **Analyze** the trade-off between Horizon Length ($N$) and Computational Cost.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Optimization:** Minima of functions.
-   **Linear Algebra:** Quadratic Forms ($x^T Q x$).

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `matplotlib`, `cvxpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Receding Horizon

MPC works like a chess player:
1.  **Look Ahead:** Predict what happens for the next $N$ steps if I do $u_0, u_1, \dots, u_{N-1}$.
2.  **Optimize:** Find the sequence $u$ that minimizes the "Badness" (Cost) while obeying the Rules (Constraints).
3.  **Act:** Execute only the *first* step $u_0$.
4.  **Repeat:** At the next step, look ahead $N$ steps again.

### 🔹 Part 2: The Cost Function ($J$)

We want to minimize:
$$ J = \sum_{k=0}^{N-1} (x_k - x_{ref})^T Q (x_k - x_{ref}) + u_k^T R u_k $$
-   **Tracking Error ($Q$):** Stay close to the path.
-   **Control Effort ($R$):** Don't use too much gas/steering.
-   **Smoothness:** Minimize $\Delta u$ (don't jerk the wheel).

### 🔹 Part 3: The Model

We need a model to predict the future.
**Linear Kinematic Model:**
$$ x_{k+1} = A x_k + B u_k $$
For a car, the model is non-linear. We linearize it around the reference trajectory (Linear Time-Varying MPC) or use a Non-Linear Solver (NMPC). Today, we use a simplified Linear Model for understanding.

---

## 💻 Implementation: Linear MPC

**Scenario:**
-   **State:** $[x, y, v, \theta]$ (Simplified to 1D for clarity first, or 2D Linearized).
-   Let's do **2D Path Tracking** using a linearized error model.
-   **Reference:** A Figure-8 path.

### 🛠️ Setup
Create `week9_day60` and `mpc.py`.

```bash
mkdir -p ~/ros2_ws/src/week9_day60
cd ~/ros2_ws/src/week9_day60
touch mpc.py
```

### 👨‍💻 Code: MPC Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import math

# --- Configuration ---
NX = 4  # [x, y, v, theta]
NU = 2  # [accel, steer]
T = 5   # Horizon Length
DT = 0.2 # Time Step

# Weights
R = np.diag([0.01, 0.01]) # Input cost
Rd = np.diag([0.01, 1.0]) # Input difference cost
Q = np.diag([1.0, 1.0, 0.5, 0.5]) # State cost
Qf = Q # Terminal cost

# Limits
MAX_STEER = np.deg2rad(45.0)
MAX_ACCEL = 1.0
MAX_SPEED = 5.0 # m/s

class MPC:
    def __init__(self):
        pass

    def get_linear_model_matrix(self, v, phi, delta):
        # Linearized Kinematic Bicycle Model
        # x' = x + v*cos(phi)*dt
        # y' = y + v*sin(phi)*dt
        # v' = v + a*dt
        # phi' = phi + v/L*tan(delta)*dt
        
        L = 2.5 # Wheelbase
        
        A = np.zeros((NX, NX))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = DT * math.cos(phi)
        A[0, 3] = -DT * v * math.sin(phi)
        A[1, 2] = DT * math.sin(phi)
        A[1, 3] = DT * v * math.cos(phi)
        A[3, 2] = DT * math.tan(delta) / L
        
        B = np.zeros((NX, NU))
        B[2, 0] = DT
        B[3, 1] = DT * v / (L * math.cos(delta)**2)
        
        return A, B

    def solve(self, xref, x0, dref, vref):
        # xref: Reference Trajectory (NX, T+1)
        # x0: Current State
        # dref: Current Steering (Linearization point)
        # vref: Current Velocity (Linearization point)
        
        # Variables
        x = cp.Variable((NX, T + 1))
        u = cp.Variable((NU, T))
        
        cost = 0.0
        constraints = []
        
        # Initial State Constraint
        constraints += [x[:, 0] == x0]
        
        A, B = self.get_linear_model_matrix(vref, x0[3], dref)
        
        for t in range(T):
            cost += cp.quad_form(x[:, t] - xref[:, t], Q)
            cost += cp.quad_form(u[:, t], R)
            
            if t < T - 1:
                cost += cp.quad_form(u[:, t+1] - u[:, t], Rd)
                
            # Model Constraint
            # Note: This is a simplified LTI approximation for the horizon
            # Ideally, we re-linearize at every step of the horizon (LTV)
            constraints += [x[:, t+1] == A @ x[:, t] + B @ u[:, t]]
            
            # Input Constraints
            constraints += [cp.abs(u[0, t]) <= MAX_ACCEL]
            constraints += [cp.abs(u[1, t]) <= MAX_STEER]
            
            # State Constraints (Speed)
            constraints += [x[2, t] <= MAX_SPEED]
            
        # Terminal Cost
        cost += cp.quad_form(x[:, T] - xref[:, T], Qf)
        
        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.ECOS, verbose=False)
        
        if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
            return u[:, 0].value, x[:, :].value
        else:
            print("Error: Cannot solve MPC")
            return None, None

def main():
    mpc = MPC()
    
    # Reference Path (Straight Line for simplicity)
    # x goes 0 to 50, y = 0
    cx = np.arange(0, 50, 0.1)
    cy = np.sin(cx / 5.0) * 5.0 # Sine Wave
    cyaw = np.arctan2(np.diff(cy), np.diff(cx))
    cyaw = np.append(cyaw, cyaw[-1])
    ck = np.zeros_like(cx) # Curvature (ignored for now)
    
    target_speed = 2.0
    
    # Initial State
    state = np.array([0.0, 0.0, 0.0, 0.0]) # x, y, v, yaw
    
    # History
    x_h, y_h = [], []
    
    time = 0.0
    while time < 20.0:
        # 1. Find Reference Horizon
        # Simple: Take closest point and look ahead T steps
        min_ind = np.argmin(np.hypot(cx - state[0], cy - state[1]))
        
        xref = np.zeros((NX, T + 1))
        for i in range(T + 1):
            ind = min(min_ind + i, len(cx) - 1)
            xref[0, i] = cx[ind]
            xref[1, i] = cy[ind]
            xref[2, i] = target_speed
            xref[3, i] = cyaw[ind]
            
        # 2. Solve MPC
        u, pred_x = mpc.solve(xref, state, 0.0, state[2])
        
        if u is None: break
        
        # 3. Update State (Simulation)
        # x' = x + v*cos(phi)*dt
        state[0] += state[2] * math.cos(state[3]) * DT
        state[1] += state[2] * math.sin(state[3]) * DT
        state[2] += u[0] * DT
        state[3] += state[2] * math.tan(u[1]) / 2.5 * DT
        
        x_h.append(state[0])
        y_h.append(state[1])
        time += DT
        
        # Visualization
        plt.cla()
        plt.plot(cx, cy, "-r", label="Ref")
        plt.plot(x_h, y_h, "-b", label="Traj")
        plt.plot(pred_x[0, :], pred_x[1, :], "xg", label="MPC Pred")
        plt.axis("equal")
        plt.legend()
        plt.pause(0.001)
        
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Horizon Effect

### Lab Objectives
1.  Run the simulation.
2.  **Observation:** The green crosses (Prediction) hug the red line (Reference). The blue line (Actual) follows smoothly.
3.  **Experiment A:** Set `T = 2` (Short Horizon).
    -   *Result:* The car might cut corners or oscillate. It's "shortsighted".
4.  **Experiment B:** Set `T = 20` (Long Horizon).
    -   *Result:* Better tracking, but slower computation.
5.  **Experiment C:** Increase `R` (Control Cost).
    -   *Result:* The car turns very slowly, potentially missing the curve.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Optimization Fails (Infeasible)
**Symptom:** `prob.status` is `INFEASIBLE`.
**Cause:** Constraints are too tight. E.g., asking to reach 100m in 1s with max speed 1m/s.
**Solution:** Relax constraints (Soft Constraints) or check initial conditions.

#### 2. Instability
**Symptom:** Car wobbles.
**Cause:** Model mismatch (Linearization error) or Time Step `DT` too large.
**Solution:** Reduce `DT` or use Non-Linear MPC (NMPC).

---

## ⚡ Optimization & Best Practices

### 1. Warm Start
Optimization is iterative.
Instead of starting from 0, start with the solution from the *previous* time step (shifted by 1).
-   Drastically reduces computation time.

### 2. Soft Constraints
Instead of `v < v_max` (Hard), use `v < v_max + slack`.
-   Add `slack^2` to the cost function.
-   Ensures the solver always finds a solution, even if it has to slightly violate the limit (better than crashing).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is MPC better than PID?
    *   **A:** MPC handles constraints explicitly (e.g., "Don't hit the wall", "Don't turn wheel > 45 deg"). PID doesn't know about limits until it hits them (saturation).
2.  **Q:** What is the "Receding Horizon"?
    *   **A:** The concept of planning $N$ steps, executing 1, and then planning $N$ steps again from the new position.
3.  **Q:** Why do we need `cvxpy`?
    *   **A:** To solve the Quadratic Program (QP). We can't solve it analytically like a PID equation because of the inequality constraints.

### Challenge Task
**Task:** Obstacle Avoidance.
1.  Add a circular obstacle at (20, 0).
2.  Add a constraint: $(x - x_{obs})^2 + (y - y_{obs})^2 \ge r^2$.
3.  Note: This constraint is Non-Convex (The "hole" in the donut). Linear MPC can't handle it directly. You need to linearize the constraint or use NMPC.

---

## 📚 Further Reading & References
-   [CVXPY Documentation](https://www.cvxpy.org/)
-   [Model Predictive Control for Autonomous Vehicles (Udacity)](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)

---

**Day 60 Complete** | Phase 4: ADAS & Robotics Systems | Week 9: Path Planning & Navigation
