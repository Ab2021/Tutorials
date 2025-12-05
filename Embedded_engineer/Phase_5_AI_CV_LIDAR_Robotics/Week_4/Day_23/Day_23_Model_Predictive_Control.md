# Day 23: Model Predictive Control (MPC)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 4: Advanced Control & Dynamics

---

> **📝 Content Creator Instructions:**
> LQR is elegant but doesn't respect limits. Robots have max voltage, max torque, and walls they shouldn't hit.
> - **Focus:** Receding Horizon Control, Constrained Optimization, and Quadratic Programming.
> - **Code:** Implementation of Linear MPC using `cvxpy` / `osqp`.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Infinite Horizon (LQR) and Finite Horizon (MPC) control.
2.  **Formulate** a generic MPC optimization problem with State and Input constraints.
3.  **Implement** a Linear MPC controller for a vehicle model.
4.  **Analyze** the computational cost vs. control frequency trade-off.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install cvxpy numpy matplotlib osqp
```

### Prior Knowledge
- LQR (Day 22).
- Convex Optimization (Min Convex Function over Convex Set).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Receding Horizon

We solve for the optimal control sequence $U = \{u_0, u_1, ..., u_{N-1}\}$ for the next $N$ steps.
1.  Measure current state $x_t$.
2.  Solve optimization problem.
3.  Apply **only** $u_0$.
4.  At $t+1$, measure $x_{t+1}$ and repeat.

*   *Why?* Feedback. If our model was perfect, we could execute the whole sequence. Since models drift, re-planning every step corrects errors.

### 🔹 Part 2: The Optimization Problem

$$ \min_{U} \sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) + x_N^T P x_N $$

Subject to:
1.  **Dynamics:** $x_{k+1} = A x_k + B u_k$
2.  **Input Limits:** $u_{min} \le u_k \le u_{max}$ (e.g., -10V to +10V).
3.  **State Limits:** $x_{min} \le x_k \le x_{max}$ (e.g., Stay within lane $\pm 2m$).

### 🔹 Part 3: Solving via QP

Both the Cost (Quadratic) and Constraints (Linear) describe a **Quadratic Program (QP)**.
$$ \min \frac{1}{2} z^T H z + q^T z $$
$$ s.t. \ lb \le C z \le ub $$
*   Solvers like **OSQP** (Operator Splitting Quadratic Program) are designed for embedded real-time use (kHz rates).

---

## 💻 Implementation: Adaptive Cruise Control (ACC)

Goal: Maintain a safe distance $d_{safe}$ from the leading car, but don't exceed $v_{limit}$.

### 🛠️ Project Structure
```text
day23_mpc/
├── src/
│   ├── mpc_controller.py
│   └── vehicle_model.py
└── run_acc.py
```

### 👨‍💻 Code Implementation (`src/mpc_controller.py`)

```python
import cvxpy as cp
import numpy as np

class LinearMPC:
    def __init__(self, A, B, N=10):
        self.A = A
        self.B = B
        self.N = N
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        
    def solve(self, x0, ref):
        # Variables
        x = cp.Variable((self.nx, self.N + 1))
        u = cp.Variable((self.nu, self.N))
        
        cost = 0
        constraints = [x[:, 0] == x0]
        
        Q = np.diag([10.0, 1.0]) # [dist_error, vel_error]
        R = np.eye(self.nu) * 0.1
        
        # Max Accel/Decel
        u_max = 5.0 
        u_min = -5.0
        
        for k in range(self.N):
            # Cost
            error = x[:, k] - ref[:, k]
            cost += cp.quad_form(error, Q) + cp.quad_form(u[:, k], R)
            
            # Constraints
            constraints += [x[:, k+1] == self.A @ x[:, k] + self.B @ u[:, k]]
            constraints += [u[:, k] <= u_max]
            constraints += [u[:, k] >= u_min]
            
        # Terminal Cost (Stability)
        cost += cp.quad_form(x[:, self.N] - ref[:, self.N], Q * 10)
        
        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP)
        
        if prob.status != cp.OPTIMAL:
            print("OSQP Failed!")
            return np.zeros(self.nu)
            
        return u[:, 0].value
```

---

## 🔬 Lab Exercise: Lane Keeping

### 1. Lab Objectives
- Model lateral vehicle dynamics (bicycle model linearized).
- Constraint: $y \in [-2, 2]$ (Road width).
- Situation: High initial error ($y=5$).
- **Observe:**
    - LQR would turn smoothly but might overshoot or act aggressively.
    - MPC will turn as hard as allowed ($u_{max}$) until it hits the lane constraint, then smooth out.

### 2. Step-by-Step Guide
1.  Define Dynamics: State = $[y, v_y, \psi, \dot{\psi}]$.
2.  Define Constraints: Steering angle $\delta \in [-0.5, 0.5]$ rad.
3.  Simulate: Start robot at $y=3$ (off-road).
4.  Plot control input $u$. It should "clip" at the limit (Saturation).

---

## 🚀 Project: "Race Car Controller"

**Scenario:** We want to drive a lap as fast as possible.
**Model:** Kinematic Bicycle Model.
**Constraints:**
- Friction Circle: $a_{lat}^2 + a_{long}^2 \le \mu g$.
- Track Boundaries: Map constraints.

**Implementation (Linear Time-Varying MPC):**
1.  Take a reference path.
2.  Linearize dynamics *around the reference path* at each step $k$.
    $$ x_{k+1} \approx f(x_{ref}, u_{ref}) + A_k (x-x_{ref}) + B_k (u-u_{ref}) $$
3.  Solve QP.
4.  This handles the non-linearity of the car at high speeds.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Infeasible Problem"
*   **Symptom:** Solver returns `None` or Status `INFEASIBLE`.
*   **Cause:** Limits are impossible to satisfy. (e.g., "Be at 100m in 1sec" with max speed 1m/s).
*   **Fix:** **Soft Constraints**. Instead of $x < 5$, use $x < 5 + \epsilon$ and penalize $\epsilon$ heavily ($1000 \epsilon^2$) in the cost. This allows violation in emergencies.

#### 2. "Oscillation"
*   **Cause:** Horizon $N$ is too short. The controller acts greedy because it can't see the curve coming.
*   **Fix:** Increase $N$. Or increase Terminal Cost $P$.

---

## ⚡ Optimization: Code Generation

Parsing python expressions to C++ matrices is slow.
**CVXGEN / OSQP-CodeGen:**
*   Takes your problem description.
*   Generates optimized, branch-free C code.
*   Execution time drops from 10ms (Python) to 0.1ms (C).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is MPC slower than LQR?
    *   **A:** LQR is a static matrix multiplication ($u=-Kx$). MPC solves an optimization problem iteratively *every single timestep*.
2.  **Q:** What is the "Terminal Cost"?
    *   **A:** A cost on the final state $x_N$. It approximates "The cost of the infinite future from here", ensuring stability even with finite horizons.

### Challenge Task
> **Task:** Obstacle Avoidance with MPC.
> 1. Add a constraint: $-(x - x_{obs})^2 - (y - y_{obs})^2 \le -r^2$.
> 2. Note: This assumes a convex constraint. Obstacles are non-convex (holes).
> 3. Use "Slack Variables" or linearize the obstacle constraint as a half-plane.

---

## 📚 Further Reading
- **Model Predictive Control:** Borrelli, Bemporad, Morari (The Book).
- **OSQP:** osqp.org.
- **Acados:** Fast solvers for Non-Linear MPC.

---

**Day 23 Complete**
