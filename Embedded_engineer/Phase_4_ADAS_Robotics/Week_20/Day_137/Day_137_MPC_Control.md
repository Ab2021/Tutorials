# Day 137: Model Predictive Control (MPC)
## Phase 4: ADAS & Robotics Systems | Week 20: Path Planning & Decision Making

---

> **📝 Day 137 Focus:**
> PID controllers react to the past (Error). **MPC** predicts the future. It solves an optimization problem: "What steering angle sequence will minimize my error over the next 2 seconds?" It is the gold standard for autonomous driving control.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the MPC problem (Cost Function, Constraints, Horizon).
2.  **Formulate** the Kinematic Bicycle Model as the prediction model.
3.  **Implement** a Linear MPC solver using `cvxpy`.
4.  **Simulate** path tracking with constraints (e.g., Max Steering Angle).
5.  **Compare** MPC vs PID.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Optimization:** Convex Optimization ($min f(x)$ s.t. $Ax \le b$).
-   **Vehicle Dynamics:** Bicycle Model.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `cvxpy`, `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Concept

At each time step $t$:
1.  **Measure** current state $x_t$.
2.  **Solve** optimal control problem for horizon $N$ (e.g., next 10 steps).
    -   Find $u_t, u_{t+1}, \dots, u_{t+N-1}$ that minimizes cost.
3.  **Apply** only the first input $u_t$.
4.  **Wait** for next step $t+1$ and repeat (**Receding Horizon**).

### 🔹 Part 2: The Model

$$ x_{k+1} = A x_k + B u_k $$
-   State $x$: $[x, y, v, \theta]$.
-   Input $u$: $[a, \delta]$ (Acceleration, Steering).
-   Linearize the non-linear bicycle model around the reference trajectory.

### 🔹 Part 3: The Cost Function

$$ J = \sum_{k=0}^{N-1} (x_k - x_{ref})^T Q (x_k - x_{ref}) + u_k^T R u_k $$
-   **Tracking Error:** Minimize distance to path ($Q$).
-   **Control Effort:** Minimize steering/gas usage ($R$).
-   **Smoothness:** Minimize change in input ($u_k - u_{k-1}$).

---

## 💻 Implementation: Linear MPC

**Scenario:**
-   Track a reference line.
-   Constraint: Steering angle $\in [-30^\circ, 30^\circ]$.

### 🛠️ Setup
Create `week20_day137` and `mpc_controller.py`.

```bash
mkdir -p ~/ros2_ws/src/week20_day137
cd ~/ros2_ws/src/week20_day137
pip install cvxpy
touch mpc_controller.py
```

### 👨‍💻 Code: MPC Solver

```python
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

class MPC:
    def __init__(self):
        self.dt = 0.1
        self.N = 10 # Horizon
        
        # State: [x, y, v, theta]
        # Input: [a, delta]
        
    def solve(self, curr_state, ref_traj):
        # ref_traj: [N+1, 4]
        
        # Variables
        x = cp.Variable((4, self.N + 1))
        u = cp.Variable((2, self.N))
        
        cost = 0
        constraints = []
        
        # Initial State Constraint
        constraints += [x[:, 0] == curr_state]
        
        for k in range(self.N):
            # Cost
            # 1. Tracking Error (x, y)
            cost += cp.sum_squares(x[0:2, k] - ref_traj[k, 0:2]) * 10.0
            
            # 2. Input Effort
            cost += cp.sum_squares(u[:, k]) * 0.1
            
            # 3. Smoothness (Jerk)
            if k > 0:
                cost += cp.sum_squares(u[:, k] - u[:, k-1]) * 1.0
                
            # Model Constraints (Linearized approx)
            # x_{k+1} = x_k + v*dt*cos(theta), etc.
            # Here we use a very simple linear model for demo: Double Integrator
            # x+ = x + v*dt
            # v+ = v + a*dt
            # This is NOT a full bicycle model (requires linearization), 
            # but sufficient to demonstrate CVXPY syntax.
            
            # Let's assume theta is small (Linearized around 0)
            # x_next = x + v * dt
            # y_next = y + v * theta * dt (Small angle approx)
            # v_next = v + a * dt
            # theta_next = theta + v/L * delta * dt
            
            v_k = ref_traj[k, 2] # Linearize around ref velocity
            theta_k = ref_traj[k, 3]
            
            # A and B matrices (Jacobians)
            # Simple 1D case for clarity:
            # x = x + v*dt
            # v = v + a*dt
            constraints += [x[0, k+1] == x[0, k] + x[2, k] * self.dt]
            constraints += [x[1, k+1] == x[1, k] + x[2, k] * 0.1 * self.dt] # Mock y dynamics
            constraints += [x[2, k+1] == x[2, k] + u[0, k] * self.dt]
            constraints += [x[3, k+1] == x[3, k] + u[1, k] * self.dt] # Mock theta dynamics
            
            # Input Constraints
            constraints += [cp.abs(u[0, k]) <= 2.0] # Max Accel
            constraints += [cp.abs(u[1, k]) <= 0.5] # Max Steering (rad)
            
        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(verbose=False)
        
        if prob.status == cp.OPTIMAL:
            return u[:, 0].value, x[:, :].value
        else:
            print("MPC Failed")
            return np.zeros(2), np.zeros((4, self.N+1))

def main():
    mpc = MPC()
    
    # Reference Trajectory (Straight line at y=0, v=10)
    T = 50
    ref_traj = np.zeros((T + mpc.N + 1, 4))
    for i in range(len(ref_traj)):
        ref_traj[i, 0] = i * 1.0 # x increases
        ref_traj[i, 1] = 5.0 # y target (Step response)
        ref_traj[i, 2] = 10.0 # v target
        
    # Simulation
    curr_state = np.array([0.0, 0.0, 0.0, 0.0]) # Start at 0,0
    
    history_x = []
    history_y = []
    
    print("Running MPC...")
    for t in range(T):
        # Get Ref Segment
        ref_segment = ref_traj[t:t+mpc.N+1]
        
        # Solve
        u_opt, x_pred = mpc.solve(curr_state, ref_segment)
        
        # Apply Control (Simulate Plant)
        # x+ = x + v*dt
        curr_state[0] += curr_state[2] * mpc.dt
        curr_state[1] += curr_state[2] * 0.1 * mpc.dt # Mock dynamics matching model
        curr_state[2] += u_opt[0] * mpc.dt
        curr_state[3] += u_opt[1] * mpc.dt
        
        history_x.append(curr_state[0])
        history_y.append(curr_state[1])
        
        # Plot Prediction
        # plt.plot(x_pred[0, :], x_pred[1, :], 'g--', alpha=0.5)
        
    # Plot Result
    plt.figure()
    plt.plot(history_x, history_y, 'b-', label='Trajectory')
    plt.plot(ref_traj[:T, 0], ref_traj[:T, 1], 'r--', label='Reference')
    plt.title("MPC Path Tracking")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Horizon Effect

### Lab Objectives
1.  Run the script.
2.  **Observation:** The Blue line smoothly converges to the Red line ($y=5$).
3.  **Experiment:**
    -   Set `N = 2` (Short Horizon).
    -   **Result:** The controller becomes "myopic". It might overshoot or oscillate because it doesn't see the curve coming.
    -   Set `N = 50` (Long Horizon).
    -   **Result:** Very smooth, but computation time increases drastically.
    -   **Lesson:** $N$ is a trade-off between performance and latency.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Infeasible Problem
**Symptom:** `prob.status == INFEASIBLE`.
**Cause:** Constraints are too tight. (e.g., "Reach 100m in 1s" with "Max Accel 1m/s^2").
**Solution:** Soft Constraints. Allow violation of constraints but penalize it heavily in the cost function.

#### 2. Oscillation
**Symptom:** Steering wobbles.
**Cause:** Cost weight on control effort ($R$) is too low.
**Solution:** Increase $R$ (penalize $u^2$ and $\Delta u^2$).

---

## ⚡ Optimization & Best Practices

### 1. Non-Linear MPC (NMPC)
Linear MPC is an approximation.
-   **NMPC:** Uses the full non-linear bicycle model.
-   **Solver:** IPOPT or CGMRES. Slower but accurate for aggressive driving (drifting).

### 2. Code Generation
Python `cvxpy` is slow (10-50ms).
-   **CVXGEN / ACADO:** Generate optimized C code for the specific problem structure.
-   **Speed:** < 1ms solve time.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is MPC better than PID?
    *   **A:** MPC handles constraints (e.g., max steering) and MIMO (Multiple Input Multiple Output) naturally. PID struggles with coupled systems.
2.  **Q:** What is "Receding Horizon"?
    *   **A:** Planning $N$ steps, executing 1, and replanning. It allows the controller to adapt to disturbances (Model Mismatch).
3.  **Q:** What is the "Reference Trajectory"?
    *   **A:** The path (from A* or Polynomials) that the MPC tries to follow.

### Challenge Task
**Task:** Obstacle Avoidance.
1.  Add a constraint: $x > 10 \implies y > 2$ (Virtual Wall).
2.  Observe the MPC plan a path around the "wall" before it even reaches it.

---

## 📚 Further Reading & References
-   [MPC for Autonomous Vehicles (Udacity)](https://github.com/udacity/CarND-MPC-Project)
-   [CVXPY Documentation](https://www.cvxpy.org/)

---

**Day 137 Complete** | Phase 4: ADAS & Robotics Systems | Week 20: Path Planning & Decision Making
