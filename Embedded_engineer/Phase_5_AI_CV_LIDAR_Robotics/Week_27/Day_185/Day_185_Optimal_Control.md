# Day 185: Optimal Control for Robotics
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 27: Advanced Control & Dynamics

---

> **📝 Content Creator Instructions:**
> Finding the "Best" path, not just "A" path.
> - **Focus:** Pontryagin's Minimum Principle, Hamilton-Jacobi-Bellman (HJB), and the Linear Quadratic Regulator (LQR).
> - **Code:** `lqr_inverted_pendulum.py`. Balancing a cart-pole using infinite-horizon LQR derived from scratch (Riccati Equation).
> - **Concept:** Cost Function Engineering.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Derive** the LQR controller $u = -Kx$ for a linear system.
2.  **Solve** the Algebraic Riccati Equation (ARE) numerically.
3.  **Explain** the trade-off between $Q$ (State penalty) and $R$ (Control penalty).
4.  **Apply** LQR to balance a nonlinear inverted pendulum (via linearization).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install control numpy matplotlib scipy
```

### Prior Knowledge
- State Space Representation (Ax + Bu).
- Linear Algebra (Eigenvalues).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Cost Function

We want to minimize:
$$ J = \int_{0}^{\infty} (x^T Q x + u^T R u) dt $$
*   $Q$: "How bad is error?" (High Q $\to$ Precision).
*   $R$: "How expensive is fuel?" (High R $\to$ Gentle).

### 🔹 Part 2: Hamilton-Jacobi-Bellman (HJB)

Optimization over time is recursive.
"The optimal path from A to C passes through B. Thus, path A->B must be optimal, and B->C must be optimal." (Bellman Principle).

### 🔹 Part 3: The Riccati Equation

For Linear systems, the optimal cost-to-go is quadratic: $V(x) = x^T P x$.
$P$ must satisfy:
$$ A^T P + P A - P B R^{-1} B^T P + Q = 0 $$
Solve for $P$. Then $K = R^{-1} B^T P$.

---

## 💻 Implementation: Cart-Pole LQR

We balance a pole on a cart. The system is nonlinear, so we linearize around the upright fixed point.

### 🛠️ Project Structure
```text
day185_lqr/
├── src/
│   ├── lqr_cartpole.py
│   └── riccati_solver.py
└── README.md
```

### 👨‍💻 Manual Riccati Solver (`src/riccati_solver.py`)

Solving ARE via Eigenvalue decomposition (Kleinman or Schur method is better, but Gradient Descent for learning). Let's use `scipy` for robustness but show the math.

```python
import numpy as np
import scipy.linalg

def solve_lqr(A, B, Q, R):
    """
    Solves Continuous Algebraic Riccati Equation (CARE).
    A^T P + P A - P B R^-1 B^T P + Q = 0
    Returns K (Gain Matrix)
    """
    # 1. Solve P using Scipy
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    
    # 2. Compute K
    # K = R^-1 B^T P
    R_inv = np.linalg.inv(R)
    K = R_inv @ B.T @ P
    
    return K, P
```

### 👨‍💻 Cart Pole Sim (`src/lqr_cartpole.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
from riccati_solver import solve_lqr
from scipy.integrate import odeint

class CartPole:
    def __init__(self):
        # Params
        self.M = 1.0  # Cart Mass
        self.m = 0.1  # Pole Mass
        self.L = 1.0  # Pole Length
        self.g = 9.81
        
    def dynamics(self, state, u):
        x, x_dot, theta, theta_dot = state
        force = u[0]
        
        # Nonlinear EOM
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        
        # From derived Lagrangian
        denom = self.L * (4.0/3.0 - self.m * cos_t**2 / (self.M + self.m))
        
        theta_acc = (self.g * sin_t - cos_t * (force + self.m * self.L * theta_dot**2 * sin_t)/(self.M + self.m)) / denom
        x_acc = (force + self.m * self.L * (theta_dot**2 * sin_t - theta_acc * cos_t)) / (self.M + self.m)
        
        return [x_dot, x_acc, theta_dot, theta_acc]
        
    def get_linearized_matrices(self):
        # Linearize around theta = 0 (Upright), x = 0, u = 0
        # State: [x, x_dot, theta, theta_dot]
        
        denom = 4.0/3.0 - self.m/(self.M + self.m)
        
        # Jacobian df/dx
        # x_acc approx ...
        # theta_acc approx g/L * theta / denom
        
        # Simplified A Matrix
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, -self.m*self.g/self.M, 0], # approx
            [0, 0, 0, 1],
            [0, 0, (self.M+self.m)*self.g/(self.M*self.L), 0] # approx
        ])
        
        # Refined Linearization (Standard CartPole A)
        p = self.M + self.m
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, -self.m*self.g/self.M, 0], # Note: this term depends on conventions
            [0, 0, 0, 1],
            [0, 0, p*self.g/(self.M*self.L*(4/3 - self.m/p)), 0] # Rough analyticals
        ])
        
        # Actually, let's substitute exact values for M=1, m=0.1, L=1
        # Denom approx 1.33
        # Theta_acc = g/L * theta / 1.33 = 7.5 * theta
        
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0], # Frictionless cart
            [0, 0, 0, 1],
            [0, 0, 10.0, 0] # Small angle approx
        ])
        
        B = np.array([
            [0],
            [1.0/self.M],
            [0],
            [1.0/(self.M*self.L)] # approx
        ])
        
        return A, B

def simulate_system(sys, K, t_span):
    def control_loop(state, t):
        # u = -Kx
        # Error is state - goal (0,0,0,0)
        u = -K @ state
        u = np.clip(u, -10, 10) # Motor limits
        res = sys.dynamics(state, u)
        return res
        
    state_0 = [0, 0, 0.2, 0] # Start slightly tipped (0.2 rad)
    traj = odeint(control_loop, state_0, t_span)
    return traj

def main():
    sys = CartPole()
    A, B = sys.get_linearized_matrices()
    
    # Design Weights
    Q = np.diag([1.0, 1.0, 10.0, 1.0]) # Penalize Theta heavily
    R = np.eye(1) * 0.1 # Cheap control
    
    print("Computing LQR Gain...")
    K, P = solve_lqr(A, B, Q, R)
    print(f"K: {K}")
    
    t = np.linspace(0, 10, 1000)
    traj = simulate_system(sys, K, t)
    
    plt.plot(t, traj[:, 2], label='Theta (rad)')
    plt.plot(t, traj[:, 0], label='Cart Pos (m)')
    plt.legend()
    plt.grid()
    plt.title("LQR Cart-Pole Stabilization")
    plt.xlabel("Time (s)")
    plt.savefig('lqr_response.png')
    print("Done.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Tuning Wars"

### 1. Lab Objectives
- **Run:** `lqr_cartpole.py`. Observe recovery from 0.2 rad.
- **Fail:** Set initial angle to `3.0` rad (~180 deg, pendant). Observe LQR fail.
    *   Why? Linearization is invalid at 180 deg. The "Local" optimal policy is correct for Top, but not Bottom.
- **Tune:** Increase $Q_{pos}$. Observe robot prioritizing cart position over angle (might drop the pole).

---

## 🚀 Project: "Double Inverted Pendulum"

**Goal:** Balance a stick on a stick.
1.  **Matrices:** A is 6x6. B is 6x1.
2.  **Solver:** Are Eigenvalues controllable? (Check Controllability Matrix).
3.  **Result:** It works in sim, but in real life, actuator bandwidth limits stability.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Unstable System"
*   **Cause:** A Matrix has positive eigenvalues (Unstable poles).
*   **Fix:** LQR *guarantees* stability if (A,B) is controllable. If it fails, check your B matrix signs.

#### 2. "Oscillation"
*   **Cause:** $Q$ is huge relative to $R$. The controller slams the actuator to fix error instantly $\to$ High-frequency chatter.
*   **Fix:** Increase $R$.

---

## ⚡ Optimization: Iterative LQR (iLQR)

For nonlinear systems:
1.  Linearize trajectory around nominal.
2.  Solve LQR for $\delta u$.
3.  Update nominal.
4.  Repeat.
*   Allows swinging up the pendulum (Nonlinear) then balancing it (Linear).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What happens if $R \to 0$?
    *   **A:** Infinite control authority. The system moves instantly to the goal (Dirac delta force). In reality, actuators saturate/break.
2.  **Q:** Is LQR robust?
    *   **A:** It has infinite gain margin and 60 deg phase margin (for SISO). Very robust *modeled* dynamics. Not robust to unmodeled dynamics.

### Challenge Task
> **Task:** "Swing Up".
> 1. Use Energy Shaping ($E = E_{pot} + E_{kin}$).
> 2. Controller: $u = k(E - E_{target}) \dot{\theta}$.
> 3. Switch to LQR when $|\theta| < 0.2$.

---

## 📚 Further Reading
- **Steve Brunton:** "Control Bootcamp" (YouTube).
- **Underactuated Robotics:** Russ Tedrake (MIT).

---

## 🔗 External Resources
### 📜 Open Source Libraries
- [AtsushiSakai/PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) - LQR and MPC implementations in Python.
- [optimal_control_examples](https://github.com/andrespulido8/optimal_control_examples) - Direct Collocation and Single Shooting examples.

### 📺 Video Tutorials
- [CMU 16-745: Optimal Control](https://www.youtube.com/results?search_query=CMU+16-745+Optimal+Control) - Full lecture series by Prof. Zac Manchester.
- [Optimal Control Trajectory Optimization](https://www.youtube.com/results?search_query=Optimal+Control+Trajectory+Optimization) - Tutorial by Alphonsus Adu-Bredu.

---

**Day 185 Complete**
