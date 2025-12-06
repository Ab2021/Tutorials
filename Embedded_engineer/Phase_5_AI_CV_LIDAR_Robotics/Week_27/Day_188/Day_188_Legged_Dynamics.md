# Day 188: Legged Robot Dynamics
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 27: Advanced Control & Dynamics

---

> **📝 Content Creator Instructions:**
> Four legs, one brain.
> - **Focus:** Floating Base Dynamics, Centroidal Dynamics, The SRB (Single Rigid Body) Approximation.
> - **Code:** `quadruped_mpc.py` (Simplified convex MPC formulation). Optimizing Ground Reaction Forces (GRF) to track a body trajectory.
> - **Concept:** Friction Cones and QP Solvers.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Derive** the Floating Base Dynamics equation ($M(q)\dot{v} + C = \tau + J^T \lambda$).
2.  **Explain** the Centroidal Dynamics simplification (The "Potato" Model).
3.  **Formulate** a Quadratic Program (QP) to solve for optimal Foot Forces.
4.  **Simulate** a Standing Balance Controller (Virtual Spring-Damper).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy cvxpy
```

### Prior Knowledge
- Convex Optimization.
- Rigid Body Dynamics (Newton-Euler).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Floating Base Dynamics

Unlike an arm fixed to the ground, a dog robot floats.
*   **Base DoF:** 6 (Pos + Orientation).
*   **Joint DoF:** 12 (3 per leg).
*   **Underactuated:** Can't directly apply torque to the Base. We can only push the ground.

### 🔹 Part 2: Structure of Locomotion Control

1.  **State Estimator:** Where am I? (IMU + Kinematics).
2.  **Gait Planner:** Where to put feet? (Trot, Gallop).
3.  **ToS (Trajectory Optimization):** How to move the CoM?
4.  **WBC (Whole Body Control):** Convert CoM force to Joint Torques.

### 🔹 Part 3: The SRB (Single Rigid Body) Model

Assumption: Legs are massless. All mass is in the Body.
Equation:
$$ M \ddot{p}_{com} = \sum_{i=1}^4 f_i + M g $$
$$ I \dot{\omega} = \sum_{i=1}^4 (r_i \times f_i) $$
*   Linear relationship between Foot Forces $f_i$ and Body Acceleration.
*   We can use Convex MPC!

---

## 💻 Implementation: Standing Controller

We assume 4 feet are on the ground. We want to keep the body at height $h$.
We calculate required forces using a QP solver (`cvxpy`).

### 🛠️ Project Structure
```text
day188_legged/
├── src/
│   ├── quadruped_balance.py
│   └── qp_solver.py
└── README.md
```

### 👨‍💻 QP Foot Force Solver (`src/qp_solver.py`)

Using `cvxpy` (CVX for Python).
Minimize $\| F_{total} - F_{des} \|^2$ subject to Friction Cone.

```python
import cvxpy as cp
import numpy as np

class ForceOptimizer:
    def __init__(self, mu=0.5):
        self.mu = mu # Friction coeff
        
    def solve_forces(self, mass, I, r_feet, F_des, M_des):
        """
        r_feet: List of 4 arrays [rx, ry, rz] (Pos relative to CoM)
        F_des: Desired Force [Fx, Fy, Fz]
        M_des: Desired Torque [Mx, My, Mz]
        """
        # Variables: 4 forces (x 3 dims) = 12 vars
        f = cp.Variable((4, 3))
        
        # Constraints
        constr = []
        total_F = np.zeros(3)
        total_M = np.zeros(3)
        
        for i in range(4):
            # 1. Friction Cone Pyramid Approximation
            # |fx| <= mu * fz
            # |fy| <= mu * fz
            # fz >= 0
            constr += [cp.abs(f[i, 0]) <= self.mu * f[i, 2]]
            constr += [cp.abs(f[i, 1]) <= self.mu * f[i, 2]]
            constr += [f[i, 2] >= 0]
            
            # Summation
            # We can't sum CVX variables in loop easily without list comp
            pass 
            
        # Re-formulate for Vectorization
        # F_sum = Sum(f_i)
        # M_sum = Sum(cross(r_i, f_i))
        
        # Cross product matrix for r_i
        def cross_mat(r):
            return np.array([
                [0, -r[2], r[1]],
                [r[2], 0, -r[0]],
                [-r[1], r[0], 0]
            ])
            
        A_force = np.tile(np.eye(3), (1, 4)) # [I I I I]
        A_moment = np.hstack([cross_mat(r_feet[i]) for i in range(4)])
        
        A = np.vstack([A_force, A_moment]) # 6x12 Matrix
        b = np.concatenate([F_des, M_des]) # 6x1 vector
        
        f_vec = cp.reshape(f, (12, 1))
        
        # Objective: Tracking + Regularization (Min forces)
        # alpha * ||Ax - b||^2 + beta * ||x||^2
        cost = cp.sum_squares(A @ f_vec - b) * 10.0 + cp.sum_squares(f_vec) * 0.01
        
        prob = cp.Problem(cp.Minimize(cost), constr)
        prob.solve(solver=cp.ECOS)
        
        return f.value
```

### 👨‍💻 Balance Sim (`src/quadruped_balance.py`)

A virtual PD controller drives the desired body acceleration.
$F_{des} = K_p (p_{target} - p) + K_d (\dot{p}_{target} - \dot{p}) + Mg$.

```python
import numpy as np
from qp_solver import ForceOptimizer

def run_sim():
    opt = ForceOptimizer()
    
    # Robot Metrics (Mini Cheetah scale)
    mass = 9.0 # kg
    inertia = np.eye(3) * 0.1
    g = np.array([0, 0, 9.81])
    
    # State
    pos = np.array([0.0, 0.0, 0.25]) # Lowered
    vel = np.zeros(3)
    rpy = np.zeros(3)
    
    # Target
    pos_des = np.array([0.0, 0.0, 0.3]) # Stand Height
    
    # Gains
    Kp = 100.0
    Kd = 10.0
    
    # Feet (Relative to CoM)
    # FL, FR, RL, RR
    feet_rel = [
        np.array([ 0.2,  0.1, -0.3]),
        np.array([ 0.2, -0.1, -0.3]),
        np.array([-0.2,  0.1, -0.3]),
        np.array([-0.2, -0.1, -0.3])
    ]
    
    print("Simulating Balance control...")
    
    for t in range(5):
        # 1. Virtual Model Control (Compute Desired Wrench)
        # F = ma
        acc_des = Kp * (pos_des - pos) + Kd * (0 - vel)
        F_des = mass * (acc_des + g)
        M_des = np.zeros(3) # Keep body level
        
        # 2. Solve QP
        forces = opt.solve_forces(mass, inertia, feet_rel, F_des, M_des)
        
        if forces is None:
            print("QP Failed")
            break
            
        print(f"T={t}: F_des_Z={F_des[2]:.1f}N -> Solved Sum_Fz={np.sum(forces[:,2]):.1f}N")
        # In full sim, we would apply these forces and integrate equations of motion.

if __name__ == "__main__":
    run_sim()
```

---

## 🔬 Lab Exercise: "Push the Dog"

### 1. Lab Objectives
- **Modify:** Add an external disturbance force $F_{ext} = [50N, 0, 0]$ (Kick to side) at $t=2$.
- **Observe:** The VMC (Virtual Model Controller) generates a counter-force. The QP distributes this to the feet.
    *   To push back Left, the Right feet push harder? Or angling the force vectors?
- **Fail:** Push too hard ($> 200N$). The Friction Cone constraint limits the reaction force. The QP saturates. The robot falls.

---

## 🚀 Project: "Trot Gait"

**Goal:** Dynamic locomotion.
1.  **Gait:** Switch feet [FL, RR] and [FR, RL] every 0.25s.
2.  **MPC:** Prediction Horizon = 10 steps.
3.  **Solver:** Use `qpOASES` (C++) or an OSQP binding for speed (1kHz).
4.  **Swing:** Benzier curve for foot tip trajectory.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Slipping feet"
*   **Cause:** $\mu$ in optimizer is 0.8, but real floor is 0.3.
*   **Fix:** Be conservative. Set $\mu_{opt} = 0.4$.

#### 2. "QP Infeasible"
*   **Cause:** Desired torque is impossible (e.g., maintain balance while only 2 feet on one side are on ground - CoM outside support line).
*   **Fix:** Relax constraints (Soft constraints) or use a better planner (Divergent Component of Motion) to move CoM *before* lifting leg.

---

## ⚡ Optimization: Condensed vs Sparse QP

*   **Sparse:** Keep all variables $x_0 \dots x_N, u_0 \dots u_N$. Matrix is huge (banded). Good for long horizons.
*   **Condensed:** Eliminate $x$ using dynamics. Only optimize $u$. Matrix is dense but small. Good for short horizons.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we minimize $F^2$ in the cost?
    *   **A:** To distribute load evenly. Keeps forces well inside friction cones (away from edges) and reduces motor heat ($I^2 R$ losses).
2.  **Q:** What is ZMP (Zero Moment Point)?
    *   **A:** The point on the ground where the Net Moment is zero (tipping point). For stability, ZMP must be inside the Convex Hull of support polygon.

### Challenge Task
> **Task:** "Backflip".
> 1. Requires Full Body Dynamics (SRB assumption breaks because legs swing fast).
> 2. Requires Pre-Choreographed Trajectory optimization (Day 185) offline.
> 3. Replay on hardware with high-gain PD.

---

## 📚 Further Reading
- **MIT Cheetah 3 Paper:** "Model Predictive Control for Multi-Legged Robot".
- **Cheetah Software (MIT):** Open source controller logic.

---

## 🔗 External Resources
### 📜 Open Source Libraries
- [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) - Isaac Gym environments for training legged locomotion.
- [qiayuanl/legged_control](https://github.com/qiayuanl/legged_control) - OCS2 and ROS 2 Control implementation for Unitree robots.

### 📺 Video Tutorials
- [Optimization-based Control of Legged Robots](https://www.youtube.com/results?search_query=Optimization-based+Control+of+Legged+Robots) - 2023 Course on multi-body dynamics.
- [Gait Optimization for Legged Robots](https://www.youtube.com/results?search_query=Gait+Optimization+for+Legged+Robots) - Trajectory generation techniques.

---

**Day 188 Complete**
