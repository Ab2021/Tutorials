# Day 87: Whole-Body Control (WBC)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 13: Humanoid Robotics

---

> **📝 Content Creator Instructions:**
> Don't move just the arm. Move everything.
> - **Focus:** Inverse Dynamics, Null-Space Projection, and Optimization-based Control (QP).
> - **Code:** A Python script using `cvxpy` (or `osqp`) to solve a simple Inverse Kinematics task with torque constraints and singularity avoidance.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Formulate** robotics problems as Optimization Programs ($\min ||Ax - b||^2$).
2.  **Explain** Hierarchical Control: Priority 1 (Balance) > Priority 2 (Manipulation) > Priority 3 (Posture).
3.  **Implement** a QP Solver to resolve conflicting tasks while satisfying constraints (Torque/Joint Limits).
4.  **Differentiate** between Inverse Kinematics (IK) and Whole-Body Control (WBC).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- CPU sufficient.

### Software Environment
```bash
pip install cvxpy numpy matplotlib pinocchio
```

### Prior Knowledge
- Jacobians ($J$).
- Convex Optimization Basics.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Limits of Decoupled Control

*   **Standard Way:** 
    *   Leg Controller (Walking).
    *   Arm Controller (Manipulation).
*   **The Problem:** If the Arm reaches far Right, the CoM shifts. The Leg Controller doesn't know. The robot falls.
*   **WBC Standard:** One central controller manages ALL joints to satisfy ALL tasks (CoM, Hand Pose, Feet Contact).

### 🔹 Part 2: Task Hierarchy (Nullspace)

How to fulfill Task 2 without disturbing Task 1?
$$ \dot{q} = J_1^\dagger v_1 + (I - J_1^\dagger J_1) \dot{q}_{null} $$
*   $J_1^\dagger$: Pseudo-inverse of Task 1 Jacobian.
*   $(I - J_1^\dagger J_1)$: Nullspace Projector.
*   $\dot{q}_{null}$: Velocities for Task 2.
*   **Result:** The legs keep balance. The arms move in the *Nullspace* of the balance task.

### 🔹 Part 3: Quadratic Programming (QP)

Instead of analytical Pseudo-inverses (which handle inequality constraints poorly), we use solvers.
$$ \min_{\tau, \dot{q}, \ddot{q}} \sum w_i || Task_i ||^2 $$
Subject to:
1.  **Dynamics:** $M \ddot{q} + C + G = \tau + J_c^T F_c$.
2.  **Friction Cones:** $F_c \in \mathcal{K}$ (No slipping).
3.  **Limits:** $\tau_{min} < \tau < \tau_{max}$.

---

## 💻 Implementation: QP-based Inverse Kinematics

We will use `cvxpy` to solve a redundant IK problem with joint limits.
*   **Task:** End-effector to Target.
*   **Constraint:** Joint 2 must stick to [0, 1.0].
*   **Secondary Task:** Minimize Joint Velocity (Damping).

### 🛠️ Project Structure
```text
day87_wbc/
├── src/
│   ├── simple_ik_qp.py
│   └── robot_model.py (Mock Jacobian)
└── output/
    └── velocity_profile.png
```

### 👨‍💻 QP Solver (`src/simple_ik_qp.py`)

A 3-DOF Planar Arm trying to reach $(x,y)$.

```python
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

class PlanarArmQP:
    def __init__(self):
        # 3 Links, length 1.0 each
        self.L = [1.0, 1.0, 1.0]
        self.q = np.array([0.1, 0.1, 0.1]) # Initial Config
        self.dt = 0.01

    def forward_kinematics(self, q):
        x = self.L[0]*np.cos(q[0]) + self.L[1]*np.cos(q[0]+q[1]) + self.L[2]*np.cos(q[0]+q[1]+q[2])
        y = self.L[0]*np.sin(q[0]) + self.L[1]*np.sin(q[0]+q[1]) + self.L[2]*np.sin(q[0]+q[1]+q[2])
        return np.array([x, y])

    def get_jacobian(self, q):
        # Analytical Jacobian for 3-Link Planar
        # x = l1 c1 + l2 c12 + l3 c123
        # dx/dq1 = -l1 s1 - l2 s12 - l3 s123
        s1 = np.sin(q[0])
        c1 = np.cos(q[0])
        s12 = np.sin(q[0]+q[1])
        c12 = np.cos(q[0]+q[1])
        s123 = np.sin(q[0]+q[1]+q[2])
        c123 = np.cos(q[0]+q[1]+q[2])
        
        l1, l2, l3 = self.L
        
        J = np.zeros((2, 3))
        # Row X
        J[0,0] = -l1*s1 - l2*s12 - l3*s123
        J[0,1] =        - l2*s12 - l3*s123
        J[0,2] =                   - l3*s123
        # Row Y
        J[1,0] =  l1*c1 + l2*c12 + l3*c123
        J[1,1] =          l2*c12 + l3*c123
        J[1,2] =                     l3*c123
        return J

    def solve_step(self, target_pos):
        # Variables: Joint Velocities
        dq = cp.Variable(3)
        
        # Current Pos
        current_pos = self.forward_kinematics(self.q)
        error = target_pos - current_pos
        
        # Desired End Effector Velocity (P-Gain)
        v_des = 10.0 * error
        if np.linalg.norm(v_des) > 1.0: v_des = v_des / np.linalg.norm(v_des) # Clamp
        
        J = self.get_jacobian(self.q)
        
        # Objective: Min || J dq - v_des ||^2 + lambda || dq ||^2
        cost = cp.sum_squares(J @ dq - v_des) + 0.1 * cp.sum_squares(dq)
        
        # Constraints
        constraints = []
        # Joint Limits (Velocity)
        constraints += [dq <=  5.0]
        constraints += [dq >= -5.0]
        
        # Joint Limits (Position) - Convert to velocity constraints
        # q_next = q + dq * dt
        # q_min <= q + dq * dt <= q_max
        q_min = np.array([-np.pi, -np.pi, -2.0]) # Limit joint 3
        q_max = np.array([ np.pi,  np.pi,  2.0])
        
        constraints += [self.q + dq * self.dt <= q_max]
        constraints += [self.q + dq * self.dt >= q_min]
        
        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        
        if dq.value is None:
            print("Infeasible!")
            return np.zeros(3)
        return dq.value

# Simulation Loop
solver = PlanarArmQP()
target = np.array([1.5, 1.5])
history = []

for t in range(100):
    dq = solver.solve_step(target)
    solver.q += dq * solver.dt
    history.append(solver.q.copy())

# Plotting...
```

---

## 🔬 Lab Exercise: "The Impossible Reach"

### 1. Lab Objectives
- Set a target *outside* the workspace (Result: Arm stretches fully).
- Set a constraint: Joint 1 Locked ($dq_1 = 0$).
- **Observe:** The solver automatically uses Joints 2 and 3 to get as close as possible.
- **Compare:** Standard Inverse Jacobian ($J^{-1}$) would crash or return Infinite velocities at singularity. QP handles it gracefully (by minimizing error rather than enforcing zero error).

---

## 🚀 Project: "Standing Balance"

**Goal:** WBC for a 2-Link Leg.
1.  **Task 1 (Priority High):** Keep CoM over ankle.
2.  **Task 2 (Priority Low):** Move Hip Down (Squat).
3.  **Implementation:**
    *   Formulate QP.
    *   If squatting pushes CoM too far, the CoM task wins (Robot stops squatting rather than falling).
    *   This is the core of Safety in WBC.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Solver Slow"
*   **Cause:** Creating the `cp.Problem` inside the loop is slow (Parsing overhead).
*   **Fix:** Use `parameters` in CVXPY or switch to `OSQP-Eigen` (C++) for kHz loops.

#### 2. "Constraint Fighting"
*   **Symptom:** Infeasible.
*   **Cause:** Joint Limits vs Task Requirements.
*   **Fix:** Use "Slack Variables". $J \dot{q} = v_{des} + \delta$. Minimize $||\delta||$. This ensures the problem is always feasible (Soft Constraint).

---

## ⚡ Optimization: TSID (Task Space Inverse Dynamics)

Pinocchio TSID library.
*   Handles the Rigid Body Dynamics equation automatically.
*   Real-time capable (1 kHz).
*   Used on TALOS, HRP-2, and major research humanoids.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is Jacobian Inverse insufficient for humanoids?
    *   **A:** It implies we control velocities directly. In reality, we control Torques ($M\ddot{q} \dots$). Also, it doesn't handle Inequality Constraints (Limits).
2.  **Q:** What is a "Strict Hierarchy"?
    *   **A:** Task 2 cannot degrade Task 1 at all.
3.  **Q:** What is "Weighted Hierarchy"?
    *   **A:** Sum of weighted errors ($w_1 E_1 + w_2 E_2$). Task 1 might suffer slightly to help Task 2 massively.

### Challenge Task
> **Task:** Singularity Damping.
> 1. Move arm to full extension.
> 2. Analytical $J^{-1}$ explodes.
> 3. QP adds Damping term ($\lambda ||\dot{q}||^2$).
> 4. Result: Robot slows down near singularity instead of shaking.

---

## 📚 Further Reading
- **QP-OASES:** Active Set solver for robotics.
- **Pinocchio:** Fast Rigid Body Dynamics.

---

**Day 87 Complete**
