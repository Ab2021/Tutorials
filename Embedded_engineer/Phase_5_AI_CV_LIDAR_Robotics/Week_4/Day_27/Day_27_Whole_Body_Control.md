# Day 27: Whole-Body Control (Humanoids)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 4: Advanced Control & Dynamics

---

> **📝 Content Creator Instructions:**
> Humanoids are "Underactuated" (Floating Base). We cannot control position directly, only through contact forces.
> - **Focus:** Whole-Body Control (WBC), Floating Base Dynamics, and Hierarchical QP.
> - **Code:** Implementation of Inverse Dynamics for a floating-base robot.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Derive** Floating Base Dynamics ($M(q)\ddot{q} + C + g = S^T \tau + J^T F_{contact}$).
2.  **Explain** the concept of Virtual Model Control (VMC) and Operational Space Control (OSC).
3.  **Implement** a QP-based Whole-Body Controller that prioritizes Balance over Hand Motion.
4.  **Simulate** a humanoid balancing on one leg while waving.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install pinocchio cvxpy numpy # Pinocchio is best for Robot Dynamics
```

### Prior Knowledge
- Rigid Body Dynamics (Lagrangian).
- Friction Cones.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Floating Base Dynamics

A robot arm is bolted to the ground. A Humanoid is distinct:
$$ q = [p_{base}, R_{base}, q_{joints}]^T $$
*   6 unactuated DOFs (Base position/orientation).
*   $n$ actuated DOFs (Joints).

$$ M \ddot{q} + h(q, \dot{q}) = S^T \tau + J_c^T f_c $$
*   $\tau$: Motor Torques.
*   $f_c$: Contact Forces (Feet on ground).
*   $S^T$: Selection matrix (zeros for base, ones for joints).

**The Challenge:** To move the base (CoM), we must push against the ground ($f_c$).

### 🔹 Part 2: Task Hierarchy (Stack of Tasks)

We have multiple conflicting goals:
1.  **Priority 0 (Safety):** Keep Feet flat on ground. Keep Friction inside Cone.
2.  **Priority 1 (Balance):** Keep Center of Mass (CoM) over polygon of support.
3.  **Priority 2 (Manipulation):** Move Hand to target.
4.  **Priority 3 (Posture):** Keep knees bent naturally.

**Null Space Projection:**
Task 2 operates in the *Null Space* of Task 1.
$$ \tau = \tau_1 + N_1 \tau_2 $$
This ensures Task 2 does not disturb Task 1.

### 🔹 Part 3: QP Formulation

We solve for $(\ddot{q}, \tau, f_c)$ simultaneously using Quadratic Programming.
$$ \min w_1 || \ddot{x}_{com} - \ddot{x}_{des} ||^2 + w_2 || \ddot{x}_{hand} - \ddot{x}_{target} ||^2 $$
Subject to:
1.  **Dynamics:** $M \ddot{q} + h = S^T \tau + J_c^T f_c$
2.  **Contact:** $f_c$ inside Friction Cone ($\mu$).
3.  **Limits:** $\tau_{min} \le \tau \le \tau_{max}$.

---

## 💻 Implementation: Inverse Dynamics QP

We will use a simplified 2D Walker model.

### 🛠️ Project Structure
```text
day27_wbc/
├── src/
│   ├── robot_model.py
│   ├── wbc_solver.py
└── run_balance.py
```

### 👨‍💻 Code Implementation (`src/wbc_solver.py`)

```python
import numpy as np
import cvxpy as cp
import pinocchio as pin

class WholeBodyController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
    def solve(self, q, v, des_com_acc, des_foot_acc):
        # Update Kinematics
        pin.computeAllTerms(self.model, self.data, q, v)
        
        # Dimensions
        nv = self.model.nv # Num DoF (velocity)
        nu = nv - 6 # Actuated DoF
        nc = 3 # Contact dim (2D point contact: x, z, theta_y usually, we assume 3 linear)
        
        # Variables: [q_ddot, tau, f_c]
        ddq = cp.Variable(nv)
        tau = cp.Variable(nu)
        f_c = cp.Variable(nc)
        
        M = self.data.M
        h = self.data.nle
        J = pin.computeFrameJacobian(self.model, self.data, q, self.foot_id)[:nc, :] # Contact Jacobian
        S = np.hstack([np.zeros((nu, 6)), np.eye(nu)])
        
        # 1. Physics Constraints (Equations of Motion)
        # M * ddq + h = S.T * tau + J.T * f_c
        physics_constr = [M @ ddq + h == S.T @ tau + J.T @ f_c]
        
        # 2. Contact Constraints (No slip: foot accel = 0)
        # J * ddq + dJ * v = 0
        dJ = pin.getFrameJacobianTimeVariation(self.model, self.data, q, v, self.foot_id)[:nc, :]
        kin_constr = [J @ ddq + dJ @ v == des_foot_acc] # usually 0
        
        # 3. Friction Cone (Simplied 2D: |fx| <= mu * fz)
        mu = 0.5
        friction_constr = [cp.abs(f_c[0]) <= mu * f_c[2], f_c[2] >= 0]
        
        constraints = physics_constr + kin_constr + friction_constr
        
        # Cost Function
        # Track CoM Acceleration
        J_com = pin.jacobianCenterOfMass(self.model, self.data, q)
        dJ_com = pin.getCenterOfMassVelocity(self.model, self.data, q) # ... drift term
        com_acc = J_com @ ddq # + drift
        
        cost = cp.sum_squares(com_acc - des_com_acc)
        cost += 1e-3 * cp.sum_squares(tau) # Min Effort
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP)
        
        return tau.value
```

---

## 🔬 Lab Exercise: The Push Recovery

### 1. Lab Objectives
- Simulate a robot standing still.
- Apply a horizontal force (Push) for 0.2s.
- **Observe:**
    - Small Push: Robot uses ankle torque to balance (Center of Pressure shifts).
    - Large Push: Robot uses hip torque (lunges).
    - Huge Push: Friction Cone constraint binds ($f_x = \mu f_z$), robot slips/falls.

### 2. Implementation Guide
- Desired CoM Accel: $K_p (x_{ref} - x_{com}) + K_d (v_{ref} - v_{com})$.
- Feed this into the QP.

---

## 🚀 Project: "Walking Controller"

**Goal:** Combine High-Level MPC (Footstep Planner) with Low-Level WBC.
1.  **MPC (10Hz):** Plans foot locations and CoM trajectory (ZMP - Zero Moment Point) for next 2 steps.
2.  **WBC (1000Hz):** Tracks the CoM trajectory and swings the leg.
    *   **Phase 1 (Double Support):** Constraints on Left and Right foot.
    *   **Phase 2 (Single Support):** Constraint on Left only. Tracking task for Right Foot (Swing).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Singularities"
*   **Symptom:** Joint velocities explode (Infinite Jacobian Inverse).
*   **Cause:** Knee fully straight (Locking).
*   **Fix:** **Damped Least Squares** in QP weighting. Or joint limit avoidance task.

#### 2. "Floating Base Drift"
*   **Symptom:** Simulation base floats away slowly.
*   **Cause:** Numerical integration error.
*   **Fix:** Use a dedicated integrator (Pinocchio / Dart) that enforces manifold constraints, rather than Euler integration.

---

## ⚡ Optimization: Hierarchical QP (HQP)

Instead of Weighted Sum (Soft Priorities):
$$ J = w_1 J_1 + w_2 J_2 $$
Use Strict Priorities (Hard Priorities):
1.  Solve QP 1 (Balance). Get nullspace $N_1$.
2.  Solve QP 2 (Hand) projected into $N_1$.
3.  Guarantees Balance is *never* sacrificed for Hand motion.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Zero Dynamics"?
    *   **A:** The internal dynamics of the system that remain when the output is constrained to zero. In walking, if CoM is perfectly tracked, the limbs might still flail wildly if zero dynamics are unstable.
2.  **Q:** Why do we need the Friction Cone?
    *   **A:** Motors can torque hard, but the ground can't pull back. If $f_{tangential} > \mu f_{normal}$, the foot slips, and the kinematic chain breaks.
3.  **Q:** Difference between ZMP and CoM?
    *   **A:** CoM is the geometric center of mass. ZMP (Zero Moment Point) is the point on the ground where the tipping moment is zero. To not fall, ZMP must live inside the Support Polygon.

### Challenge Task
> **Task:** Jumping Controller.
> 1. Plan a trajectory with high vertical velocity.
> 2. Break contact constraint (Contact Force = 0).
> 3. Re-establish contact (Impact).

---

## 📚 Further Reading
- **WBC:** "Rigid Body Dynamics Algorithms" (Featherstone).
- **Control of Walking Robots:** Wieber (2016).
- **Pinocchio Library:** stack-of-tasks.github.io/pinocchio

---

**Day 27 Complete**
