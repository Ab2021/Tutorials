# Day 51: Inverse Kinematics (Numerical Solvers)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 8: Advanced Manipulation

---

> **📝 Content Creator Instructions:**
> FK is easy (Geometry). IK is hard (Optimization).
> - **Focus:** Analytical (Closed-form) vs Numerical (Iterative). Jacobian Inverse, Pseudo-Inverse, Damped Least Squares.
> - **Code:** A Numerical IK solver using Newton-Raphson for 6-DOF.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Understand** the IK problem: Find $\theta$ such that $FK(\theta) = T_{target}$.
2.  **Calculate** the Jacobian Matrix $J(\theta)$ mapping joint velocities $\dot{\theta}$ to EE velocities $\dot{x}$.
3.  **Implement** the Newton-Raphson method for IK: $\theta_{new} = \theta_{old} + J^{-1} \cdot \text{error}$.
4.  **Handle** Singularities (Gimbal Lock) using Damped Least Squares (Levenberg-Marquardt).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy robotics-toolbox-python
```

### Prior Knowledge
- Derivatives / Gradient Descent.
- SVD (Singular Value Decomposition).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Analytical vs Numerical

1.  **Analytical:**
    *   Solve lots of trigonometry triangles.
    *   *Pros:* Exact, Instant (nanoseconds), Finds all solutions (Elbow Up/Down).
    *   *Cons:* Deriving it for a 6-DOF is a nightmare ($T_{01}T_{12}... = T_{goal}$). Only works for specific geometries (Pieper's criteria: 3 wrist axes intersect).
2.  **Numerical:**
    *   Treat it as a minimization problem: $\min || FK(\theta) - T_{target} ||^2$.
    *   *Pros:* Works for ANY robot (7-DOF, 20-DOF snake).
    *   *Cons:* Iterative (slower), Can get stuck in local minima.

### 🔹 Part 2: The Jacobian Matrix $J$

The Jacobian relates differential changes:
$$ \dot{x} = J(\theta) \cdot \dot{\theta} $$
*   $\dot{x}$: EE velocity ($v_x, v_y, v_z, \omega_x, \omega_y, \omega_z$) [6x1].
*   $\dot{\theta}$: Joint velocities [Nx1].
*   $J$: [6xN] Matrix.

### 🔹 Part 3: Inverse Kinematics via Jacobian

We want to move EE by $\Delta x$ (Limit of error).
$$ \Delta x \approx J \cdot \Delta \theta $$
$$ \Delta \theta \approx J^{-1} \cdot \Delta x $$
**Update Rule:** $\theta_{k+1} = \theta_k + \alpha J^{\dagger} (\text{Target} - \text{Current})$.
*   $J^{\dagger}$: Moore-Penrose Pseudo-Inverse (Use if $N \ne 6$ or Singularity).

### 🔹 Part 4: Singularities

When robot arm is fully extended, it loses a degree of freedom (Cannot move further out).
*   Determinant of $J \to 0$.
*   $J^{-1} \to \infty$.
*   Ideally, robot should stop. Numerically, it explodes (Joints spin at infinite speed).
*   **Fix:** **Damped Least Squares (DLS)**.
    $J^* = J^T (JJ^T + \lambda^2 I)^{-1}$
    Adds "damping" ($\lambda$) to keep velocities sane near singularity.

---

## 💻 Implementation: Newton-Raphson IK

We will solve IK for the UR5 to reach a target $(x,y,z)$.

### 🛠️ Project Structure
```text
day51_ik/
├── src/
│   ├── numerical_ik.py
│   └── jacobian.py
└── run_ik_trace.py
```

### 👨‍💻 Jacobian Calculation (`src/jacobian.py`)

Using Finite Difference (Easiest to implement without heavy math derivation).

```python
import numpy as np
from src.dh_solver import UR5Solver # From Day 50

class JacobianSolver:
    def __init__(self):
        self.fk = UR5Solver()
        self.dt = 1e-6 # Perturbation size
        
    def get_jacobian(self, theta):
        J = np.zeros((6, 6))
        
        # Get current pose (Base state)
        T_curr, _ = self.fk.forward_kinematics(theta)
        pos_curr = T_curr[:3, 3]
        
        # For orientation, simplified: specific Euler angles
        # Full Jacobian usually requires quaternions or rotation vectors deviation
        # Here we do Partial Jacobian (Position Only) 3x6 for simplicity
        
        for i in range(6):
            # Perturb joint i
            theta_p = theta.copy()
            theta_p[i] += self.dt
            
            T_new, _ = self.fk.forward_kinematics(theta_p)
            pos_new = T_new[:3, 3]
            
            # Derivative = (f(x+h) - f(x)) / h
            diff = (pos_new - pos_curr) / self.dt
            
            J[:3, i] = diff
            
        return J[:3, :] # Return 3x6 (Position Only)
```

### 👨‍💻 Numerical IK (`src/numerical_ik.py`)

```python
from src.jacobian import JacobianSolver
import numpy as np

class IKSolver:
    def __init__(self):
        self.jac = JacobianSolver()
        self.max_iter = 100
        self.tolerance = 1e-3
        
    def solve(self, target_pos, seed_guess=None):
        if seed_guess is None:
            seed_guess = np.zeros(6)
            
        theta = seed_guess.copy()
        
        for i in range(self.max_iter):
            # 1. Forward Kinematics
            T, _ = self.jac.fk.forward_kinematics(theta)
            curr_pos = T[:3, 3]
            
            # 2. Error
            error = target_pos - curr_pos
            if np.linalg.norm(error) < self.tolerance:
                print(f"Converged in {i} iters.")
                return theta, True
                
            # 3. Jacobian
            J = self.jac.get_jacobian(theta) # 3x6
            
            # 4. Inverse Jacobian (Pseudo-Inverse for Non-Square)
            # Use Damped Least Squares: J_pinv = J.T @ inv(J @ J.T + lambda*I)
            lam = 0.01
            A = J @ J.T + lam * np.eye(3)
            J_dls = J.T @ np.linalg.inv(A)
            
            # 5. Update
            d_theta = J_dls @ error
            theta += d_theta * 0.5 # Learning rate
            
        print("Failed to converge.")
        return theta, False
```

### 👨‍💻 Drawing a Circle (`run_ik_trace.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
from src.numerical_ik import IKSolver

ik = IKSolver()

# Generate Circle Trajectory
angles = np.linspace(0, 2*np.pi, 50)
radius = 0.2
center = [0.4, 0.0, 0.4] # X, Y, Z

path = []
for a in angles:
    path.append([
        center[0], 
        center[1] + radius * np.cos(a),
        center[2] + radius * np.sin(a)
    ])

# Trace
joints = np.zeros(6) # Start guess
final_joints = []
actual_path = []

for pt in path:
    sol, success = ik.solve(pt, seed_guess=joints)
    if success:
        joints = sol # Warm start next iter
        final_joints.append(sol)
        
        # Verify FK
        T, _ = ik.jac.fk.forward_kinematics(sol)
        actual_path.append(T[:3, 3])

# Plot comparison
actual_path = np.array(actual_path)
path = np.array(path)

plt.plot(path[:,1], path[:,2], 'b--', label='Target')
plt.plot(actual_path[:,1], actual_path[:,2], 'r-', label='IK Result')
plt.legend()
plt.title("YZ Plane Circle Tracing")
plt.show()
```

---

## 🔬 Lab Exercise: The "Elbow" Flip

### 1. Lab Objectives
- Numerical IK leads to *one* solution near the seed guess.
- **Task:** Solve for a point reachable by both "Elbow Up" and "Elbow Down" configs.
- **Experiment:** Start with seed `[0,0,0,0,0,0]` $\to$ result A. Start with seed `[0, -1.5, 1.5, ...]` $\to$ result B.
- **Goal:** Understand Multimodality. Analytical IK gives all 8 solutions. Numerical gives 1.

---

## 🚀 Project: "Writing Your Name"

**Goal:** Create a list of XY points forming letters (e.g., "HELLO").
1.  Map XY to robot workspace (e.g., Z constant = table height).
2.  Iterate IK solver over points.
3.  Visualize the End Effector "Pen" drawing the path.
4.  **Constraint:** Keep the pen vertical ($R_{EE}$ fixed pointing down). Requires Full 6x6 Jacobian.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Local Minima"
*   **Symptom:** Error > Tolerance but Updates $\to 0$. Robot stuck in weird pose not reaching goal.
*   **Cause:** Iterative method stuck in a valley.
*   **Fix:** Random Restarts. If stuck, randomize seed and try again.

#### 2. "Oscillation"
*   **Symptom:** Robot vibrates around target.
*   **Fix:** Reduce "Learning Rate" ($\alpha$). Increase Damping ($\lambda$).

---

## ⚡ Optimization: TRAC-IK

Standard ROS package `trac_ik`.
*   Competition-winning solver.
*   Runs **Parallel Threads**:
    1.  KDL (Newton-Raphson).
    2.  SQP (Sequential Quadratic Programming).
*   Returns whichever finishes first. Robust and Fast.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why Pseudo-Inverse?
    *   **A:** Standard Inverse only exists for Square, Full-Rank matrices. Jacobian can be 3x6 or Singular (Rank Deficient).
2.  **Q:** Jacobian Transpose Method?
    *   **A:** Approximation: $\Delta \theta \approx J^T \cdot \text{error}$. Simpler math (no inversion), but slower convergence. Used in Physics Engines.
3.  **Q:** Joint Limits?
    *   **A:** Numerical IK ignores limits (might output $\theta = 400^\circ$). Must clamp inside the loop or add constraint terms to cost function.

### Challenge Task
> **Task:** Null Space Control.
> 1. For a 7-DOF robot, $J$ is 6x7.
> 2. We can move joints without moving EE (Self-motion).
> 3. Use Null Space projection to minimize "Distance from comfortable pose" while keeping EE fixed.

---

## 📚 Further Reading
- **Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods:** Samuel Buss.
- **TRAC-IK:** Patrick Beeson.

---

**Day 51 Complete**
