# Day 50: 6-DOF Kinematics (DH Parameters)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 8: Advanced Manipulation

---

> **📝 Content Creator Instructions:**
> We leave the ground. We enter the 3D workspace.
> - **Focus:** Forward Kinematics (FK), Denavit-Hartenberg (DH) Parameters, and Transformation Matrices.
> - **Code:** Implementing a FK solver for a UR5 robot arm from scratch (No ROS libraries yet).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Construct** the DH Table for a 6-DOF Serial Manipulator (e.g., UR5, Panda).
2.  **Compute** the Homogeneous Transformation Matrix $T_{06}$ given joint angles $\theta_1...\theta_6$.
3.  **Visualize** the robot frame-by-frame using Matplotlib 3D or OpenGL.
4.  **Differentiate** between Standard DH and Modified DH conventions.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib scipy robotics-toolbox-python
```

### Prior Knowledge
- Linear Algebra (Matrix Multiplication).
- SE(3) Rigid Body Transforms.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Serial Chain

A robot arm is a chain of rigid bodies (Links) connected by joints (Revolute/Prismatic).
*   **Base Frame (0):** Fixed to ground.
*   **End-Effector (EE):** The gripper.
*   **Goal:** Find Position $(x,y,z)$ and Orientation $(R_{3x3})$ of EE given Joint Angles $(\theta)$.

### 🔹 Part 2: Denavit-Hartenberg (DH) Convention

A systematic way to attach coordinate frames to links.
4 Parameters per link:
1.  **$\theta_i$ (Joint Angle):** Rotation around $z_{i-1}$.
2.  **$d_i$ (Link Offset):** Translation along $z_{i-1}$.
3.  **$a_i$ (Link Length):** Translation along $x_i$ (Common Normal).
4.  **$\alpha_i$ (Twist):** Rotation around $x_i$.
*   **Transform:** $A_i = Rot_z(\theta) \cdot Trans_z(d) \cdot Trans_x(a) \cdot Rot_x(\alpha)$.

### 🔹 Part 3: Forward Kinematics (FK)

The global position is the product of local transforms:
$$ T_{0n} = A_1 \cdot A_2 \cdot ... \cdot A_n $$
For a 6-DOF arm: $T_{06} = A_1 A_2 A_3 A_4 A_5 A_6$.
The result is a $4 \times 4$ matrix:
$$
T_{06} = \begin{bmatrix}
R_{3 \times 3} & P_{3 \times 1} \\
0 & 1
\end{bmatrix}
$$
*   $P$: Position of gripper.
*   $R$: Orientation of gripper.

---

## 💻 Implementation: UR5 FK Solver

We will implement the math for the Universal Robots UR5 (Standard industrial arm).

### 🛠️ Project Structure
```text
day50_kinematics/
├── src/
│   ├── dh_solver.py
│   └── visualize_arm.py
└── run_fk.py
```

### 👨‍💻 DH Solver (`src/dh_solver.py`)

```python
import numpy as np

class UR5Solver:
    def __init__(self):
        # UR5 DH Parameters (Standard)
        # [theta, d, a, alpha]
        # Theta is variable (input), others are fixed specs
        self.d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
        self.a = [0, -0.425, -0.39225, 0, 0, 0]
        self.alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]
    
    def get_transform(self, theta, d, a, alpha):
        c = np.cos(theta)
        s = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        # Standard DH Matrix
        return np.array([
            [c, -s*ca, s*sa, a*c],
            [s, c*ca, -c*sa, a*s],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
        
    def forward_kinematics(self, joints):
        # joints: [th1, th2, th3, th4, th5, th6]
        T = np.eye(4)
        
        # Store individual frame positions for plotting
        frames = [T[:3, 3]] 
        
        for i in range(6):
            A = self.get_transform(joints[i], self.d[i], self.a[i], self.alpha[i])
            T = np.dot(T, A)
            frames.append(T[:3, 3])
            
        return T, np.array(frames)
```

### 👨‍💻 Visualization (`run_fk.py`)

Using Matplotlib to draw the stick figure.

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.dh_solver import UR5Solver
import numpy as np

solver = UR5Solver()

# Test Configuration (Home position: all zeros? UR5 home is usually L-shape or Upright)
# Let's try "Zero" config (Arm stretched out horizontally usually)
joints = [0, -np.pi/2, 0, -np.pi/2, 0, 0] # Upright L shape

T_ee, frames = solver.forward_kinematics(joints)

print("End Effector Pose:")
print(T_ee)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# X, Y, Z coords of joints
xs = frames[:, 0]
ys = frames[:, 1]
zs = frames[:, 2]

ax.plot(xs, ys, zs, '-o', linewidth=2, markersize=8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.8, 0.8)
ax.set_zlim(0, 1.0)
ax.set_title(f"UR5 Arm Config: {joints}")

# Draw Base
ax.text(xs[0], ys[0], zs[0], "Base")
ax.text(xs[-1], ys[-1], zs[-1], "EE")

plt.show()
```

### 3. Expected Output
*   A 3D plot showing the robot links connected.
*   If we change `joints[0]` (Base pan), the whole arm should rotate around Z-axis.

---

## 🔬 Lab Exercise: The "Reach" Test

### 1. Lab Objectives
- Calculate FK for random valid joint angles.
- **Check 1:** Is length of Link 2 constant? ($||P_2 - P_1|| \approx 0.425m$).
- **Check 2:** Move only Joint 6 (Wrist Roll). EE position should stay same, only Orientation changes.
- **Check 3:** Singularity check. Fully extend arm -> Determinant of Jacobian goes to 0 (Next Lesson), but FK still works.

---

## 🚀 Project: "Robot Digital Twin"

**Goal:** Connect to a Simulator (PyBullet/Gazebo).
1.  **Read:** Joint States `/joint_states`.
2.  **Compute:** Your custom FK.
3.  **Compare:** ROS TF `/tf` topic for `end_effector_link`.
4.  **Result:** Error should be $< 10^{-5}$ meters. If error is large, your DH parameters are wrong.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Arm explodes"
*   **Symptom:** Links are 100 meters long in plot.
*   **Cause:** Units! DH table usually in Meters. If you put 425 (mm) instead of 0.425 (m), the math breaks.

#### 2. "Wrong Axis of Rotation"
*   **Symptom:** Joint 2 rotates around X instead of Y.
*   **Cause:** Standard DH requires specific frame alignment (Z-axis is joint axis). Checking `alpha` twist is tricky.
*   **Fix:** Use **URDF** (Unified Robot Description Format) parsing instead of manual DH if available. (Wait for Day 52).

---

## ⚡ Optimization: Symbolic Math

Calculating 6 matrix multiplications of sines/cosines is slow (0.1ms).
*   **SymPy:** Generate the analytical equation for $x, y, z$ as a huge string of trig functions.
*   **Lambdify:** Compile that string to C code.
*   **Result:** FK evaluation in 1 microsecond.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Difference between Revolute and Prismatic?
    *   **A:** Revolute rotates ($\theta$ variable). Prismatic slides ($d$ variable).
2.  **Q:** What is a "redundant" robot?
    *   **A:** A robot with > 6 DOF (e.g., 7 DOF Panda). Infinite solutions for the same EE pose.
3.  **Q:** Why is $T_{06}$ a $4 \times 4$ matrix?
    *   **A:** To assume translation and rotation in one linear operation (Homogeneous coordinates).

### Challenge Task
> **Task:** 2-Link Planar Arm.
> 1. Derive FK equations manually on paper:
>    $x = l_1 \cos(\theta_1) + l_2 \cos(\theta_1 + \theta_2)$
>    $y = l_1 \sin(\theta_1) + l_2 \sin(\theta_1 + \theta_2)$
> 2. Verify your code matches these equations.

---

## 📚 Further Reading
- **Introduction to Robotics:** Craig (Standard Textbook).
- **Modern Robotics:** Lynch & Park (Twist/Screw Theory approach - Alternative to DH).

---

**Day 50 Complete**
