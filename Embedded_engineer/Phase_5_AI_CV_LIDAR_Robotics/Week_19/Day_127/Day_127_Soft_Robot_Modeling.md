# Day 127: Soft Robot Modeling (Continuum Manipulators)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 19: Soft Robotics & Bio-Inspired Control

---

> **📝 Content Creator Instructions:**
> Robots don't have to be rigid.
> - **Focus:** Constant Curvature Assumption (CCA), Modeling Continuum arms (Elephant Trunks), and Mapping Actuator Space (Pressure) to Config Space (Arc, Curvature, Phi) to Task Space (XYZ).
> - **Code:** A Python Kinematics Solver `continuum_kinematics.py` that visualizes a 3-segment soft arm using Matplotlib.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Contrast** Discrete Joint Robots (Rigid Links) vs Continuum Robots (Infinite DOF).
2.  **Apply** the Constant Curvature Assumption (CCA) to simplify the kinematics.
3.  **Map** Tendon Lengths/Pressure $\to$ Radius of Curvature ($R$) and Angle ($\phi$).
4.  **Visualize** the workspace of a soft manipulator.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Simulation).

### Software Environment
```bash
pip install numpy matplotlib scipy
```

### Prior Knowledge
- FK/IK (Week 3).
- Arc Geometry ($s = R \theta$).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Soft?

Rigid robots are dangerous and heavy. Soft robots (Silicone, Fabric) are safe and adaptive.
*   **Degrees of Freedom:** Theoretically Infinite.
*   **Modeling:** Hard. We use approximations.

### 🔹 Part 2: Constant Curvature Assumption (CCA)

We assume each "Segment" of the robot bends into a perfect circular arc.
State variables for one segment:
1.  **$\kappa$ (Kappa):** Curvature ($1/Radius$).
2.  **$\phi$ (Phi):** Plane of bending (Rotation around Z).
3.  **$s$ (Arc Length):** Length of the neutral axis.

### 🔹 Part 3: Forward Kinematics

Transformation from Base to Tip (Denavit-Hartenberg for Soft Robots):
$$
T(s) = \begin{bmatrix}
\cos\phi \cos(\kappa s) & -\sin\phi & \cos\phi \sin(\kappa s) & \frac{1}{\kappa}\cos\phi(1-\cos(\kappa s)) \\
\sin\phi \cos(\kappa s) & \cos\phi & \sin\phi \sin(\kappa s) & \frac{1}{\kappa}\sin\phi(1-\cos(\kappa s)) \\
-\sin(\kappa s) & 0 & \cos(\kappa s) & \frac{1}{\kappa}\sin(\kappa s) \\
0 & 0 & 0 & 1
\end{bmatrix}
$$
(Simplified version. Often broken into Rotation(Phi) * In-Plane-Bend(Kappa) * Rotation(-Phi)).

---

## 💻 Implementation: Soft Arm Simulator

We simulate a 2-segment tendon-driven arm.

### 🛠️ Project Structure
```text
day127_soft/
├── src/
│   ├── continuum_kinematics.py
└── output/
    ├── workspace.png
```

### 👨‍💻 Kinematics Solver (`src/continuum_kinematics.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ContinuumSegment:
    def __init__(self, length):
        self.l0 = length

    def get_transform(self, kappa, phi, s=None):
        if s is None: s = self.l0
        
        # Avoid singularity at kappa=0 (Straight line)
        if abs(kappa) < 1e-6:
            # Limit as kappa -> 0 is a translation along Z
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, s],
                [0, 0, 0, 1]
            ])

        rad = 1.0 / kappa
        
        # Position
        x = rad * np.cos(phi) * (1 - np.cos(kappa * s))
        y = rad * np.sin(phi) * (1 - np.cos(kappa * s))
        z = rad * np.sin(kappa * s)
        
        # Rotation (Tangental frame)
        # Simplified: Just rotate the frame by angle theta = kappa * s around an axis perp to phi plane
        
        # Full Homogeneous Transform
        # Method: Rot(z, phi) * Trans(x_in_plane, 0, z_in_plane) * Rot(y, theta) * Rot(z, -phi) 
        # But let's use the explicit calculated matrix for efficiency
        
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_ks = np.cos(kappa * s)
        s_ks = np.sin(kappa * s)
        
        T = np.array([
            [c_phi*c_phi*(c_ks-1) + 1, c_phi*s_phi*(c_ks-1), c_phi*s_ks, x],
            [s_phi*c_phi*(c_ks-1), s_phi*s_phi*(c_ks-1) + 1, s_phi*s_ks, y],
            [-c_phi*s_ks, -s_phi*s_ks, c_ks, z],
            [0, 0, 0, 1]
        ])
        
        return T

class SoftRobot:
    def __init__(self):
        self.seg1 = ContinuumSegment(0.2) # 20cm
        self.seg2 = ContinuumSegment(0.2)
        
    def forward_kinematics(self, q):
        # q = [kappa1, phi1, kappa2, phi2]
        k1, p1, k2, p2 = q
        
        points = []
        points.append([0,0,0])
        
        # Trace Segment 1
        num_pts = 10
        for i in range(1, num_pts+1):
            s = self.seg1.l0 * (i/num_pts)
            T = self.seg1.get_transform(k1, p1, s)
            points.append(T[:3, 3])
            
        T1_end = self.seg1.get_transform(k1, p1)
        
        # Trace Segment 2 (Relative to T1_end)
        for i in range(1, num_pts+1):
            s = self.seg2.l0 * (i/num_pts)
            T_rel = self.seg2.get_transform(k2, p2, s)
            T_global = T1_end @ T_rel
            points.append(T_global[:3, 3])
            
        return np.array(points)

def main():
    robot = SoftRobot()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-0.3, 0.3])
    ax.set_ylim([-0.3, 0.3])
    ax.set_zlim([0, 0.5])
    
    # 1. Straight Up
    pts = robot.forward_kinematics([0.0001, 0, 0.0001, 0])
    ax.plot(pts[:,0], pts[:,1], pts[:,2], 'k--', label='Straight')
    
    # 2. Curl
    # k = 5.0 (R=0.2m), phi=0
    pts = robot.forward_kinematics([5.0, 0, 5.0, 0])
    ax.plot(pts[:,0], pts[:,1], pts[:,2], 'r-', label='Curl X')
    
    # 3. S-Shape
    # k1=5, k2=-5 (phi2 = pi)
    pts = robot.forward_kinematics([5.0, 0, 5.0, np.pi])
    ax.plot(pts[:,0], pts[:,1], pts[:,2], 'b-', label='S-Shape')
    
    ax.legend()
    plt.savefig("output/workspace.png")
    print("Soft Robot Viz Saved.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Tentacle"

### 1. Lab Objectives
- **Run:** The simulator.
- **Modify:** `q` to `[10.0, 0, 10.0, np.pi/2]`.
- **Result:** A 3D corkscrew shape.
- **Actuation Mapping:**
    *   Real robots use Tendons ($l_1, l_2, l_3$) or Pneumatics ($P_1, P_2, P_3$).
    *   $\kappa \propto \frac{l_1 - l_2}{diameter}$.
    *   $\phi \propto \arctan(...)$.

---

## 🚀 Project: "Inverse Kinematics (Jacobian)"

**Goal:** Reach a point $(x,y,z)$.
1.  **State:** $q = [\kappa_1, \phi_1, \kappa_2, \phi_2]$.
2.  **Jacobian:** $J = \frac{\partial X}{\partial q}$. (Calculate numerically by perturbing $q$).
3.  **Update:** $q_{new} = q_{old} + J^{\dagger} (X_{target} - X_{current})$.
4.  **Result:** The soft arm slowly bends to touch the target.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Singularity at Kappa=0"
*   **Math:** The formulas involve $1/\kappa$. Use `if abs(k)<eps` check.
*   **Physics:** Straight line is structurally stiff (buckling). Small perturbations cause snap-through.

#### 2. "Model Mismatch"
*   **Reality:** Gravity causes the soft arm to droop. CCA assumes zero mass or internal stiffness domination.
*   **Fix:** Use Cosserat Rod Theory (Finite Element) for heavy arms (simulated in PyBullet or SoRoSim).

---

## ⚡ Optimization: Piecewise Constant Curvature (PCC)

Approximating a continuous trunk as $N$ discrete circular arcs.
*   $N=1$: Bad approximation.
*   $N=10$: Good approximation.
*   Computation scales linearly.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What determines the stiffness of a pneumatic robot?
    *   **A:** The Pressure. Higher P = Stiffer robot.
2.  **Q:** Why not use standard DH parameters?
    *   **A:** Soft robots don't have discrete axes of rotation. CCA is the equivalent "Standard" for continuum.
3.  **Q:** Can soft robots handle high payloads?
    *   **A:** Generally no. They deform under load. They are better for handling delicate objects (fruit, tissues).

### Challenge Task
> **Task:** Workspace Cloud.
> 1. Monte Carlo: Sample 1000 random $q$ vectors.
> 2. Plot the Tip Position for each.
> 3. Visualize the "Reachable Workspace" (It looks like a donut/mushroom).

---

## 📚 Further Reading
- **Webster & Jones:** "Design and Kinematic Modeling of Constant Curvature Continuum Robots: A Review".
- **SoRoSim:** Soft Robotics Simulation Toolbox (Matlab).

---

**Day 127 Complete**
