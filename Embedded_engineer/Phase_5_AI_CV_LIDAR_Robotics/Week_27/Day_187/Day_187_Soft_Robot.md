# Day 187: Soft Robot Control
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 27: Advanced Control & Dynamics

---

> **📝 Content Creator Instructions:**
> No joints, just deformation.
> - **Focus:** Finite Element Method (FEM) for control, Model Order Reduction, and controlling Pneumatic/Cable-driven Continuum Robots.
> - **Code:** `fem_beam_sim.py`. A lumped-parameter model (chain of rigid bodies with springs) approximating a soft silicone arm.
> - **Concept:** Infinite Degrees of Freedom (DoF).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Discrete (Rigid) and Continuum (Soft) mechanics.
2.  **Implement** a PCC (Piecewise Constant Curvature) kinematic model.
3.  **Simulate** a soft beam using a Spring-Damper chain.
4.  **Explain** the challenges of Hysteresis and Nonlinear Elasticity.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Hooke's Law ($F = -kx$).
- Rotation Matrices (Day 10).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Continuum Problem

Rigid robots have 6 DoF. Soft robots have $\infty$ DoF.
How do we control them?
1.  **Finite Element Method (FEM):** Split into 1000 tetrahedrons. Accurate but slow.
2.  **Piecewise Constant Curvature (PCC):** Assume the arm creates smooth circular arcs. Fast.

### 🔹 Part 2: Actuation

*   **Pneumatic Artificial Muscles (PAMs/McKibben):** Air pressure causes contraction.
*   **Tendon Driven:** Cables pull the tip.
*   **Hysteresis:** Rubber doesn't return to original shape instantly.

### 🔹 Part 3: Model Order Reduction

We can't control 1000 tetrahedrons in real time.
We project the dynamics onto a low-dimensional subspace (e.g., first 3 bending modes).

---

## 💻 Implementation: The "Lumped" Snake

We approximate a continuous silicone arm as a series of 10 rigidly linked masses connected by torsional springs.

### 🛠️ Project Structure
```text
day187_soft/
├── src/
│   ├── lumped_mass_sim.py
│   └── pcc_kinematics.py
└── output/
    └── soft_arm.png
```

### 👨‍💻 PCC Kinematics (`src/pcc_kinematics.py`)

Mapping (Cable Lengths) $\to$ (Tip Position).

```python
import numpy as np

def pcc_forward_kinematics(l1, l2, l3, diameter, length):
    """
    3-Tendon Robot Section (PCC Model).
    Calculates Curvature (kappa) and Plane (phi) from cable lengths.
    """
    # Simply: Difference in lengths causes bending.
    # l_center = (l1 + l2 + l3) / 3
    
    # Differential length
    # Simplified planar case:
    # l_diff = l_right - l_left
    
    # 3D is complex formulation (Webster & Jones, 2010).
    # Let's do 2D Planar bending for clarity.
    pass #(See Sim below)
```

### 👨‍💻 Lumped Mass Dynamic Sim (`src/lumped_mass_sim.py`)

A chain of rigid links with rotational springs.
$M \ddot{q} + C \dot{q} + K(q - q_{rest}) = \tau$.
Actuation changes $q_{rest}$ (The equilibrium shape).

```python
import numpy as np
import matplotlib.pyplot as plt

class SoftArmSim:
    def __init__(self, segments=10):
        self.N = segments
        self.L_total = 1.0
        self.L_seg = self.L_total / self.N
        
        # State: Angles [theta_0 ... theta_N-1] relative to prev link
        self.theta = np.zeros(self.N)
        self.omega = np.zeros(self.N)
        
        # Physics
        self.stiffness = 10.0 # Nm/rad
        self.damping = 0.5
        self.mass = 0.1
        
        # Actuation (Cable tension moment)
        self.u = 0.0 # Applied Torque at base/distributed
        
    def step(self, dt=0.01):
        # EOM for each joint (Simplified: Decoupled Pendulums + Coupling Spring)
        # Torque_spring_i = -k * (theta_i - theta_i_next) ? 
        # Actually discrete beam: Moment M_i = E*I * curvature
        # M_i = k * (theta_i) relative to straight.
        
        alpha = np.zeros(self.N)
        
        for i in range(self.N):
            # Spring Torque (Restoring to 0)
            tau_spring = -self.stiffness * (self.theta[i])
            
            # Damping
            tau_damp = -self.damping * self.omega[i]
            
            # External Torque (Actuation - e.g. Cable creates discrete moments)
            # Assumption: Cable tension creates uniform moment along arm
            tau_act = self.u 
            
            # Gravity (Rough approx)
            # Center of mass of this segment
            # tau_grav = ... (Skipping for simplicity of example)
            
            torque_net = tau_spring + tau_damp + tau_act
            
            # Inertia I = m*r^2
            I = (self.mass * self.L_seg**2) / 3.0
            
            alpha[i] = torque_net / I
            
        # Integration
        self.omega += alpha * dt
        self.theta += self.omega * dt
        
    def get_coords(self):
        # Forward Kinematics to get XY points
        x, y = [0], [0]
        curr_angle = np.pi/2 # Vertically up
        
        for i in range(self.N):
            curr_angle += self.theta[i] # Relative Bending
            
            nx = x[-1] + self.L_seg * np.cos(curr_angle)
            ny = y[-1] + self.L_seg * np.sin(curr_angle)
            
            x.append(nx)
            y.append(ny)
            
        return x, y

def main():
    sim = SoftArmSim(segments=10)
    
    plt.ion()
    fig, ax = plt.subplots()
    
    # Simulation: Bend the arm
    for t in range(200):
        if t < 100:
            sim.u = 2.0 # Apply Bending Torque
        else:
            sim.u = 0.0 # Release (Elastic Rebound)
            
        sim.step()
        
        x, y = sim.get_coords()
        
        ax.clear()
        ax.plot(x, y, 'o-', linewidth=3, markersize=5)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, 1.2)
        ax.set_aspect('equal')
        plt.title(f"Soft Arm Sim t={t}")
        plt.pause(0.01)
        
    plt.ioff()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Tentacle"

### 1. Lab Objectives
- **Run:** `lumped_mass_sim.py`.
- **Modify:** Make the `stiffness` variable along the length (Stiff base, Soft tip).
- **Observe:** The "Whiplash" effect when torque is released.
- **Control:** Implement a PID controller on `sim.u` to maintain tip position at $(x, y) = (0.3, 0.8)$.
    *   Requires simple Jacobian (Inverse Kinematics).

---

## 🚀 Project: "Soft Gripper"

**Goal:** Grasping irregular objects.
1.  **Sim:** 2 Soft Fingers (PneuNets).
2.  **Contact:** When finger hits object, it deforms *around* it (Form Closure).
3.  **Advantage:** No precise planning needed. Just "close".
4.  **Visualize:** FEM-like deformation in 2D.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Numerical Explosion"
*   **Cause:** Stiff springs + large $dt$.
*   **Fix:** Decrease $dt$. Or use Implicit Euler integration.

#### 2. "Gravity Sag"
*   **Cause:** Soft robots are heavy. They droop.
*   **Fix:** Dynamics model must include gravity vector computation for every segment.

---

## ⚡ Optimization: SOFA Framework

Writing FEM from scratch is hard.
*   **SOFA (Simulation Open Framework Architecture):** Standard for medical/soft robotics.
*   **Integration:** ROS 2 + SOFA junction exists.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Reduced Order Model vs Full FEM?
    *   **A:** FEM has 10,000 states. ROM has 10 states (generalized coordinates). ROM is fast enough for control, FEM is for ground truth validation.
2.  **Q:** What is "PCC"?
    *   **A:** Piecewise Constant Curvature. An assumption that each section of the arm bends into a perfect circle arc.

### Challenge Task
> **Task:** "Obstacle Wrap".
> 1. Soft arm hits a cylinder obstacle.
> 2. Continue actuating.
> 3. Arm should wrap around the obstacle naturally (Passive Compliance).

---

## 📚 Further Reading
- **Webster & Jones:** "Design and Kinematic Modeling of Constant Curvature Continuum Robots".
- **Soft Robotics Toolkit:** Open source hardware designs.

---

## 🔗 External Resources
### 📜 Open Source Libraries
- [jgilin/soft-robot-simulator](https://github.com/jgilin/soft-robot-simulator) - PyBullet based simulation for cable-driven robots.
- [skriegman/evosoro](https://github.com/skriegman/evosoro) - Evolutionary Soft Robotics Simulator.

### 📺 Video Tutorials
- [Soft Robotics Control (IEEE)](https://www.youtube.com/results?search_query=Soft+Robotics+Control+IEEE) - Design and Applications overview.
- [Controlling Soft Robots in Task Space](https://www.youtube.com/results?search_query=Controlling+Soft+Robots+in+Task+Space) - Model-based approaches.

---

**Day 187 Complete**
