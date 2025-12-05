# Day 28: Week 4 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 4: Advanced Control & Dynamics

---

> **📝 Content Creator Instructions:**
> We have covered the spectrum from LQR (Optimal Linear) to RL (Learned Non-linear).
> - **Goal:** Build a robust Flight Controller for a Racing Drone.
> - **Code:** Cascaded Control Loop combining MPC (Trajectory) and Geometric Tracking (Attitude).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Select** the right controller for the job (e.g., MPC for position, PID/LQR for attitude).
2.  **Architect** a Cascaded Control System (Fast Inner Loop, Slow Outer Loop).
3.  **Tune** a complex multi-variable system for high performance.
4.  **Validate** stability using Lyapunov Analysis or randomized stress testing.

---

## 📚 Week 4 Review: The Control Zoo

| Day | Controller | Best For | Pros | Cons |
|-----|------------|----------|------|------|
| **22** | **LQR** | Linear Systems | Optimally Efficient, Stable | Unconstrained, Linear only |
| **23** | **MPC** | Constraints | Respects limits, looks ahead | CPU intensive |
| **24** | **SMC** | Uncertainty | Robust to disturbances | Chattering, High Control Effort |
| **25** | **MRAC** | Adaptation | Parameter estimation | Drifting, Complex tuning |
| **26** | **RL** | Complex Tasks | Learns non-intuitive policies | Sim-to-Real gap, Training time |
| **27** | **WBC** | Humanoids | Physics-consistent contacts | Requires perfect model |

### The "Universal" Architecture
1.  **Outer Loop (10-50Hz):** MPC. Plans trajectory ($x, y, z$) avoiding obstacles. Outputs Desired Acceleration / Attitude.
2.  **Inner Loop (200-1000Hz):** LQR/SMC/PID. Tracks Desired Attitude ($\phi, \theta, \psi$). Outputs Motor Mixer commands.

---

## 🚀 Weekly Capstone: "Mach-5" Racing Drone Controller

**Scenario:** A drone must fly through a sequence of gates at 15 m/s. Winds are gusting at 5 m/s.
**Architecture:** Cascaded Control.

### 🛠️ Project Structure
```text
week4_capstone/
├── config/
│   └── drone_params.yaml
├── src/
│   ├── mpc_position_ctrl.py (Outer)
│   ├── geometric_att_ctrl.py (Inner)
│   ├── mixer.py
│   └── wind_sim.py
└── launch/
    └── high_speed_flight.launch.py
```

### 👨‍💻 Code Implementation: Geometric Attitude Controller (Inner Loop)

Conventional PID fails at 90-degree pitch (Singularity). We use **Geometric Control** on $SO(3)$.

```python
import numpy as np

class GeometricController:
    def __init__(self, kR, kW):
        self.kR = np.eye(3) * kR
        self.kW = np.eye(3) * kW
        self.J = np.diag([0.08, 0.08, 0.1]) # Inertia
        
    def compute(self, R_curr, w_curr, R_des, w_des, acc_des_scalar):
        """
        R_curr: 3x3 Rot Matrix (Body to World)
        w_curr: Angular Velocity
        acc_des_scalar: Desired Thrust Acceleration (from Position Ctrl)
        """
        # 1. Orientation Error (on SO(3) Manifold)
        # e_R = 0.5 * vee(R_des.T * R_curr - R_curr.T * R_des)
        R_err_matrix = 0.5 * (R_des.T @ R_curr - R_curr.T @ R_des)
        e_R = np.array([R_err_matrix[2,1], R_err_matrix[0,2], R_err_matrix[1,0]]) # Vee map
        
        # 2. Angular Velocity Error
        e_w = w_curr - R_curr.T @ R_des @ w_des
        
        # 3. Control Moment
        # M = -kR*eR - kW*ew + w x Jw
        M = -self.kR @ e_R - self.kW @ e_w + np.cross(w_curr, self.J @ w_curr)
        
        # 4. Total Thrust
        # T = m * acc_des
        T = acc_des_scalar # Normalized mass
        
        return T, M
```

### 👨‍💻 Code Implementation: MPC Position Controller (Outer Loop)

Uses Linear MPC (Day 23) linearized around hover, or Non-Linear MPC (Acados).
*   **Input:** Gates ($x, y, z$).
*   **Output:** Desired Acceleration vector $a_{des}$.
*   **Conversion:** $a_{des}$ is converted to $R_{des}$ (Tilt Angle) for the inner loop.
    $$ z_B = \frac{a_{des} + g}{||a_{des} + g||} $$

### 👨‍💻 Logic: Wind Rejection

To handle the 5 m/s wind:
1.  **Integral Term:** Add Integrator to Position Error.
2.  **Adaptive:** Use an Observer to estimate Wind Vector $\hat{w}$.
    $$ \dot{\hat{v}} = R a + g + \hat{w} $$
    $$ \dot{\hat{w}} = L(v_{meas} - \hat{v}) $$
    *   Feed $\hat{w}$ into MPC as a known disturbance to cancel it.

---

## 📝 Self-Assessment Quiz

1.  **Frequency:**
    *   Why must the Attitude Loop run faster than the Position Loop?
        *   **A:** Dynamics Separation. Rotational dynamics (Inertia) are much faster than Translational dynamics (Mass). The inner loop must stabilize rotation before the outer loop can effectively command a direction vector.

2.  **Singularities:**
    *   Why avoid Euler Angles (Roll/Pitch/Yaw) for a racing drone?
        *   **A:** Gimbal Lock at Pitch=90. Quaternions or Rotation Matrices ($SO(3)$) are globally unique and singularity-free.

3.  **Tuning:**
    *   If the drone wobbles quickly during hover, which gain is too high?
        *   **A:** Inner Loop D-gain (Derivative) or P-gain. It's reacting too aggressively to noise or delay.

---

## ⏭️ Look Ahead: Week 5
We have robust code. But can it run on a Raspberry Pi?
**Week 5: Edge AI & Optimization.**
*   Model Compression (Quantization/Pruning).
*   TensorRT / TFLite deployment.
*   C++ / CUDA Optimization of our Algorithms.

---

**Week 4 Complete**
