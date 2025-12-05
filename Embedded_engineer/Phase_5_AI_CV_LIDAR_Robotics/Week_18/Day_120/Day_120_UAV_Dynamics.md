# Day 120: UAV Dynamics & Control
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 18: Aerial & Underwater Robotics

---

> **📝 Content Creator Instructions:**
> Gravity is strictly enforced.
> - **Focus:** Quadrotor Physics (Thrust/Torque), The Cascaded PID Control Loop, and the concept of "Underactuated Systems" (4 rotors, 6 DOF).
> - **Code:** A Python script simulating the physics of a quadrotor (Euler Integration) and implementing a Position-Attitude-Rate controller to hold a hover.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Derive** the basic Equations of Motion for a Quadrotor ($\dot{v} = g z_w - \frac{T}{m} R z_b$).
2.  **Explain** the mixing matrix (How 4 motors generate Roll, Pitch, Yaw, Thrust).
3.  **Implement** a Nested PID structure (Outer Loop: Maximize Position $\to$ Inner Loop: Maximize Rate).
4.  **Visualize** the drone stability in `matplotlib` or Rviz.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Simulation). Use standard Laptop.

### Software Environment
```bash
pip install numpy matplotlib scipy
```

### Prior Knowledge
- Rigid Body Dynamics.
- PID Control (Day 86).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Physics

A quadrotor has 4 props.
*   **Thrust ($T$):** Sum of all 4 motor forces. Lifts the drone ($Z$ axis).
*   **Torque ($\tau$):** Difference in motor forces creates Roll/Pitch. Difference in Drag Torque (CW vs CCW props) creates Yaw.
*   **Underactuated:** You cannot move sideways (Y) without tilting (Roll).
    *   To move Right: Roll Right $\to$ Thrust Vector points Right $\to$ Accelerate Right.

### 🔹 Part 2: The Mixer

How do we convert $(T, Roll, Pitch, Yaw_{torque})$ to Motor Speeds $(\omega_1, \omega_2, \omega_3, \omega_4)$?
$$
\begin{bmatrix} \omega_1^2 \\ \omega_2^2 \\ \omega_3^2 \\ \omega_4^2 \end{bmatrix} = 
\begin{bmatrix} 
1 & 1 & -1 & -1 \\
1 & -1 & -1 & 1 \\
1 & -1 & 1 & -1 \\
1 & 1 & 1 & 1 
\end{bmatrix}^{-1} 
\begin{bmatrix} T \\ \tau_\phi \\ \tau_\theta \\ \tau_\psi \end{bmatrix}
$$
(Simplification for X configuration).

### 🔹 Part 3: Cascaded Control

You cannot control Position directly. You control Fast things to influence Slow things.
1.  **Position Controller (Slow):** Error in X $\to$ Desired Pitch Angle.
2.  **Attitude Controller (Fast):** Error in Pitch $\to$ Desired Pitch Rate.
3.  **Rate Controller (Very Fast):** Error in Pitch Rate $\to$ Motor Torque.

---

## 💻 Implementation: Quadrotor Simulator

We build a physics engine from scratch.

### 🛠️ Project Structure
```text
day120_uav/
├── src/
│   ├── quad_sim.py
└── output/
    └── trajectory.png
```

### 👨‍💻 Physics & Control (`src/quad_sim.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class Quadrotor:
    def __init__(self):
        # Constants
        self.m = 1.0 # kg
        self.g = 9.81
        self.dt = 0.01
        
        # State [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        self.state = np.zeros(12)
        
        # Controller State
        self.integral_error_pos = np.zeros(3)
        self.last_error_pos = np.zeros(3)

    def dynamics(self, u):
        # u = [Thrust, TorqueX, TorqueY, TorqueZ] (Simplified inputs)
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = self.state
        
        # Rotation Matrix (Body to Earth) - Small angle approx for simplicity or full R
        cph, sph = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cps, sps = np.cos(psi), np.sin(psi)
        
        # R_z * R_y * R_x ... standard ZYX
        # Let's use simplified equations for hovering
        
        # Linear Acceleration
        # Thrust acts in Body Z direction.
        # F_earth = R * [0, 0, T] - [0, 0, mg]
        
        # Z-axis (Up is positive in Earth Frame for this code)
        az = (np.cos(phi)*np.cos(theta) * u[0] / self.m) - self.g
        ax = (np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)) * u[0] / self.m
        ay = (np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)) * u[0] / self.m
        
        # Angular Acceleration (Inertia I=diag(1,1,1) for simplicity)
        dp = u[1] # / Ixx
        dq = u[2] # / Iyy
        dr = u[3] # / Izz
        
        return np.array([vx, vy, vz, ax, ay, az, p, q, r, dp, dq, dr])

    def step(self, u):
        # RK1 (Euler)
        ds = self.dynamics(u)
        self.state += ds * self.dt
        return self.state

    def controller(self, target_pos):
        # Cascaded PID
        
        # 1. Position Loop
        curr_pos = self.state[0:3]
        curr_vel = self.state[3:6]
        
        pos_err = target_pos - curr_pos
        vel_err = np.array([0,0,0]) - curr_vel
        
        # Desired Accelerations (P-D controller on Pos)
        kp_pos = 2.0
        kd_pos = 2.0
        des_acc = kp_pos * pos_err + kd_pos * vel_err
        
        # Feedforward Gravity
        thrust_des = (des_acc[2] + self.g) * self.m
        # Clamp Thrust
        thrust_des = np.clip(thrust_des, 0, 20)
        
        # Roll/Pitch from X/Y acceleration 
        # Small angle: ax ~ g * theta, ay ~ -g * phi
        des_theta = des_acc[0] / self.g
        des_phi = -des_acc[1] / self.g
        
        des_theta = np.clip(des_theta, -0.5, 0.5) # Limit tilt 30 deg
        des_phi = np.clip(des_phi, -0.5, 0.5)
        
        # 2. Attitude Loop
        curr_att = self.state[6:9] # phi, theta, psi
        att_err = np.array([des_phi, des_theta, 0.0]) - curr_att
        
        kp_att = 10.0
        des_rates = kp_att * att_err
        
        # 3. Rate Loop
        curr_rates = self.state[9:12]
        rate_err = des_rates - curr_rates
        
        kp_rate = 5.0
        torques = kp_rate * rate_err
        
        return np.array([thrust_des, torques[0], torques[1], torques[2]])

def main():
    quad = Quadrotor()
    history = []
    times = []
    
    target = np.array([2.0, 2.0, 5.0]) # Hover at 5m height
    
    for t in np.arange(0, 10.0, quad.dt):
        u = quad.controller(target)
        s = quad.step(u)
        history.append(s[:3].copy())
        times.append(t)
        
    history = np.array(history)
    
    # Plot
    plt.figure()
    plt.plot(times, history[:, 0], label='X')
    plt.plot(times, history[:, 1], label='Y')
    plt.plot(times, history[:, 2], label='Z')
    plt.axhline(target[2], color='k', linestyle='--', label='Target Z')
    plt.legend()
    plt.title("Quadrotor Step Response")
    plt.savefig("output/trajectory.png")
    print("Simulation Complete. Saved plot.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Crash and Burn"

### 1. Lab Objectives
- **Run:** The simulator. Observe stable rise to 5m.
- **Modify:** `kp_att` (Attitude P gain) from 10.0 to 1.0.
- **Run:** The drone oscillates wildly or flips.
- **Modify:** `kd_pos` (Position D gain) to 0.0.
- **Run:** Drone overshoots target (goes to 7m, falls to 3m, bounces).
- **Lesson:** Drones are unstable systems. Tuning is critical.

---

## 🚀 Project: "Trajectory Tracking"

**Goal:** Fly a figure-8.
1.  **Input:** Parametric equations.
    *   $x_d(t) = \sin(t)$
    *   $y_d(t) = \sin(t) \cos(t)$
2.  **Controller:** Feed $x_d, y_d$ into the Position Loop.
3.  **Feedforward:** Calculate velocity/acceleration of the path and add to controller ($u_{ff}$). This reduces tracking lag (Lag Error).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Flip on Takeoff"
*   **Cause:** Motors mappings are wrong. Front-Left motor is connected to Rear-Right output.
*   **Result:** The PID tries to correct Pitch but makes it worse. Positive feedback loop.
*   **Fix:** Verify Mixer map and Motor direction.

#### 2. "Flyaway"
*   **Cause:** Vibration affecting the IMU (specifically Accelerometer).
*   **Result:** The EKF thinks "Down" is "Sideways" and tries to correct, accelerating into infinity.
*   **Fix:** Soft mount the flight controller.

---

## ⚡ Optimization: Nonlinear Control

PID assumes linear dynamics (small angles).
*   **Geometric Control:** Works on $SO(3)$ manifold (Rotation matrices).
*   Allows the drone to do loops and recover from being upside down.
*   PID fails at 90 degree pitch (Gimbal Lock in Euler angles).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can a quadrotor hover upside down?
    *   **A:** Only if the props are reversible (3D flying) or if thrust vectoring. Standard quads cannot generate negative thrust.
2.  **Q:** What is "Yaw Authority"?
    *   **A:** Yaw is generated by drag torque. It is much weaker than Roll/Pitch torque (lever arm). Yaw is the slowest axis.
3.  **Q:** Why 400Hz update rate?
    *   **A:** The "Rate Loop" needs to be faster than the motor response time. Small props speed up/slow down in ~10-20ms.

### Challenge Task
> **Task:** Wind Disturbance.
> 1. Add constant force $F_x = 2N$ in `dynamics()`.
> 2. Observe "Steady State Error" in X position with P-Controller.
> 3. Add **Integrator** ($I$ term) to Position loop.
> 4. Verify Error goes to zero (Drone leans into the wind).

---

## 📚 Further Reading
- **Kumar Robotics (UPenn):** "Minimum Snap Trajectory Generation".
- **PX4 Docs:** "mc_pos_control" diagram.

---

**Day 120 Complete**
