# Day 123: AUV Dynamics (Buoyancy/Drag)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 18: Aerial & Underwater Robotics

---

> **📝 Content Creator Instructions:**
> In space, no one can hear you scream. Underwater, everyone hears the sonar.
> - **Focus:** Hydrodynamics (Fossen's Equations), Buoyancy vs Gravity, Added Mass, and Quadratic Drag.
> - **Code:** A Python simulator `auv_sim.py` that models a BlueROV2-like vehicle, demonstrating the passive stability (Righting Moment) when $z_B < z_G$.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Derive** the Hydrostatic Restoring Forces (Why ships don't flip).
2.  **Explain** "Added Mass" (Pulling water with you).
3.  **Simulate** Quadratic Drag ($F_d = -0.5 \rho C_d A v |v|$).
4.  **Implement** a Depth Controller using pressure feedback.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Simulation). Gazebo with UUV Simulator plugin (Optional).

### Software Environment
```bash
pip install numpy matplotlib scipy
```

### Prior Knowledge
- Rigid Body Dynamics.
- Density of Water ($\rho = 1000 kg/m^3$).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Hydrostatics

*   **Gravity ($W$):** Acts downwards at Center of Gravity (CG).
*   **Buoyancy ($B$):** Acts upwards at Center of Buoyancy (CB).
    *   $B = \rho g \nabla$ (Archimedes' Principle, where $\nabla$ is Displaced Volume).
*   **Stability:** If CB is *above* CG, the vehicle is passively stable in Roll/Pitch (Metacentric Height). AUVs are designed this way (Heavy batteries at bottom, Foam at top).

### 🔹 Part 2: Hydrodynamics (Drag)

Air drag is linear at low speeds. Water drag is **Quadratic**.
*   $D = -\frac{1}{2} \rho C_d A v |v|$.
*   This provides huge damping. AUVs stop almost instantly when thrusters cut off.

### 🔹 Part 3: Added Mass

Water is heavy. When you accelerate an AUV, you also accelerate the water around it.
*   **Concept:** $F = (m + m_{added}) a$.
*   For a sphere, $m_{added} = 0.5 \times m_{displaced}$. It acts like "Virtual Inertia".

---

## 💻 Implementation: AUV Simulator (Fossen Model)

We will simulate a 4-DOF AUV (Surge, Sway, Heave, Yaw). Roll/Pitch assume stable.

### 🛠️ Project Structure
```text
day123_auv/
├── src/
│   ├── auv_sim.py
└── output/
    ├── depth_response.png
```

### 👨‍💻 Physics Engine (`src/auv_sim.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class AUV:
    def __init__(self):
        # Physical Parameters (BlueROV2 approx)
        self.mass = 11.0 # kg
        self.buoyancy = 11.2 * 9.81 # Newtons (Slightly positive buoyant)
        self.weight = self.mass * 9.81
        
        # State [x, y, z, psi, u, v, w, r]
        # Pos: x,y,z (NED), Yaw: psi
        # Vel: u,v,w (Body), YawRate: r
        self.state = np.zeros(8)
        self.dt = 0.05
        
        # Drag Coefficients (Linear + Quadratic)
        self.Xu = -5.0; self.Xuu = -10.0
        self.Yv = -5.0; self.Yvv = -10.0
        self.Zw = -5.0; self.Zww = -10.0
        self.Nr = -1.0; self.Nrr = -2.0
        
        # Added Mass (Simplified - Diagonal)
        self.Xudot = -2.0 
        self.Yvdot = -5.0
        self.Zwdot = -5.0 # High added mass in heave (flat shape)
        self.Nrdot = -0.5
        
        # Inertia Matrix (Mass + Added Mass)
        self.M = np.diag([
            self.mass - self.Xudot,
            self.mass - self.Yvdot,
            self.mass - self.Zwdot,
            1.0 - self.Nrdot # Yaw inertia
        ])
        
        self.Minv = np.linalg.inv(self.M)

    def dynamics(self, tau):
        # tau = [Tx, Ty, Tz, Tpsi] (Thruster inputs)
        x, y, z, psi, u, v, w, r = self.state
        
        # 1. Hydrostatic Forces (Restoring)
        # Weight acts down, Buoyancy acts up.
        # In Body Frame:
        # F_restoring_z = -(W - B)  (If W > B, sink. If B > W, float)
        # Note: AUV coordinate system Z is DOWN.
        # So Positive Depth = Deeper.
        # Weight is Positive Force in Z. Buoyancy is Negative Force in Z.
        F_hydro_z = (self.weight - self.buoyancy)
        
        # 2. Damping Forces (Drag)
        # F_drag = X_u * u + X_uu * u * |u|
        D = np.array([
            self.Xu * u + self.Xuu * u * abs(u),
            self.Yv * v + self.Yvv * v * abs(v),
            self.Zw * w + self.Zww * w * abs(w),
            self.Nr * r + self.Nrr * r * abs(r)
        ])
        
        # 3. Coriolis (Ignore for slow speeds)
        
        # Total Forces
        F_total = tau + D + np.array([0, 0, F_hydro_z, 0])
        
        # Acceleration
        # M * acc = F_total
        acc = self.Minv @ F_total
        
        # Kinematics (Body Velocity -> Earth Velocity)
        # x_dot = u cos(psi) - v sin(psi)
        # y_dot = u sin(psi) + v cos(psi)
        # z_dot = w
        # psi_dot = r
        
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        
        vel_earth = np.array([
            u*cpsi - v*spsi,
            u*spsi + v*cpsi,
            w,
            r
        ])
        
        return vel_earth, acc

    def step(self, tau):
        # Euler Integration
        vel_dot, acc = self.dynamics(tau)
        
        # Update Position
        self.state[0:4] += vel_dot * self.dt
        # Update Velocity
        self.state[4:8] += acc * self.dt
        
        return self.state

    def depth_controller(self, target_depth):
        # PI Controller
        kp = 50.0
        ki = 5.0
        
        current_depth = self.state[2]
        error = target_depth - current_depth
        
        # Integrator (Simple accumulation)
        if not hasattr(self, 'int_err'): self.int_err = 0
        self.int_err += error * self.dt
        
        thrust_z = kp * error + ki * self.int_err
        
        # Limit Thrust
        thrust_z = np.clip(thrust_z, -50, 50)
        
        return thrust_z

def main():
    sub = AUV()
    history = []
    times = []
    
    target_z = 10.0 # Dive to 10m
    
    for t in np.arange(0, 30.0, sub.dt):
        # Controller
        tz = sub.depth_controller(target_z)
        
        # Input Vector
        tau = np.array([10.0, 0.0, tz, 0.0]) # Move forward (10N) and Dive
        
        s = sub.step(tau)
        history.append(s.copy())
        times.append(t)
        
    history = np.array(history)
    
    # Plot
    fig, axs = plt.subplots(2, 1)
    
    # Depth
    axs[0].plot(times, history[:, 2], label='Depth (Z)')
    axs[0].axhline(target_z, color='r', linestyle='--', label='Target')
    axs[0].set_ylabel('Depth (m)')
    axs[0].invert_yaxis() # Depth convention
    axs[0].legend()
    
    # Velocity (Surge)
    axs[1].plot(times, history[:, 4], label='Surge Speed (u)')
    axs[1].set_ylabel('Speed (m/s)')
    axs[1].set_xlabel('Time (s)')
    axs[1].legend()
    
    plt.savefig("output/depth_response.png")
    print("Simulation Complete.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Emergency Ascent"

### 1. Lab Objectives
- **Run:** Sim. Observe depth settles at 10m.
- **Fail:** At $t=15s$, set `tau` to `[0,0,0,0]` (Power Failure).
- **Observe:** The sub slowly rises to the surface ($z \to 0$).
- **Why?:** Because $W < B$ (Positive Buoyancy).
- **Modify:** Set `buoyancy < weight`.
- **Result:** The sub sinks to the abyss (" Davy Jones' Locker") upon failure.
- **Lesson:** Always design AUVs to be positively buoyant!

---

## 🚀 Project: "Station Keeping"

**Goal:** Hold position against a current.
1.  **Current:** Add constant velocity to water vector in drag equation.
    *   $v_{rel} = v_{vehicle} - v_{current}$.
2.  **Controller:** PID on $X$ and $Y$ position.
3.  **Result:** Thrusters fight the current to stay at $(0,0)$.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Unstable Depth"
*   **Cause:** Lag in pressure sensor + high $K_p$.
*   **Fix:** Use $K_d$ (D-term). But wait! Depth differentiation is noisy. Use Vertical Velocity from DVL (Doppler Velocity Log) if available for D-term.

#### 2. "Roll Instability"
*   **Cause:** Ideally, roll is passive. But torque from propeller spin can cause roll.
*   **Fix:** Use counter-rotating propellers (CW/CCW pairs) to cancel torque.

---

## ⚡ Optimization: Thruster Allocation

Like Quadrotors, AUVs have a mixing matrix.
BlueROV2 has 6 or 8 thrusters ("Heavy" config).
*   **Matrix:** Maps $(F_x, F_y, F_z, \tau_x, \tau_y, \tau_z)$ to 8 PWM signals.
*   **Optimization:** Some configurations (Vector maneuvering) allow movement in any direction (6-DOF fully actuated).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between AUV and ROV?
    *   **A:** AUV is Autonomous (No cable). ROV is Remotely Operated (Tethered). ROVs don't worry about battery life as much.
2.  **Q:** Why is Drag Quadratic?
    *   **A:** Reynolds number. At macroscopic scales in water, inertial forces dominate viscous forces.
3.  **Q:** What is "Metacenter"?
    *   **A:** The theoretical point that determines stability. If CG is below Metacenter, it's stable.

### Challenge Task
> **Task:** Pitch PID.
> 1. Add Pitch ($\theta$) and Pitch Rate ($q$) to the simulator state.
> 2. Add restoring moment $M_{rest} = - \overline{BG} W \sin(\theta)$ (Pendulum effect).
> 3. Verify simulator oscillates like a pendulum if disturbed.

---

## 📚 Further Reading
- **Fossen:** "Handbook of Marine Craft Hydrodynamics and Motion Control" (The Bible of AUVs).
- **UUV Simulator:** Gazebo plugins for underwater.

---

**Day 123 Complete**
