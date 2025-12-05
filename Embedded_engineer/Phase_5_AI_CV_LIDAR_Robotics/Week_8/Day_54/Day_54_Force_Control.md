# Day 54: Force Control (Impedance/Admittance)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 8: Advanced Manipulation

---

> **📝 Content Creator Instructions:**
> Robots are strong. If they hit a wall in Position Mode, they break the wall (or the motor).
> - **Focus:** Stiffness, Damping, Impedance Control, and Admittance Control.
> - **Code:** Simulating a "Virtual Spring" Controller for safe interaction.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Position Control (Stiff) and Force Control (Compliant).
2.  **Derive** the Impedance Control Law: $\tau = J^T (K_p \tilde{x} + K_d \dot{\tilde{x}})$.
3.  **Implement** Admittance Control: Measuring Force $F_{ext} \to$ Modifying Target Velocity $v_{cmd}$.
4.  **Execute** a Surface Wiping task (Keep $F_z = 10N$ while moving in $XY$).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Force/Torque Sensor (Simulated).

### Software Environment
```bash
pip install numpy matplotlib control
```

### Prior Knowledge
- Spring-Mass-Damper Systems.
- Jacobian Transpose (Day 51).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Stiff vs Compliant

*   **Position Control:** Error $e \to 0$. $K_p \to \infty$.
    *   If obstructed, Torque $\to \infty$. Dangerous.
*   **Force Control:** Force $F \to F_{ref}$.
    *   If nothing to touch, Velocity $\to \infty$. Dangerous.
*   **Impedance Control:** The robot behaves like a **Spring-Damper**.
    *   $F_{ext} = K(x_{des} - x) + D(\dot{x}_{des} - \dot{x})$.
    *   If pushed, it deviates like a spring. When released, it bounces back.

### 🔹 Part 2: Impedance vs Admittance

1.  **Impedance (Torque Based):**
    *   Input: Position Deviation ($x$). Output: Force ($\tau$).
    *   Requires: Torque-controllable motors (e.g., Franka, Kuka iiwa).
    *   Physics: $F = Z(x)$.
2.  **Admittance (Position Based):**
    *   Input: Measured Force ($F_{ext}$). Output: Position Change ($\Delta x$).
    *   Requires: F/T Sensor. Works on stiff position-controlled robots (e.g., UR5).
    *   Logic: "I feel 10N pushing me back. I will move back 1cm to relieve pressure."
    *   Physics: $x = Y(F) = F/Z$.

---

## 💻 Implementation: Admittance Controller

We will implement Admittance Control for a 1-DOF joint (Linear Actuator).
Goal: Maintain contact force $F_{ref}$ against a wall.

### 🛠️ Project Structure
```text
day54_force/
├── src/
│   ├── admittance.py
│   └── environment.py
└── run_simulation.py
```

### 👨‍💻 Environment (`src/environment.py`)

Simulates a wall at $x=0.5$.

```python
class WallEnv:
    def __init__(self, wall_loc=0.5, stiffness=1000.0):
        self.wall_loc = wall_loc
        self.stiffness = stiffness # N/m
        
    def get_force(self, pos):
        if pos > self.wall_loc:
            # Hooke's Law: F = k * penetration
            penetration = pos - self.wall_loc
            return self.stiffness * penetration
        return 0.0
```

### 👨‍💻 Controller (`src/admittance.py`)

A Mass-Spring-Damper system in software.
$$ M_d \ddot{x}_r + D_d \dot{x}_r + K_d (x_r - x_{des}) = F_{ext} - F_{ref} $$
Simplified (no accel term):
$$ D_d \dot{x}_r = (F_{ext} - F_{ref}) - K_d (x_r - x_{des}) $$

```python
import numpy as np

class AdmittanceController:
    def __init__(self, M=1.0, D=20.0, K=0.0, dt=0.01):
        self.M = M # Virtual Mass
        self.D = D # Virtual Damping
        self.K = K # Virtual Stiffness (0 for pure force tracking)
        self.dt = dt
        
        self.x_ref = 0.0
        self.v_ref = 0.0
        
    def step(self, f_meas, f_target):
        # Equation: M*a + D*v + K*x = F_error
        # F_error = f_target - f_meas (Force we WANT to add to system)
        
        f_err = f_target - f_meas
        
        # Solving for Acceleration (a)
        # a = (F_err - D*v - K*x) / M
        
        accel = (f_err - self.D * self.v_ref - self.K * self.x_ref) / self.M
        
        # Integrate
        self.v_ref += accel * self.dt
        self.x_ref += self.v_ref * self.dt
        
        return self.x_ref, self.v_ref
```

### 👨‍💻 Simulation Loop (`run_simulation.py`)

```python
import matplotlib.pyplot as plt
from src.environment import WallEnv
from src.admittance import AdmittanceController

env = WallEnv(wall_loc=0.05) # Wall at 5cm
ctrl = AdmittanceController(M=2.0, D=50.0, K=0.0) # Pure damping/mass behavior

# Initial State
robot_pos = 0.0 # Start at 0
f_target = 10.0 # Want to push wall with 10N

history_pos = []
history_force = []

for t in range(200):
    # 1. Physics: Interaction
    f_meas = env.get_force(robot_pos)
    
    # 2. Control: Compute reference offset
    # Note: x_ref is output relative to "Collision Point" ideally
    # Here, output is absolute position command
    
    # We want robot to move Forward until Force = 10N
    delta_x, _ = ctrl.step(f_meas, f_target)
    
    # For simulation, assume inner position loop is perfect:
    robot_pos = delta_x 
    
    history_pos.append(robot_pos)
    history_force.append(f_meas)

# Plot
fig, ax1 = plt.subplots()

ax1.set_xlabel('Time')
ax1.set_ylabel('Position (m)', color='tab:blue')
ax1.plot(history_pos, color='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Force (N)', color='tab:red')
ax2.plot(history_force, color='tab:red')
plt.title("Force Control Response")
plt.show()
```

### 3. Expected Output
*   **Time 0-10:** Force = 0. Pos increases (moving towards wall due to Error 10N).
*   **Time 10:** Hits Wall. Force spikes.
*   **Time 11+:** Controller sees Force > 0. Slows down.
*   **Steady State:** Robot penetrates wall just enough ($10N / 1000_{stiff} = 1cm$) to maintain 10N. Force settles at 10N.

---

## 🔬 Lab Exercise: The "Peg-in-Hole"

### 1. Lab Objectives
- Insert a square peg ($10mm$) into a square hole ($10.5mm$).
- **Position Control:** Slight misalignment ($1mm$) $\to$ Jamming $\to$ Massive Force $\to$ Error.
- **Admittance Control:**
    - Robot feels X-force upon contact.
    - Controller moves robot in -X to comply.
    - Robot naturally "slides" into the hole.
- **Spiral Search:** If contact Z is high (surface), spiral XY to find hole (Z drop).

---

## 🚀 Project: "Surface Polishing"

**Goal:** Polish a curved car hood.
1.  **Trajectory:** Follow a zig-zag pattern on the surface.
2.  **Constraint:** Maintain Normal Force $F_n = 5N$.
3.  **Hybrid Control:**
    - Tangential Axis ($X_{surf}$): Position Control (Velocity).
    - Normal Axis ($Z_{surf}$): Force Control.
4.  **Result:** Even polishing despite curvature errors.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Instability / Vibrating"
*   **Symptom:** Gripper bangs against the wall repeatedly $\to$ BANG BANG BANG.
*   **Cause:** Damping ($D$) is too low, or Inner Loop Latency is too high. Admittance control is unstable on stiff environments if time delay exists.
*   **Fix:** Increase Damping. Lower Virtual Stiffness.

#### 2. "Floating Away"
*   **Symptom:** In free space (Force=0), robot drifts to infinity.
*   **Cause:** $F_{target} > 0$. Controller accelerates to find resistance.
*   **Fix:** Velocity Limits. Or switch to Position Mode when $F_{meas} \approx 0$.

---

## ⚡ Optimization: Gravity Compensation

Before doing Force Control, we must cancel Gravity.
*   $\tau_{motor} = \tau_{dyn} + G(q) + \tau_{ext}$.
*   If we don't subtract $G(q)$ (Payload weight), the robot thinks the arm weight is an external force and "yields" (falls down).
*   **Torque Sensors:** Modern sensors subtract arm weight automatically, but Payload weight (Gripper + Object) must be identified.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is Admittance safer for humans?
    *   **A:** If a human pushes the robot, $F_{meas}$ increases. Controller moves robot *away* from human implies "yielding".
2.  **Q:** What is "Stiffness"?
    *   **A:** Low Stiffness = Soft/Springy. High Stiffness = Rigid.
3.  **Q:** Can I do this with a Stepper Motor?
    *   **A:** No torque sensing. Only if you add an external F/T sensor at the wrist.

### Challenge Task
> **Task:** Weighing Scale.
> 1. Hold an object.
> 2. Measure Joint Torques $\tau$.
> 3. Use $J^T F = \tau$ to solve for Force $F_z$.
> 4. Divide by $g$. Output Mass.

---

## 📚 Further Reading
- **Impedance Control:** Hogan (1985). Classic paper.
- **Modern Robotics:** Chapter on Force Control.

---

**Day 54 Complete**
