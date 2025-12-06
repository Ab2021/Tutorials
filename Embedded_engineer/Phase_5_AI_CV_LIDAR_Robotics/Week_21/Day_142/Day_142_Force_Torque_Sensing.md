# Day 142: Force/Torque Sensing (Admittance Control)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 21: Collaborative Robotics (Cobots)

---

> **📝 Content Creator Instructions:**
> Don't fight the human.
> - **Focus:** Impedance vs Admittance Control, Mass-Spring-Damper virtual models, and Force/Torque (F/T) Sensor integration.
> - **Code:** A Python simulation `admittance_sim.py` where a robot joint behaves like a "Heavy Door" (Virtual Inertia) or a "Spring" (Virtual Stiffness) in response to external forces.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Distinguish** Impedance Control (Input: Pos, Output: Force) vs Admittance Control (Input: Force, Output: Pos).
2.  **Tune** Virtual Mass ($M$), Damping ($B$), and Stiffness ($K$).
3.  **Process** raw signals from a 6-Axis F/T sensor (Removing Gravity Bias).
4.  **Implement** a compliant joint controller.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Simulation (PyBullet) or Python. Real F/T sensor is expensive ($5k).

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Newton's Second Law.
- Differential Equations.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Philosophy of Touch

Industrial robots are stiff (Position Control). If you push them, they push back with max torque.
Cobots are compliant.
*   **Active Compliance:** Using sensors and software to mimic a spring.
*   **Passive Compliance:** Using actual springs (Series Elastic Actuators).

### 🔹 Part 2: Admittance Control (Position-Based)

Most industrial cobots (UR) use this. The robot accepts a Force input ($F_{ext}$) and calculates a Desired Trajectory ($x_d$).
$$ M \ddot{x}_d + B \dot{x}_d + K x_d = F_{ext} $$
*   **Virtual Mass ($M$):** How heavy it feels.
*   **Virtual Damping ($B$):** How much friction it has (prevents oscillation).
*   **Virtual Stiffness ($K$):** Does it return to center? ($K=0$ for Free Drive).

### 🔹 Part 3: Gravity Compensation

Before you feel the human, you feel the tool.
$$ F_{sensor} = F_{human} + F_{gravity} + F_{inertial} $$
*   You must subtract $m \cdot g \cdot R(\theta)$ dynamically to find $F_{human}$.

---

## 💻 Implementation: The Virtual Mass

We simulate a single joint being pushed by a human.

### 🛠️ Project Structure
```text
day142_force/
├── src/
│   ├── admittance_sim.py
└── output/
    ├── force_response.png
```

### 👨‍💻 Admittance Simulation (`src/admittance_sim.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class AdmittanceController:
    def __init__(self, M, B, K):
        self.M = M # Virtual Mass (kg)
        self.B = B # Virtual Damping (Ns/m)
        self.K = K # Virtual Stiffness (N/m)
        
        # Internal Interaction State (Virtual Position)
        self.x_v = 0.0
        self.v_v = 0.0
        self.a_v = 0.0
        self.dt = 0.001

    def step(self, f_ext):
        # Solves: M*a + B*v + K*x = F_ext
        # a = (F_ext - B*v - K*x) / M
        
        self.a_v = (f_ext - self.B * self.v_v - self.K * self.x_v) / self.M
        
        # Integrate
        self.v_v += self.a_v * self.dt
        self.x_v += self.v_v * self.dt
        
        return self.x_v, self.v_v

def main():
    # 1. Setup: Feeling "Heavy" but smooth
    # High Torque motors can mask their inertia, but here we ADD inertia?
    # No, usually M_virtual < M_physical to make it feel light.
    # Or M_virtual describes the target behavior.
    
    controller = AdmittanceController(M=5.0, B=10.0, K=0.0) # K=0 -> Free Float
    
    duration = 5.0
    times = np.arange(0, duration, controller.dt)
    
    forces = np.zeros_like(times)
    positions = np.zeros_like(times)
    velocities = np.zeros_like(times)
    
    # 2. Scenario: Human pushes for 1 second, then lets go
    # Push with 10N from t=1 to t=2
    start_idx = int(1.0 / controller.dt)
    end_idx = int(2.0 / controller.dt)
    forces[start_idx:end_idx] = 10.0
    
    print("Simulating Admittance Interaction...")
    
    for i, t in enumerate(times):
        f = forces[i]
        
        # Admittance Step
        pos, vel = controller.step(f)
        
        positions[i] = pos
        velocities[i] = vel
        
        # Note: In a real robot, 'pos' is sent to the Servos as the Target Position.
        # The Low-Level PID loop makes the robot go there.
    
    # 3. Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    ax1.plot(times, forces, 'r-', label='Human Force (N)')
    ax1.set_ylabel('Force (N)')
    ax1.legend()
    ax1.grid()
    
    ax2.plot(times, positions, 'b-', label='Robot Position (m)')
    ax2.plot(times[start_idx-100:end_idx+200], velocities[start_idx-100:end_idx+200], 'g--', label='Velocity (m/s)')
    ax2.set_ylabel('Motion')
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.grid()
    
    plt.savefig("output/force_response.png")
    print("Sim Complete.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Tuning the Feel"

### 1. Lab Objectives
- **Run:** Sim. Observe that after force stops ($t=2$), the robot coasts and slows down (Damping). It does not return to 0. (Free Mode).
- **Modify:** Set $K = 50.0$.
- **Result:** The robot pushes back. When force stops, it springs back to Position 0.
- **Modify:** Set $M = 0.5$ (Low Inertia).
- **Result:** Robot accelerates wildly with small touches. "Twitchy".
- **Safety:** Low Damping ($B$) causes oscillations. Always maintain critically damped or overdamped ratios ($B \ge 2\sqrt{MK}$).

---

## 🚀 Project: "Tool Weight Estimation"

**Goal:** Auto-calibrate the tool.
1.  **Move:** Robot to 4 different poses.
2.  **Measure:** F/T sensor readings.
3.  **Solve:** Least Squares ($Ax = b$) to find Tool Mass ($m$) and Center of Mass ($x,y,z$).
4.  **Result:** Updates Gravity Compensation model.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Robot Drifts Up"
*   **Cause:** Gravity Compensation error. The model thinks the tool is heavier than it is, so it applies upward force.
*   **Fix:** Accurate payload identification.

#### 2. "Buzzing/Vibration"
*   **Cause:** Admittance Loop Frequency too low or Gains too high.
*   **Fix:** Run Admittance loop at >500Hz. If stiff (High K), reduce simulation timestep.

---

## ⚡ Optimization: Force Bandwidth

F/T sensors are noisy.
*   **Filter:** Low Pass Filter (Cutoff 5-10Hz).
*   **Impact:** Delays the robot's reaction. It feels "Spongy".
*   **Solution:** Better sensors with internal DSP.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Admittance vs Impedance?
    *   **A:** Admittance: Measure Force $\to$ Output Motion (Good for Stiff Robots/Pis). Impedance: Measure Motion $\to$ Output Torque (Good for Backdrivable Robots/Torque Control).
2.  **Q:** What happens if B=0?
    *   **A:** The robot never stops moving after a push (Frictionless surface). Dangerous.
3.  **Q:** Why 6-Axis?
    *   **A:** Force (XYZ) + Torque (Roll/Pitch/Yaw). You need torque to do screw-driving or peg-in-hole alignment.

### Challenge Task
> **Task:** Wall Following.
> 1. Set $K_x = 0$ (Free X), $K_y = 1000$ (Stiff Y).
> 2. Push robot against a wall (Y-axis).
> 3. Robot maintains contact force but slides freely in X.

---

## 📚 Further Reading
- **Hogan, N.:** "Impedance Control: An Approach to Manipulation".
- **Modern Robotics:** Chapter on Force Control.

---

**Day 142 Complete**
