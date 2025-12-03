# Day 155: Model-in-the-Loop (MIL)
## Phase 4: ADAS & Robotics Systems | Week 23: Testing & Validation

---

> **📝 Day 155 Focus:**
> Before we write C++ code, before we compile, and certainly before we drive, we must prove the math works. **Model-in-the-Loop (MIL)** is the first stage of testing. We simulate the logic and the plant (vehicle) in a high-level language (Python/Matlab) to verify requirements.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the V-Model of Systems Engineering.
2.  **Define** MIL, SIL, HIL, and VIL.
3.  **Create** a Plant Model (Vehicle Dynamics) in Python.
4.  **Implement** a Controller Model (PID/MPC).
5.  **Run** a Closed-Loop Simulation to verify requirements (e.g., Overshoot < 5%).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Control Theory:** Transfer Functions, State Space.
-   **Python:** `scipy.integrate` (ODE Solver).

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `numpy`, `matplotlib`, `scipy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The V-Model

1.  **Requirements Analysis:** "Car must stop within 30m".
2.  **System Design:** "Use AEB with -0.5g braking".
3.  **Implementation:** Writing Code.
4.  **Verification (The Right Side):**
    -   **MIL:** Does the math work?
    -   **SIL:** Does the compiled code work?
    -   **HIL:** Does the ECU hardware work?
    -   **VIL:** Does the car work?

### 🔹 Part 2: Model-in-the-Loop (MIL)

-   **The Model:** A mathematical representation of the algorithm (Controller) and the physical world (Plant).
-   **Goal:** Quick iteration. No compilation, no real-time constraints.
-   **Tools:** MATLAB/Simulink is industry standard. We will use Python for accessibility.

### 🔹 Part 3: Requirement Tracing

Every test must link back to a requirement.
-   **Req ID:** `REQ-LAT-001`: "Lateral error shall not exceed 0.2m on straight roads."
-   **Test:** Run simulation, check `max(abs(error)) < 0.2`.

---

## 💻 Implementation: MIL Simulation of ACC

**Scenario:**
-   **Plant:** Longitudinal Vehicle Dynamics ($F=ma$).
-   **Controller:** Adaptive Cruise Control (PID).
-   **Requirement:** Maintain 30m distance. No overshoot > 2m.

### 🛠️ Setup
Create `week23_day155` and `mil_acc.py`.

```bash
mkdir -p ~/ros2_ws/src/week23_day155
cd ~/ros2_ws/src/week23_day155
touch mil_acc.py
```

### 👨‍💻 Code: Python MIL

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- 1. The Plant (Vehicle Physics) ---
def vehicle_dynamics(state, t, u, mass=1500.0, drag=0.3):
    # state: [position, velocity]
    # u: force (throttle/brake)
    x, v = state
    
    # F_net = F_engine - F_drag
    # F_drag = 0.5 * rho * Cd * A * v^2 (Simplified as c*v)
    f_drag = drag * v 
    
    a = (u - f_drag) / mass
    
    dxdt = v
    dvdt = a
    return [dxdt, dvdt]

# --- 2. The Controller (Algorithm) ---
class ACC_Controller:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
        
    def compute(self, target_dist, current_dist, dt):
        error = current_dist - target_dist
        
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        
        return output # Force

# --- 3. The Simulation Loop ---
def main():
    # Parameters
    dt = 0.1
    t_end = 50.0
    steps = int(t_end / dt)
    time = np.linspace(0, t_end, steps)
    
    # Initial State
    ego_state = [0.0, 20.0] # x=0, v=20 m/s
    lead_car_x = 50.0 # 50m ahead
    lead_car_v = 20.0 # Constant speed
    
    # Controller
    acc = ACC_Controller(kp=500.0, ki=10.0, kd=500.0)
    
    # History
    history_dist = []
    history_v = []
    
    print("Running MIL Simulation...")
    for i in range(steps):
        # 1. Sensor Reading (Ideal)
        dist = lead_car_x - ego_state[0]
        
        # 2. Control Logic
        # Target: 30m gap
        force = acc.compute(30.0, dist, dt)
        
        # Actuator Saturation (Engine Limit)
        force = np.clip(force, -5000, 5000)
        
        # 3. Physics Step (Plant)
        # Solve ODE for one step
        next_state = odeint(vehicle_dynamics, ego_state, [0, dt], args=(force,))
        ego_state = next_state[1]
        
        # Update Lead Car (Scenario)
        if i > 100: lead_car_v = 15.0 # Brake at t=10s
        lead_car_x += lead_car_v * dt
        
        # Log
        history_dist.append(dist)
        history_v.append(ego_state[1])
        
    # --- 4. Verification ---
    min_dist = np.min(history_dist)
    print(f"Minimum Distance: {min_dist:.2f} m")
    
    # Requirement Check
    if min_dist < 28.0: # Allow 2m overshoot
        print("TEST FAILED: Overshoot > 2m")
    else:
        print("TEST PASSED")
        
    # Plot
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, history_dist)
    plt.axhline(30.0, color='r', linestyle='--', label='Target')
    plt.title("Distance to Lead Car")
    plt.ylabel("Meters")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time, history_v, label='Ego')
    plt.plot(time, [20 if t < 10 else 15 for t in time], 'g--', label='Lead')
    plt.title("Velocity")
    plt.xlabel("Time (s)")
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: Tuning in MIL

### Lab Objectives
1.  **Run the script.**
    -   **Observation:** The car slows down when the lead car brakes.
2.  **Break the Requirement:**
    -   Set `kp = 100` (Too weak).
    -   **Result:** Distance drops below 28m. Test Fails.
    -   Set `kp = 5000` (Too strong).
    -   **Result:** Oscillation. Comfort requirement fails.
3.  **Lesson:** MIL is where you tune gains. It takes milliseconds. Doing this in a real car takes hours and is dangerous.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Numerical Instability
**Symptom:** Values explode to infinity.
**Cause:** `dt` is too large for the dynamics.
**Solution:** Reduce `dt` (e.g., 0.01s) or use a better solver (Runge-Kutta 4).

#### 2. Unrealistic Model
**Symptom:** Car stops instantly.
**Cause:** Infinite braking force allowed.
**Solution:** Always add `np.clip` to simulate physical limits (Actuator Saturation).

---

## ⚡ Optimization & Best Practices

### 1. Monte Carlo Simulation
Don't run just one scenario.
-   Run 1000 times with random parameters (Mass 1000-2000kg, Friction 0.5-1.0).
-   Ensure the controller passes in 99.9% of cases.

### 2. Code Generation
-   In MATLAB/Simulink, you can export the block diagram directly to C++ code.
-   This ensures the "MIL" logic is exactly what runs on the "HIL" hardware.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the main advantage of MIL?
    *   **A:** Speed and Safety. You can catch logic errors early when they are cheap to fix.
2.  **Q:** What is a "Plant Model"?
    *   **A:** The mathematical equations that describe how the system (car) reacts to inputs (steering/gas).
3.  **Q:** How does MIL differ from SIL?
    *   **A:** MIL tests the *logic* (Math). SIL tests the *implementation* (C++ Code).

### Challenge Task
**Task:** Latency Simulation.
1.  Add a delay buffer to the controller inputs.
2.  `sensed_dist = history_dist[-5]` (0.5s delay).
3.  Observe how delay destabilizes the PID loop.

---

## 📚 Further Reading & References
-   [ISO 26262 V-Model](https://en.wikipedia.org/wiki/V-Model)
-   [Python Control Systems Library](https://python-control.readthedocs.io/)

---

**Day 155 Complete** | Phase 4: ADAS & Robotics Systems | Week 23: Testing & Validation
