# Day 128: Pneumatic Control (Hysteresis & Pressure)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 19: Soft Robotics & Bio-Inspired Control

---

> **📝 Content Creator Instructions:**
> Air is springy.
> - **Focus:** Compressibility of Gas, PWM Control of Solenoid Valves (Inflate/Exhaust), Pressure Sensors, and Hysteresis in Elastomers (Rubber memory).
> - **Code:** A Python simulation `pneumatic_sim.py` that models the charging/discharging of a soft chamber and implements a High-Speed Bang-Bang Controller with Deadband.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** why P=nRT makes control hard (Time delays, compressibility).
2.  **Model** the Hysteresis loop (Inflation path $\neq$ Deflation path).
3.  **Implement** PWM control for binary solenoid valves.
4.  **Design** a Pressure Regulator algorithm (PID vs Hysteresis Control).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Optional: Soft actuator + Arduino + Mosfet + Solenoids + Pressure Sensor.
- Simulation: Python.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Thermodynamics (Ideal Gas Law).
- PWM (Pulse Width Modulation).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Physics of Air

Hydraulic (Water) is stiff. Pneumatic (Air) is compliant.
*   **Flow Rate:** Proportional to pressure difference ($\Delta P$).
*   **Chamber Dynamics:**
    $$ \dot{P} = \frac{k}{V} (\dot{m}_{in} - \dot{m}_{out}) $$
    (Simplified).
*   **Time Constant:** Filling a balloon takes time. Pressure doesn't jump instantly.

### 🔹 Part 2: Actuation Hardware

*   **Solenoid Valves:** Binary (On/Off).
    *   **Inlet Valve:** Connects to Compressor (Source).
    *   **Exhaust Valve:** Connects to Atmosphere (Vent).
*   **PWM Control:**
    *   To get 50% flow: Open 10ms, Close 10ms.
    *   Frequency matters (Too slow = jerky. Too fast = valve overheat).

### 🔹 Part 3: Hysteresis

Rubber stretches.
*   **Loading:** Curve A (Force vs Strain).
*   **Unloading:** Curve B.
*   **Area:** Energy lost as heat.
*   **Control Implication:** To reach Position X, you need Pressure $P_1$ if inflating, but Pressure $P_2$ if deflating. $P_1 > P_2$.

---

## 💻 Implementation: Pressure Controller Simulator

We simulate a single chamber system.

### 🛠️ Project Structure
```text
day128_pneumatics/
├── src/
│   ├── pneumatic_sim.py
└── output/
    ├── step_response.png
```

### 👨‍💻 Physics & Control (`src/pneumatic_sim.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class PneumaticChamber:
    def __init__(self):
        self.pressure = 0.0 # PSI
        self.source_pressure = 40.0 # PSI (Compressor)
        self.atm_pressure = 0.0
        
        self.volume = 100.0 # mL (Constant-ish)
        
        # Valves (0 = Closed, 1 = Open)
        self.inlet_state = 0
        self.exhaust_state = 0
        
        self.dt = 0.001 # 1ms simulation step

    def step(self):
        # Flow Rate ~ sqrt(Delta P) usually, but linear approx for simplicity
        # Flow In
        flow_in = 0
        if self.inlet_state:
            flow_in = 0.5 * (self.source_pressure - self.pressure)
            
        # Flow Out
        flow_out = 0
        if self.exhaust_state:
            flow_out = 0.5 * (self.pressure - self.atm_pressure)
            
        # Pressure Change (dP = Flow * dt / Capacity)
        dp = (flow_in - flow_out) * self.dt
        self.pressure += dp
        
        # Noise
        self.pressure += np.random.normal(0, 0.05)
        
        return self.pressure

class BangBangController:
    def __init__(self):
        self.target = 0.0
        self.deadband = 1.0 # PSI tolerance
        
    def update(self, current_pressure):
        error = self.target - current_pressure
        
        inlet = 0
        exhaust = 0
        
        if error > self.deadband:
            # Need more air
            inlet = 1
            exhaust = 0
        elif error < -self.deadband:
            # Too much air
            inlet = 0
            exhaust = 1
        else:
            # Inside deadband (Hold)
            # Both closed seals the chamber
            inlet = 0
            exhaust = 0
            
        return inlet, exhaust

def main():
    chamber = PneumaticChamber()
    controller = BangBangController()
    
    # Target Profile
    targets = [10, 20, 30, 20, 10, 0]
    duration_per_target = 2000 # steps (2s)
    
    history = []
    target_history = []
    times = []
    
    t = 0
    for target in targets:
        controller.target = target
        for _ in range(duration_per_target):
            # Read Sensor
            p = chamber.pressure
            
            # Control
            u_in, u_out = controller.update(p)
            
            # Actuate
            chamber.inlet_state = u_in
            chamber.exhaust_state = u_out
            
            # Physics Step
            chamber.step()
            
            history.append(chamber.pressure)
            target_history.append(target)
            times.append(t * chamber.dt)
            t += 1
            
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(times, target_history, 'r--', label='Target PSI')
    plt.plot(times, history, 'b-', label='Measured PSI')
    plt.title("Bang-Bang Pressure Control")
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (PSI)")
    plt.grid()
    plt.savefig("output/step_response.png")
    print("Sim Complete.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Taming the Oscillations"

### 1. Lab Objectives
- **Run:** Sim. Observe "Chatter" (Rapid switching) inside the deadband if noise is high.
- **Modify:** Set `deadband = 0.1`.
- **Observe:** Valves switch frantically at steady state. This destroys hardware.
- **Modify:** Add a `min_switch_time` (Hysteresis on time). Do not switch valves faster than 50ms.
- **Result:** Smoother control, less wear.

---

## 🚀 Project: "Soft Gripper"

**Goal:** Pick up an egg.
1.  **Hardware:** Soft robotic fingers (PneuNets).
2.  **Logic:**
    *   Inflate to 10 PSI.
    *   Check curvature (Bend sensor or Camera).
    *   If "Contact" detected (Pressure spikes or Motion stops), Switch to "Force Control".
    *   Maintain Pressure.
3.  **Challenge:** Don't crush the egg.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Leaky Integrator"
*   **Scenario:** Robot sags over time.
*   **Cause:** Micro-leaks in tubing.
*   **Control Fix:** Controller must periodically "burp" air in (Pulse inlet) to maintain pressure against leaks.

#### 2. "Explosion"
*   **Scenario:** Target >> Source Pressure.
*   **Cause:** Safety valve missing.
*   **Fix:** Software Watchdog. If Pressure > Max_Safe, Force Exhaust Open.

---

## ⚡ Optimization: PWM Control

Bang-Bang is crude.
*   **PWM:** Switch valve at 50Hz. Vary Duty Cycle.
*   **Result:** Acts like an analog valve. Smooth flow.
*   Allows PID control implementation rather than thresholding.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Difference between Closed-Loop Pressure and Closed-Loop Position?
    *   **A:** Pressure loop ensures safety/force. Position loop ensures shape. Hysteresis makes Position loop hard without external sensors (Cameras).
2.  **Q:** What is "Choked Flow"?
    *   **A:** When air velocity reaches speed of sound (Sonic). Flow rate maxes out regardless of pressure drop.
3.  **Q:** Why use 2 valves instead of 1 3-way valve?
    *   **A:** 2 valves (2-way) allow a "Hold" state (Closed/Closed). A 3-way valve is either Inflating or Exhausting (No Hold).

### Challenge Task
> **Task:** Hysteresis Compensation.
> 1. Implement a lookup table `Pressure(Angle)`.
> 2. Record Inflation Curve vs Deflation Curve.
> 3. Use the correct curve based on `target_angle > current_angle`.

---

## 📚 Further Reading
- **Soft Robotics Toolkit:** Open source hardware designs.
- **Harvard Biodesign Lab:** Soft actuator mechanics.

---

**Day 128 Complete**
