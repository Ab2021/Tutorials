# Day 193: Soft Actuators & Materials
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 28: Future Technologies

---

> **📝 Content Creator Instructions:**
> Steel feels old. Let's build with Muscle.
> - **Focus:** Dielectric Elastomer Actuators (DEAs), Shape Memory Alloys (SMAs), Fluidic Elastomers, and 4D Printing.
> - **Code:** `sma_control.py`. Simulating the thermal dynamics and hysteresis loop of a Nitinol wire actuator.
> - **Concept:** Smart Materials and Phase Transitions.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Thermal (SMA), Electric (DEA), and Fluidic actuators.
2.  **Model** the Hysteresis loop of an SMA (Martensite $\leftrightarrow$ Austenite).
3.  **Implement** a PWM-based Resistance Heating controller.
4.  **Discuss** the Self-Healing capabilities of modern soft polymers.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Nitinol (SMA) wire (Flexino).
- MOSFET/Transistor driver.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Thermodynamics (Heat Transfer).
- PWM Control.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Shape Memory Alloys (SMAs)

Nitinol (Nickel-Titanium). Memory Metal.
*   **Cold State (Martensite):** Crystal structure is sheared. Deforms easily (plastic).
*   **Hot State (Austenite):** Crystal structure becomes cubic (rigid). Returns to original shape with massive force.
*   **The Problem:** Cooling takes time. Bandwidth is low (< 1Hz).

### 🔹 Part 2: Dielectric Elastomer Actuators (DEAs)

"Artificial Muscle".
*   Capacitor made of rubber.
*   Apply High Voltage (5kV). Plates attract. Rubber squishes and expands sideways.
*   **Pros:** Fast (>100Hz), large strain.
*   **Cons:** High Voltage!

### 🔹 Part 3: 4D Printing

3D Printing with materials that change shape *after* printing (Time is the 4th dimension).
*   e.g., Hydrogels that swell in water to fold into a box.

---

## 💻 Implementation: SMA Controller

We model the heating (Joule) and cooling (Convection) of a wire.
State: Temperature $T$, Phase $\xi$ (0=Martensite, 1=Austenite).

### 🛠️ Project Structure
```text
day193_materials/
├── src/
│   ├── sma_model.py
│   └── pwm_controller.py
└── output/
    └── hysteresis_loop.png
```

### 👨‍💻 Thermal Model (`src/sma_model.py`)

$\dot{T} = \frac{1}{mc} (I^2 R - h A (T - T_{amb}))$

```python
import numpy as np
import matplotlib.pyplot as plt

class SMAModel:
    def __init__(self):
        # Wire Params (Nitinol Flexinol 0.1mm)
        self.mass = 0.0001 # kg (small)
        self.c = 837.0 # Specific Heat J/kgK
        self.R = 5.0 # Ohms
        self.h = 50.0 # Convection coeff
        self.Area = 0.001 # Surface Area
        
        # Phase Transition Temps
        self.As = 70.0 # Austenite Start
        self.Af = 90.0 # Austenite Finish
        self.Ms = 60.0 # Martensite Start
        self.Mf = 40.0 # Martensite Finish
        
        # State
        self.T = 25.0 # Ambient
        self.xi = 0.0 # Phase (0=Martensite, 1=Austenite)
        self.strain = 0.0
        
    def step(self, current_I, dt=0.01):
        # 1. Thermal Dynamics
        # Power In = I^2 R
        P_in = (current_I**2) * self.R
        
        # Power Out = h A (T - Tamb)
        P_out = self.h * self.Area * (self.T - 25.0)
        
        dT = (P_in - P_out) / (self.mass * self.c)
        self.T += dT * dt
        
        # 2. Phase Dynamics (Simplified Sigmoid Kinematics)
        # Hysteresis: Path depends on Heating vs Cooling
        if dT > 0: # Heating
            if self.T > self.As:
                target_xi = (self.T - self.As) / (self.Af - self.As)
                target_xi = np.clip(target_xi, 0.0, 1.0)
                # Simple relaxation
                self.xi += (target_xi - self.xi) * 0.5 
        else: # Cooling
            if self.T < self.Ms:
                target_xi = (self.T - self.Mf) / (self.Ms - self.Mf) # Note: Reverse logic
                # Actually, easier to model Xi_martensite
                # Let's use simple logic:
                if self.T < self.Mf: self.xi = 0.0
                elif self.T < self.Ms: self.xi = (self.T - self.Mf)/(self.Ms - self.Mf)
                
        # 3. Output Strain (Recovered Shape)
        # Max recover variance = 4%
        self.strain = self.xi * 0.04 
        
        return self.T, self.strain

def main():
    sma = SMAModel()
    
    t_hist, T_hist, s_hist = [], [], []
    
    # Pulse Current
    for t in range(1000): # 10 sec
        I = 0.0
        if 200 < t < 600:
            I = 0.5 # 500mA
            
        T, s = sma.step(I)
        
        t_hist.append(t*0.01)
        T_hist.append(T)
        s_hist.append(s)
        
    # Plot
    fig, ax1 = plt.subplots()
    
    ax1.plot(t_hist, T_hist, 'r-')
    ax1.set_ylabel('Temp (C)', color='r')
    ax1.set_xlabel('Time (s)')
    
    ax2 = ax1.twinx()
    ax2.plot(t_hist, s_hist, 'b-')
    ax2.set_ylabel('Strain (%)', color='b')
    
    plt.title("SMA Thermal Response")
    plt.savefig('output/sma_response.png')
    print("Done.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Slow Down"

### 1. Lab Objectives
- **Run:** `sma_model.py`.
- **Observe:** Heating is fast (active), Cooling is slow (passive).
- **Challenge:** Improve bandwidth.
- **Solution:** "Forced Convection". Simulate a Fan ($h$ increases 5x). Observe the cooling curve steepen.
- **Alternative:** "Antagonistic Pair". As Wire A cools, Wire B heats/pulls.

---

## 🚀 Project: "Soft Spider"

**Goal:** 8-legged crawler.
1.  **Structure:** 3D printed flexible PLA.
2.  **Actuation:** 8 SMAs (one per leg curl).
3.  **Circuit:** Arduino + MOSFET array.
4.  **Gait:** Ripple gait (Day 192). Needs careful timing to account for cooling lag.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Overheating"
*   **Cause:** $T > 300C$. Nitinol loses memory (Annealing).
*   **Fix:** **Resistance Feedback.** SMA resistance changes by ~10% during phase change. Measure $R$, estimate $T$, stop heating.

#### 2. "Drift"
*   **Cause:** "Fatigue". After 1000 cycles, strain reduces.
*   **Fix:** Don't push to 100% strain limit. Stay within safe bounds (2-3%).

---

## ⚡ Optimization: High Voltage DEAs

How to generate 5kV cleanly?
*   **Pico-HV:** Tiny DC-DC converters (0.5g) for insect robots.
*   **Optimization:** Stack layers (Multilayer DEA) to reduce voltage requirement ($V \propto \text{thickness}$).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is SMA efficiency low (<5%)?
    *   **A:** It's a heat engine. Carnot limit. You convert Elec $\to$ Heat $\to$ Work. Most heat is lost to air.
2.  **Q:** Difference between "Soft" and "Compliant"?
    *   **A:** Soft = Material modulus (Rubber). Compliant = Structure behaves softly (Springs). Series Elastic Actuators (SEA) are Compliant but Rigid.

### Challenge Task
> **Task:** "Hysteresis Compensation".
> 1. Invert the Preisach Model of hysteresis.
> 2. Feedforward control: If I want Strain X, and history was Y, apply Current Z.

---

## 📚 Further Reading
- **Soft Robotics Journal:** (SoRo).
- **Harvard Microrobotics Lab:** RoboBee (Piezo actuators).

---

**Day 193 Complete**
