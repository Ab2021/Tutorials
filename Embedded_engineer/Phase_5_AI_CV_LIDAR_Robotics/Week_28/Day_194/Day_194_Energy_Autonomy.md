# Day 194: Energy Harvesting & Autonomy
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 28: Future Technologies

---

> **📝 Content Creator Instructions:**
> Don't just work hard, work smart (and sustainably).
> - **Focus:** Solar, Kinetic, and Thermal Energy Harvesting. Power Management, MPPT, and Long-Duration Autonomy.
> - **Code:** `energy_sim.py`. Simulation of a Solar-Powered Robot across a 24h cycle (Sunlight intensity, Battery Charging/Discharging).
> - **Concept:** The Energy Budget.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Calculate** the Energy Budget for a mobile robot.
2.  **Implement** a basic MPPT (Maximum Power Point Tracking) algorithm simulation.
3.  **Simulate** a "Sun Chasing" behavior to maximize charging.
4.  **Discuss** alternative sources (Microbial Fuel Cells, Radioisotope).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Solar Panel (Small 5V).
- LiPo Battery + Charger module.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Electronics (P = IV).
- Batteries (C-rating, Capacity).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Tyranny of the Plug

Most robots last 2-4 hours. To exist in the wild, they need:
1.  **Big Battery:** Heavy.
2.  **Harvesting:** Free energy from environment.
3.  **Sleep:** Reduce consumption to micro-watts.

### 🔹 Part 2: Solar Harvesting

*   **Irradiance:** ~1000 W/m² (Peak Sun).
*   **Efficiency:** ~20%.
*   **Yield:** A $1m^2$ panel gives ~200W.
    *   Quadruped Robot needs ~200W walking. (Break even).
    *   Glider/Buoy needs ~5W. (Net positive).

### 🔹 Part 3: MPPT (Maximum Power Point Tracking)

Solar panels have a non-linear IV curve.
*   If you draw too much current, Voltage drops to 0. (P=0).
*   If you draw too little, Current is 0. (P=0).
*   **MPPT:** An algorithm that adjusts the DC-DC converter duty cycle to keep V*I at max.

---

## 💻 Implementation: The 24-Hour Cycle

We simulate a robot on Mars. It must survive the night.
*   **Input:** Solar Profile (Sinusoid + Cloud Noise).
*   **Output:** Battery SoC (State of Charge).
*   **Logic:** If SoC < 20%, Hibernate. If > 90%, Science Mode (High Power).

### 🛠️ Project Structure
```text
day194_energy/
├── src/
│   ├── energy_sim.py
│   └── mppt_algo.py
└── output/
    └── energy_plot.png
```

### 👨‍💻 MPPT Logic (`src/mppt_algo.py`)

"Perturb and Observe" algorithm.

```python
class MPPT:
    def __init__(self):
        self.voltage = 0.0
        self.power_prev = 0.0
        self.voltage_prev = 0.0
        self.duty_cycle = 0.5 # 50%
        
    def update(self, v_panel, i_panel):
        power = v_panel * i_panel
        
        dv = v_panel - self.voltage_prev
        dp = power - self.power_prev
        
        step = 0.01
        
        if dp != 0:
            if dp > 0:
                if dv > 0:
                    self.duty_cycle += step
                else:
                    self.duty_cycle -= step
            else:
                if dv > 0:
                    self.duty_cycle -= step
                else:
                    self.duty_cycle += step
                    
        self.duty_cycle = max(0.0, min(1.0, self.duty_cycle))
        self.power_prev = power
        self.voltage_prev = v_panel
        
        return self.duty_cycle
```

### 👨‍💻 Energy Sim (`src/energy_sim.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class RobotEnergy:
    def __init__(self):
        self.battery_capacity = 100.0 # Wh
        self.soc = 50.0 # Wh
        self.panel_size = 0.1 # m^2
        self.panel_eff = 0.2 # 20%
        
        self.base_load = 2.0 # W (Standby)
        self.active_load = 20.0 # W (Walking)
        
        self.state = "SLEEP"
        
    def get_solar_input(self, hour):
        # Sun rises 6am, sets 6pm. Peak at 12.
        if 6.0 <= hour <= 18.0:
            intensity = 1000.0 * np.sin((hour - 6.0) * np.pi / 12.0)
            # Add clouds
            if np.random.random() < 0.1: intensity *= 0.5
            return intensity * self.panel_size * self.panel_eff
        return 0.0

    def step(self, hour, dt_hours):
        # 1. Input
        p_in = self.get_solar_input(hour)
        
        # 2. Logic (Power Manager)
        p_load = self.base_load
        
        if self.soc > 90.0:
            self.state = "ACTIVE"
        elif self.soc < 30.0:
            self.state = "SLEEP"
            
        if self.state == "ACTIVE":
            p_load = self.active_load
            
        # 3. Integrate
        # Energy = Power * Time
        energy_net = (p_in - p_load) * dt_hours
        
        self.soc += energy_net
        
        # Clamp
        if self.soc > self.battery_capacity:
            self.soc = self.battery_capacity # Wasted excess
        if self.soc < 0:
            self.soc = 0 # Dead
            self.state = "DEAD"
            
        return p_in, p_load, self.soc

def main():
    bot = RobotEnergy()
    
    hours = np.linspace(0, 48, 480) # 2 days
    dt = hours[1] - hours[0]
    
    soc_hist = []
    pin_hist = []
    
    for h in hours:
        day_hour = h % 24
        pi, pl, soc = bot.step(day_hour, dt)
        
        soc_hist.append(soc)
        pin_hist.append(pi)
        
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(hours, pin_hist, 'y', label='Solar (W)')
    plt.fill_between(hours, pin_hist, color='yellow', alpha=0.3)
    plt.ylabel('Input Watts')
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.plot(hours, soc_hist, 'g', label='Battery (Wh)')
    plt.axhline(100, color='k', linestyle='--')
    plt.axhline(0, color='r', linestyle='--')
    plt.ylabel('SoC')
    plt.xlabel('Time (Hours)')
    plt.legend()
    
    plt.savefig('output/energy_profile.png')
    print("Simulated 48 hours.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Sunseeker"

### 1. Lab Objectives
- **Scenario:** Robot has 4 light sensors (N, S, E, W).
- **Control:** `cmd_vel.angular.z = k * (Light_Left - Light_Right)`.
- **Result:** Phototaxis (Robot turns to face the light).
- **Challenge:** Combine with Obstacle Avoidance. Don't run into a wall just because it's sunny.

---

## 🚀 Project: "The Eternal Glider"

**Goal:** Ocean monitoring.
1.  **Platform:** Wave Glider (Liquid Robotics) style.
2.  **Mechanic:** Float on surface. Submerged wings use wave motion (Up/Down) to generate Forward Thrust.
3.  **Solar:** Panels on top for Comms/GPS.
4.  **Autonomy:** Infinite.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Vampire Drain"
*   **Cause:** Voltage regulators (LDOs) burn power even when sleeping.
*   **Fix:** Use Buck Converters with low quiescent current. Physically cut power to non-essential sensors (MOSFET switches).

#### 2. "Cold Batteries"
*   **Cause:** LiPo capacity drops 50% at -10C.
*   **Fix:** Use waste heat from CPU to warm the battery compartment (Insulation).

---

## ⚡ Optimization: Event-Triggered Comms

Radio (Wifi/4G) is the biggest power hog (after motors).
*   **Dont:** Send telemetry at 10Hz.
*   **Do:** Send telemetry only when "Something Interesting happened" or once per hour.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Harvesting"?
    *   **A:** Scavenging ambient energy (Vibration, Light, Heat) vs "Storing" (Battery).
2.  **Q:** Why MPPT?
    *   **A:** To impedance match the source (Solar) to the load (Battery). Without it, you lose 30-50% efficiency.

### Challenge Task
> **Task:** "Kinetic Watch".
> 1. Model a piezoelectric harvester on a walking robot leg.
> 2. Each step impact generates pulse.
> 3. Estimate: Can this run the CPU? (Usually No, only uWatts).

---

## 📚 Further Reading
- **NASA Spirit/Opportunity:** Power management case studies.
- **EcoBot:** Microbial Fuel Cell robots (eat flies).

---

**Day 194 Complete**
