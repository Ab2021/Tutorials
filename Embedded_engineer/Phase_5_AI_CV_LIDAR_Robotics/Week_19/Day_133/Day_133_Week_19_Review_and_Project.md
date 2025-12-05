# Day 133: Week 19 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 19: Soft Robotics & Bio-Inspired Control

---

> **📝 Content Creator Instructions:**
> Merge the Soft Body (Physics) with the Bio-Brain (CPG/SNN).
> - **Goal:** Create a "Soft Ray" (Manta Ray) capable of swimming.
> - **Code:** A unified simulation where a CPG drives Soft Continuum fins, tuned by a Genetic Algorithm.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Integrate** Continuum Kinematics, CPGs, and Evolutionary methods into a single system.
2.  **Translate** Biological observation (Fin undulation) into Control primitives.
3.  **Evaluate** the robustness of Soft/Bio systems compared to Rigid/Servo systems.

---

## 📚 Week 19 Review: The "Squishy" Side of Robotics

| Day | Topic | Key Lesson | Tool |
|-----|-------|------------|------|
| **127** | **Soft Modeling** | Constant Curvature Assumption simplify infinite DOF | `ContinuumSeg` |
| **128** | **Pneumatics** | Air is compressible; Hysteresis requires logic | `BangBang` |
| **129** | **RL for Soft** | Model-Free Learning handles contact/deformation | `StableBaselines3` |
| **130** | **Bio-Gait (CPG)** | Coupled Oscillators generate stable rhythms | `Hopf` |
| **131** | **Neuromorphic** | Spikes (LIF) encode time/events efficiently | `SNN` |
| **132** | **Evolution** | GAs optimize parameters without gradients | `PyGAD` |

### The "Bio-Bot" Architecture
```mermaid
graph TD
    Gene[Genome (GA)] -->|Parameters| CPG
    CPG[CPG (Spinal Cord)] -->|Phase Signal| Soft[Soft Body (Physics)]
    Sensors -->|Spikes| SNN[Reflex Layer]
    SNN -->|Inhibition| CPG
```

---

## 🚀 Weekly Capstone: "The Soft Ray"

**Scenario:** Design an autonomous underwater soft glider.
**Components:**
1.  **Body:** Central rigid hull (Battery/Jetson).
2.  **Wings:** Two large PneuNet flaps (Left/Right).
3.  **Controller:** CPG producing a sinusoidal wave traveling down the wing (Ripple Gait).
4.  **Optimizer:** GA finds the best Wave Amplitude & Frequency.

### 🛠️ Project Structure
```text
week19_capstone/
├── src/
│   ├── ray_sim.py (Combined Physics + CPG)
│   ├── evolve_ray.py (Optimization Loop)
```

### 👨‍💻 Tying it all together (`src/ray_sim.py`)

A simplified specific simulator for the Ray.

```python
import numpy as np
import matplotlib.pyplot as plt

class RayBot:
    def __init__(self, amplitude, frequency, wave_lag):
        # Genome Parameters
        self.amp = amplitude
        self.freq = frequency
        self.lag = wave_lag # Phase lag between wing segments
        
        # State
        self.pos_x = 0.0
        self.velocity = 0.0
        
        # Wing Segments (3 per wing)
        self.phases = [0, self.lag, 2*self.lag] 
        self.time = 0
        self.dt = 0.05

    def step(self):
        # 1. CPG Update
        # Generate Flapping Angle for each segment
        # theta = A * sin(omega * t - phase)
        t = self.time
        flaps = []
        for p in self.phases:
            flaps.append(self.amp * np.sin(self.freq * t - p))
            
        # 2. Physics Proxy (Propulsion)
        # Thrust is proportional to the rearward velocity of the traveling wave
        # F_thrust ~ Sum( (dTheta/dt)^2 * drag_coeff ) * Direction_Factor
        
        # Simplified:
        # If wave travels Backwards, Robot moves Forward.
        # Wave Speed V_wave = Frequency / Lag?
        
        # Let's derive a heuristic thrust model:
        # Thrust correlates with Amplitude^2 * Frequency^2 (Power)
        # But only if Lag creates a proper wave shape.
        # If Lag = 0, wings just flap up/down (Drag based propulsion, inefficient)
        # If Lag > 0, Undulation pushes water back.
        
        undulation_factor = np.sin(self.lag) # Max at pi/2?
        thrust = (self.amp * self.freq)**2 * undulation_factor
        
        # Drag on body
        drag = 0.5 * 1.0 * self.velocity**2
        
        # Acceleration
        accel = thrust - drag # Mass=1
        
        self.velocity += accel * self.dt
        self.pos_x += self.velocity * self.dt
        
        self.time += self.dt
        return self.pos_x

def evaluate_genome(genome):
    # Genome: [Amp, Freq, Lag]
    bot = RayBot(genome[0], genome[1], genome[2])
    
    # Run for 10 seconds
    for _ in range(200):
        bot.step()
        
    return bot.pos_x # Distance traveled

def main():
    # Manual Test
    # amp=1.0, freq=3.0, lag=0.5
    dist = evaluate_genome([1.0, 3.0, 0.5])
    print(f"Manual Test Distance: {dist:.2f} m")
    
    # Evolution Loop (Simple Random Search for Capstone Demo)
    best_dist = -1
    best_gene = None
    
    print("Evolving...")
    for i in range(100):
        # Random Gene
        g = [
            np.random.uniform(0.1, 2.0), # Amp
            np.random.uniform(1.0, 10.0), # Freq
            np.random.uniform(0.0, 3.14) # Lag
        ]
        
        d = evaluate_genome(g)
        if d > best_dist:
            best_dist = d
            best_gene = g
            print(f"Gen {i}: New Best {d:.2f} m with {g}")
            
    print("Optimization Complete.")

if __name__ == "__main__":
    main()
```

---

## 📝 Self-Assessment Quiz

1.  **Architecture:**
    *   What layer handles the rhythm?
    *   **A:** The CPG (Oscillator). The High Level (Brain) just says "Swim", CPG says "Left-Right-Left".
2.  **Softness:**
    *   Why use soft wings?
    *   **A:** Efficiency. Smooth deformation sheds vortices better than rigid joints. Safety for marine life.
3.  **Evolution:**
    *   Why might a GA select huge Amplitude?
    *   **A:** Because our physics model neglected material stress. In reality, too much amplitude bursts the pneumatic chambers. *The Simulator Gap*.

---

## ⏭️ Look Ahead: Week 20
Back to the Metal.
**Week 20: Sim-to-Real & Hardware Acceleration.**
*   How to make Python code run fast (CUDA/FPGA).
*   Bridging the Gap from Isaac Gym to Real Robots.
*   Cloud Robotics.

---

**Week 19 Complete**
