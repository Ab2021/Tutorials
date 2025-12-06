# Day 192: Bio-Inspired Robotics
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 28: Future Technologies

---

> **📝 Content Creator Instructions:**
> Nature had 4 billion years of R&D. Steal from it.
> - **Focus:** Biomimicry, Central Pattern Generators (CPGs), Tensegrity, and Evolutionary Robotics.
> - **Code:** `cpg_gait.py`. A network of coupled oscillators (Hopf oscillators) generating swimming/walking gaits.
> - **Concept:** "Morphological Computation" (The body computes).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Implement** a Central Pattern Generator (CPG) using Coupled differential equations.
2.  **Explain** the concept of Morphological Computation (e.g., Passive Dynamic Walker).
3.  **Simulate** switching between gaits (Walk to Swim) by changing a single parameter.
4.  **Discuss** Swarm Intelligence (Ant Colony Optimization).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib scipy
```

### Prior Knowledge
- Differential Equations ($ \dot{x} = f(x) $).
- Oscillation (Sine waves).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Central Pattern Generators (CPGs)

Animals don't think "Move Muscle A, then B".
Spinal circuits generate rhythmic patterns automatically.
*   **Decouples Control from Actuation:** Brain says "Go Fast". Spine says "Left, Right, Left, Right" at 5Hz.
*   **Robustness:** If he brain dies, the chicken still runs (literally).

### 🔹 Part 2: Morphological Computation

The body performs computation.
*   **Example:** A soft gripper conforms to a cup. You didn't compute the shape; the material did.
*   **Passive Dynamic Walker:** A robot with no motors that walks down a slope using gravity and pendulum dynamics.

### 🔹 Part 3: Tensegrity

Structural principle based on isolated components in compression inside a net of continuous tension.
*   **NASA SuperBall:** A landing robot that is a ball of sticks and cables. Can crash land and survive.

---

## 💻 Implementation: The Salamander Spine

We model a chain of oscillators.
Hopf Oscillator:
$$ \dot{x} = \alpha (\mu - r^2) x - \omega y $$
$$ \dot{y} = \alpha (\mu - r^2) y + \omega x $$
Where $r^2 = x^2 + y^2$. Converges to limit cycle of radius $\sqrt{\mu}$.

### 🛠️ Project Structure
```text
day192_bio/
├── src/
│   ├── cpg_network.py
│   └── visualize_gait.py
└── output/
    └── spinal_wave.png
```

### 👨‍💻 CPG Network (`src/cpg_network.py`)

A chain of N oscillators, coupled to neighbors.

```python
import numpy as np
from scipy.integrate import odeint

class CPGNetwork:
    def __init__(self, n_segments=10):
        self.N = n_segments
        self.mu = 1.0 # Amplitude sq
        self.omega = 2.0 * np.pi # Freq (1Hz)
        self.alpha = 5.0 # Conv Rate
        
        # Coupling weights
        self.k = 1.0 
        
        # Phase lag between segments
        # Walk: 0 lag (S-shape standing wave)
        # Swim: 2pi/N lag (Traveling wave)
        self.phi_lag = np.pi / self.N 
        
    def dynamics(self, state, t):
        # State: [x1..xN, y1..yN]
        x = state[:self.N]
        y = state[self.N:]
        
        dxdt = np.zeros(self.N)
        dydt = np.zeros(self.N)
        
        for i in range(self.N):
            r2 = x[i]**2 + y[i]**2
            
            # Intrinsic Dynamics (Hopf)
            dx = self.alpha * (self.mu - r2) * x[i] - self.omega * y[i]
            dy = self.alpha * (self.mu - r2) * y[i] + self.omega * x[i]
            
            # Coupling (Diffusive)
            # Add influence from i-1 and i+1
            coupling_x = 0
            coupling_y = 0
            
            if i > 0: # Coupon from Left
                # Rotate neighbor by -phi
                x_prev_rot = x[i-1]*np.cos(-self.phi_lag) - y[i-1]*np.sin(-self.phi_lag)
                y_prev_rot = x[i-1]*np.sin(-self.phi_lag) + y[i-1]*np.cos(-self.phi_lag)
                coupling_x += self.k * (x_prev_rot - x[i])
                coupling_y += self.k * (y_prev_rot - y[i])
                
            if i < self.N - 1: # Couple from Right
                x_next_rot = x[i+1]*np.cos(self.phi_lag) - y[i+1]*np.sin(self.phi_lag)
                y_next_rot = x[i+1]*np.sin(self.phi_lag) + y[i+1]*np.cos(self.phi_lag)
                coupling_x += self.k * (x_next_rot - x[i])
                coupling_y += self.k * (y_next_rot - y[i])
                
            dxdt[i] = dx + coupling_x
            dydt[i] = dy + coupling_y
            
        return np.concatenate([dxdt, dydt])

def main():
    net = CPGNetwork(n_segments=8)
    
    # State: 16 vars
    x0 = np.random.rand(16) * 0.1
    t = np.linspace(0, 10, 1000)
    
    sol = odeint(net.dynamics, x0, t)
    
    # Extract X (used as joint angle command)
    X_hist = sol[:, :8]
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    
    # Plot waterfall
    for i in range(8):
        plt.plot(t, X_hist[:, i] + i*2, label=f'Seg {i}')
        
    plt.title("CPG Output (Traveling Wave)")
    plt.ylabel("Joint Angle (Shifted)")
    plt.xlabel("Time (s)")
    plt.savefig('output/spinal_wave.png')
    print("Simulated. This traveling wave drives a Snake Robot forward.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Gait Switch"

### 1. Lab Objectives
- **Run:** `cpg_network.py` with `phi_lag = pi/N`. Observe Traveling Wave (Snake/Swim).
- **Modify:** Change `phi_lag = 0` halfway through simulation.
- **Observe:** The wave collapses into a Standing Wave (All segments in sync).
- **Bio-Link:** This mimics a Salamander moving from Water (Swim) to Land (Walk/Trot). The brain sends a single signal (Drive) that saturates the coupling strength.

---

## 🚀 Project: "Evolutionary Antenna"

**Goal:** Design an optimized structure.
1.  **Sim:** 3D lattice of masses and springs.
2.  **Genome:** Which springs exist? Stiffness values?
3.  **Fitness:** Max height, Min vibration.
4.  **Algo:** Genetic Algorithm (GA). Mutate, Crossover, Select.
5.  **Result:** NASA-style evolved structure (organic looking).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Sync Failure"
*   **Cause:** Coupling strength $k$ too low. Oscillators drift apart.
*   **Fix:** Increase $k$.

#### 2. "Amplitude Explosion"
*   **Cause:** Unstable ODE integration.
*   **Fix:** Use `odeint` (LSODA) or RK4. Don't use Euler for coupled oscillators unless $dt$ is tiny.

---

## ⚡ Optimization: Neuromorphic CPG

Implement the CPG on a SNN (Day 190).
*   Oscillators are made of Excitatory/Inhibitory neuron pairs.
*   Extremely low power locomotion controller.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is a "Limit Cycle"?
    *   **A:** A closed trajectory in phase space. The system periodic behavior is stable (returns to the path if disturbed).
2.  **Q:** Benefit of Bio-Inspired over Classical?
    *   **A:** Bio solutions are often more robust to unstructured environments and require less precise sensing (Intelligence in the mechanics).

### Challenge Task
> **Task:** "Tensegrity Bot".
> 1. Use MuJoCo/PyBullet.
> 2. Build an Icosahedron tensegrity (6 struts, 24 cables).
> 3. Actuate cables to make it "roll" by shifting CoM.

---

## 📚 Further Reading
- **Ijspeert:** "Central pattern generators for locomotion control in animals and robots".
- **Karl Sims:** "Evolved Virtual Creatures" (1994).

---

**Day 192 Complete**
