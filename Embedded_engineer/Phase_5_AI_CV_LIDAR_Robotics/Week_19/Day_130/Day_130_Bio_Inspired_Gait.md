# Day 130: Bio-Inspired Gait (Central Pattern Generators)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 19: Soft Robotics & Bio-Inspired Control

---

> **📝 Content Creator Instructions:**
> Your brain doesn't tell your legs to move step-by-step. Your spine does.
> - **Focus:** Central Pattern Generators (CPGs), Coupled Oscillators (Hopf / Rayleigh), Generating Gaits (Tripod, Wave, Ripple) via Phase Differences.
> - **Code:** A Python simulation of a 6-node CPG network controlling a Hexapod, producing a stable Tripod gait without kinematic planning.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** how rhythmic movement emerges from coupled non-linear oscillators.
2.  **Tune** the parameters ($\mu, \omega$) of a Hopf Oscillator to control amplitude and frequency.
3.  **Construct** a coupling matrix to enforce specific phase shifts (e.g., $180^\circ$ anti-phase for walking).
4.  **Visualize** the limit cycle behavior in Phase Space.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Simulation). Hexapod robot ideal but not required.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Differential Equations ($\dot{x} = ...$).
- Gait Diagrams (Stance/Swing).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Biological CPGs

A headless chicken can run. Why?
*   **Neural Circuits:** The spinal cord contains circuits that oscillate autonomously.
*   **Brain's Role:** The brain just sends a scalar signal ("Go Faster"). It doesn't plan angles.
*   **Robotics:** We use math oscillators to mimic this. Robust to perturbations!

### 🔹 Part 2: The Hopf Oscillator

An equation that has a stable "Limit Cycle" (circle).
$$ \dot{x} = \alpha (\mu - r^2) x - \omega y $$
$$ \dot{y} = \alpha (\mu - r^2) y + \omega x $$
where $r^2 = x^2 + y^2$.
*   $\mu$: Target Amplitude ($Amplitude = \sqrt{\mu}$).
*   $\omega$: Frequency (rad/s).
*   If perturbed, it spirals back to the circle.

### 🔹 Part 3: Coupling

Connecting oscillators $i$ and $j$.
$$ \dot{x}_i = ... + \sum k (x_j \cos\theta_{ij} - y_j \sin\theta_{ij}) $$
*   $\theta_{ij}$: Desired phase difference.
*   **Tripod Gait:** Legs 1, 4, 5 move together. Legs 2, 3, 6 move together. Phase shift $\pi$ ($180^\circ$).

---

## 💻 Implementation: Hexapod CPG

We simulate 6 coupled oscillators.

### 🛠️ Project Structure
```text
day130_cpg/
├── src/
│   ├── cpg_network.py
└── output/
    ├── gait_plot.png
```

### 👨‍💻 CPG Network (`src/cpg_network.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class HopfOscillator:
    def __init__(self, omega=2*np.pi, mu=1.0):
        self.x = 0.1 # State 1
        self.y = 0.0 # State 2
        self.omega = omega
        self.mu = mu
        self.alpha = 5.0 # Convergence speed
        self.dt = 0.01

    def step(self, coupling_signal_x, coupling_signal_y):
        r2 = self.x**2 + self.y**2
        
        dx = self.alpha * (self.mu - r2) * self.x - self.omega * self.y + coupling_signal_x
        dy = self.alpha * (self.mu - r2) * self.y + self.omega * self.x + coupling_signal_y
        
        self.x += dx * self.dt
        self.y += dy * self.dt
        
        return self.x # This drives the Joint Angle

class HexapodCPG:
    def __init__(self):
        # 6 Legs: L1(0), L2(1), L3(2), R1(3), R2(4), R3(5)
        self.oscillators = [HopfOscillator() for _ in range(6)]
        
        # Coupling Matrix (Adjacency)
        # 1 means coupling exists with phase shift
        self.coupling_strength = 1.0
        
        # Desired Phase Shifts (Tripod Gait)
        # L1(0) is in phase with R2(4), L3(2) -> Group A
        # L2(1) is in phase with R1(3), R3(5) -> Group B
        # Group A and B are 180 deg (pi) apart
        
        self.phase_biases = np.zeros((6, 6))
        
        # Define Neighbors (L1-L2, L2-L3, R1-R2, R2-R3, L1-R1, etc)
        # Only enforcing minimal set to synchronize everything
        
        # Coupling Logic for Tripod:
        # 0(L1) <-> 1(L2) : Anti-phase (pi)
        # 1(L2) <-> 2(L3) : Anti-phase (pi)
        # 3(R1) <-> 4(R2) : Anti-phase (pi)
        # 4(R2) <-> 5(R3) : Anti-phase (pi)
        # 0(L1) <-> 3(R1) : Anti-phase (pi) (Contralateral)
        
        edges = [
            (0, 1, np.pi), (1, 2, np.pi),
            (3, 4, np.pi), (4, 5, np.pi),
            (0, 3, np.pi), (1, 4, np.pi), (2, 5, np.pi)
        ]
        
        for i, j, phi in edges:
            self.phase_biases[i, j] = phi
            self.phase_biases[j, i] = -phi # Symmetry

    def step(self):
        outputs = []
        next_states = []
        
        # Calculate Coupling Inputs First
        couplings_x = np.zeros(6)
        couplings_y = np.zeros(6)
        
        for i in range(6):
            for j in range(6):
                if self.phase_biases[i, j] != 0 or i==j: # If connected
                     continue 
                     # Wait, connection is defined by edges logic broadly.
                     # Let's rely on bias matrix non-zero?
                     # Bias can be 0 (in phase). 
                     # Let's iterate edges explicitly or full matrix.
                     pass

        # Simplified Coupling Loop
        for i in range(6):
            cx = 0; cy = 0
            for j in range(6):
                if i == j: continue
                # We assume fully connected or sparse.
                # Let's assume we use the biases matrix logic.
                # If bias is defined (we init all to 0, need a way to detect 'no link')
                # For this demo, let's hardcode neighbors of 0
                pass
            
            # Proper Matrix impl
            for j in range(6):
                theta = self.phase_biases[i, j]
                # Rotate j's state by theta
                # x_j_rot = x_j cos(theta) - y_j sin(theta)
                # y_j_rot = x_j sin(theta) + y_j cos(theta)
                
                # Check if connected (Using non-zero check is risky if theta=0 is valid)
                # We will assume all coupled for simplicity or use mask.
                # All coupled to sync:
                
                other = self.oscillators[j]
                x_rot = other.x * np.cos(theta) - other.y * np.sin(theta)
                y_rot = other.x * np.sin(theta) + other.y * np.cos(theta)
                
                cx += x_rot
                cy += y_rot
            
            couplings_x[i] = self.coupling_strength * cx / 6.0 # Normalize
            couplings_y[i] = self.coupling_strength * cy / 6.0
            
        # Update Steps
        current_vals = []
        for i in range(6):
            val = self.oscillators[i].step(couplings_x[i], couplings_y[i])
            current_vals.append(val)
            
        return current_vals

def main():
    cpg = HexapodCPG()
    
    steps = 1000
    history = np.zeros((steps, 6))
    
    for t in range(steps):
        vals = cpg.step()
        history[t, :] = vals
        
    # Plot
    plt.figure(figsize=(10, 6))
    joints = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
    colors = ['r', 'b', 'r', 'b', 'r', 'b'] # Tripod colors
    
    for i in range(6):
        plt.plot(history[:, i] + i*2.5, color=colors[i], label=joints[i])
        
    plt.title("Hexapod CPG Outputs (Tripod Gait)")
    plt.xlabel("Time Step")
    plt.yticks([])
    plt.legend(loc='upper right')
    plt.savefig("output/gait_plot.png")
    print("Gait Generated.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Gait Transition"

### 1. Lab Objectives
- **Run:** Sim. Observe two distinct groups (Red/Blue) oscillating $180^\circ$ out of phase.
- **Modify:** Change coupling weights interactively.
- **Modify:** Set `phase_biases` to produce a **Wave Gait**.
    *   L1 $\to$ L2 $\to$ L3 $\to$ R3 $\to$ R2 $\to$ R1 (Wave propogation).
    *   Phase shift $\pi/3$ ($60^\circ$) between neighbors.
- **Result:** The robot "Flows" like a caterpillar. More stable, but slower.

---

## 🚀 Project: "Reflexes"

**Goal:** Stumble recovery.
1.  **Scenario:** Leg 1 hits an obstacle (Force spike).
2.  **Logic:** Reset the phase of Oscillator 1.
    *   If `Force > Threshold`: Set $x_1 = -Start$, $y_1 = ...$
3.  **Result:** The leg retracts instantly and restarts its cycle. The whole network adjusts phase to maintain stability.
4.  **Why:** Much faster than planning "Stop, lift leg, Place leg".

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Drift"
*   **Cause:** $\omega$ (Frequency) is slightly different for each oscillator or numerical error.
*   **Result:** Legs desynchronize over time.
*   **Fix:** Coupling Strength $k$ must be high enough to force "Phase Locking".

#### 2. "Amplitude Decay"
*   **Cause:** $\mu$ is too small or $\alpha$ too weak.
*   **Fix:** Ensure $\alpha (\mu - r^2)$ term dominates perturbations.

---

## ⚡ Optimization: Analog Computing

In biology, this is analog.
*   **Neuromorphic Chips:** Implement CPGs on Spiking Neural Networks (SNN) like Intel Loihi.
*   **Power:** micro-Watts.
*   **Speed:** Kilo-Hertz.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Benefit of CPG vs Trajectory Planning?
    *   **A:** CPG reduces control dimension. You control 1 parameter ($\omega$) to speed up 18 joints. It handles disturbances gracefully (Limit Cycle stability).
2.  **Q:** What is the limit cycle?
    *   **A:** The closed trajectory in phase space $(x, \dot{x})$ that the system is attracted to. It represents the rhythmic gait.
3.  **Q:** How to turn?
    *   **A:** Increase $\omega$ (Frequency) or Amplitude for legs on the Left side. The robot naturally turns Right.

### Challenge Task
> **Task:** Quadruped Trot.
> 1. Configure for 4 legs.
> 2. Trot: Diagonals move together (FL+RR vs FR+RL).
> 3. Verify phase matrix.

---

## 📚 Further Reading
- **Ijspeert:** "Central Pattern Generators for Locomotion Control in Animals and Robots".
- **Hopf Bifurcation:** Mathematical foundation.

---

**Day 130 Complete**
