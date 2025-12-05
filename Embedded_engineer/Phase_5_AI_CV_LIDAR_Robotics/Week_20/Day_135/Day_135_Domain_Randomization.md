# Day 135: Domain Randomization (Sim2Real)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 20: Sim-to-Real & Hardware Acceleration

---

> **📝 Content Creator Instructions:**
> Don't make the simulation perfect. Make it chaotic.
> - **Focus:** The "Reality Gap", Overfitting to Sim physics, Uniform Randomization of Mass/Friction/Damping, Visual Randomization (Textures/Lights), and Automatic Domain Randomization (ADR).
> - **Code:** A Python script `robustness_check.py` that evaluates a control policy across a spectrum of randomized physical parameters to visualize the "Pass/Fail" regions.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** the Reality Gap (Why a purely sim-trained robot fails in real life).
2.  **Implement** Domain Randomization: Training on a distribution of environments $E \sim P(\xi)$ rather than a single instance.
3.  **Contrast** "System Identification" (Measuring friction exactly) vs "Domain Randomization" (Robustness to friction uncertainty).
4.  **Visualize** the Policy Success Rate as a function of environmental noise.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Reinforcement Learning (Overfitting).
- Physics Simulation.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Reality Gap

Your simulator implies perfect friction ($\mu=1.0$), zero delay, and perfect sensors.
Reality has:
*   Grease on the floor ($\mu=0.8$).
*   Network Lag (20ms jitter).
*   Sensor noise.
*   Battery Voltage droop.
If the policy relies on $\mu=1.000$, it fails at $\mu=0.99$.

### 🔹 Part 2: The Solution - Chaos

Train the agent on an ensemble of physics:
*   Mass: $[0.9kg, 1.1kg]$.
*   Friction: $[0.5, 1.2]$.
*   Motor Strength: $[0.8, 1.2]$.
*   **Result:** The agent learns a conservative strategy that works *everywhere*.

### 🔹 Part 3: Visual Randomization

For Vision-based RL:
*   Randomize floor textures.
*   Randomize lighting color/position.
*   Randomize camera FOV.
*   Real world looks like just "another random variation" to the Neural Net.

---

## 💻 Implementation: Robustness Evaluation

We simulate a simple inverted pendulum controller and check how it handles mass errors.

### 🛠️ Project Structure
```text
day135_sim2real/
├── src/
│   ├── robustness_check.py
└── output/
    ├── domain_heatmap.png
```

### 👨‍💻 Robustness Analyzer (`src/robustness_check.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class PendulumEnv:
    def __init__(self, mass=1.0, length=1.0, friction=0.1):
        self.m = mass
        self.l = length
        self.b = friction # Damping
        self.g = 9.81
        
        self.theta = np.pi # Down
        self.dtheta = 0.0
        self.dt = 0.01

    def step(self, torque):
        # Physics: Torque = I * alpha
        # I = m * l^2
        # Gravity Torque = -m * g * l * sin(theta)
        # Damping = -b * dtheta
        
        inertia = self.m * self.l**2
        grav_torque = -self.m * self.g * self.l * np.sin(self.theta)
        damping_torque = -self.b * self.dtheta
        
        alpha = (torque + grav_torque + damping_torque) / inertia
        
        self.dtheta += alpha * self.dt
        self.theta += self.dtheta * self.dt
        
        return self.theta

class PIDController:
    '''
    Tuned for specific Mass=1.0, Length=1.0
    '''
    def __init__(self):
        self.kp = 50.0
        self.kd = 10.0
        self.target = 0.0 # Upright
        
    def act(self, theta, dtheta):
        # Angle wrapping (-pi to pi)
        err = self.target - theta
        while err > np.pi: err -= 2*np.pi
        while err < -np.pi: err += 2*np.pi
        
        output = self.kp * err - self.kd * dtheta
        return output

def evaluate(mass, length):
    env = PendulumEnv(mass=mass, length=length)
    env.theta = 0.1 # Small perturbation start
    controller = PIDController()
    
    # Run for 5 seconds
    total_error = 0
    stable = True
    
    for _ in range(500):
        torque = controller.act(env.theta, env.dtheta)
        theta = env.step(torque)
        
        total_error += abs(theta)
        if abs(theta) > np.pi/2:
            stable = False # Fell over
            break
            
    return stable

def main():
    # Grid Search over Domain Parameters
    masses = np.linspace(0.5, 5.0, 50) # 0.5kg to 5kg
    lengths = np.linspace(0.5, 3.0, 50) # 0.5m to 3m
    
    results = np.zeros((len(masses), len(lengths)))
    
    print("Evaluating Domain Robustness...")
    
    for i, m in enumerate(masses):
        for j, l in enumerate(lengths):
            is_stable = evaluate(m, l)
            results[i, j] = 1.0 if is_stable else 0.0
            
    # Plot
    plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(lengths, masses)
    plt.contourf(X, Y, results, cmap='RdYlGn')
    plt.colorbar(label='Stability (1=Stable, 0=Fail)')
    plt.xlabel('Rod Length (m)')
    plt.ylabel('Rod Mass (kg)')
    plt.title('Controller Safety Margin (Nominal: M=1, L=1)')
    
    # Mark Nominal Point
    plt.plot(1.0, 1.0, 'k*', markersize=15, label='Nominal Model')
    plt.legend()
    plt.savefig("output/domain_heatmap.png")
    print("Map Generated.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Unlucky Robot"

### 1. Lab Objectives
- **Run:** The script.
- **Observe:** The PID controller (tuned for 1kg, 1m) works for masses up to ~2kg, then fails.
- **Hypothesis:** Why? Because $K_p$ is too weak to fight gravity for heavier mass ($m g l \sin \theta$).
- **Action:** Retune PID for "Mean Mass = 2.5kg".
- **Result:** The stable region shifts.
- **Challenge:** Can you find Gains that work for the *entire* range [0.5, 5.0]? (Robust Control). Often impossible with linear PID. RL might do better.

---

## 🚀 Project: "Visual Domain Randomizer"

**Goal:** Train a MNIST classifier that works on "Camouflaged" digits.
1.  **Data:** Invert colors, add static noise, rotate digits.
2.  **Train:** CNN.
3.  **Test:** Real handwritten digits on crumpled paper.
4.  **Result:** The randomized training makes the feature filters focus on shape, not color/background.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Sim works, Real fails"
*   **Cause:** Unaccounted Dynamic. E.g., Gear backlash, Motor Deadband, or Elasticity in links.
*   **Fix:** Add these effects to Sim (even crudely).

#### 2. "Impossible Randomization"
*   **Cause:** If you randomize gravity to be $[-50, 50]$, no policy can exist that satisfies all cases.
*   **Fix:** Keep randomization physically plausible.

---

## ⚡ Optimization: ADR (Automatic Domain Randomization)

Instead of setting ranges manually ($[0.5, 1.5]$), let the algorithm grow the range.
1.  Start with range $[0.99, 1.01]$.
2.  If Success > 90%: Expand range to $[0.95, 1.05]$.
3.  Repeat.
*   **OpenAI Dactyl:** Used this to learn to solve Rubik's cube with a shadow hand.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Privileged Information"?
    *   **A:** Data the Sim handles (Exact friction, Exact mass) but the Robot can't see. In RL, we can feed this to the Critic, but NOT the Actor (Asymmetric Actor-Critic).
2.  **Q:** Why Randomize Visuals?
    *   **A:** Synthetic images are too clean. Real cameras have gain noise, motion blur, and lens distortion.
3.  **Q:** Does DR guarantee safety?
    *   **A:** No. It improves probability. You still need Safety Filters (Barrier Functions) on real hardware.

### Challenge Task
> **Task:** Latency Injection.
> 1. Add a buffer to the Env step.
> 2. `obs = self.buffer[-delay]`.
> 3. Randomize delay between 0 and 5 steps.
> 4. See if PID oscillates.

---

## 📚 Further Reading
- **OpenAI:** "Solving Rubik's Cube with a Robot Hand" (ADR Paper).
- **Tobin et al.:** "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World".

---

**Day 135 Complete**
