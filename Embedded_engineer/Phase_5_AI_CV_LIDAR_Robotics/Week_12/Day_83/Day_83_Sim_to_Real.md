# Day 83: Sim-to-Real RL (Domain Randomization)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 12: Robot Learning

---

> **📝 Content Creator Instructions:**
> Training on one reality is training on overfitting.
> - **Focus:** The Reality Gap, Domain Randomization (DR) of Dynamics and Visuals, and System Identification.
> - **Code:** A Gym Wrapper that randomizes mass, friction, and visual textures every reset.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Identify** Sources of the Reality Gap (Dynamics, Latency, Noise).
2.  **Implement** Dynamics Randomization: Varying Mass, Friction, Damping.
3.  **Implement** Visual Randomization: Varying Light, Textures, Camera Position.
4.  **Evaluate** Robustness: Testing the policy on "Extreme" unseen parameters.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Sim-based training).

### Software Environment
```bash
pip install gymnasium numpy pybullet
```

### Prior Knowledge
- Reinforcement Learning PPO (Day 26).
- Physics Engines (Day 58).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Reality Gap

You train a robot to walk in PyBullet. It walks perfectly.
You deploy it on a real robot. It falls instantly.
*   **Reasons:**
    *   **Friction:** Real floor is slippery/sticky. Sim is usually $\mu=1.0$.
    *   **Latency:** Real motors have delay (10-20ms). Sim is instantaneous.
    *   **Mass Distribution:** Real robot cables shift Center of Mass.
    *   **Actuator Dynamics:** Real motors saturate and have backlash.

### 🔹 Part 2: Domain Randomization (DR)

Hypothesis: If the policy can handle *every* physics configuration in a wide range, it can handle the Real World (which falls inside that range).
*   $Mass \sim U[0.8m, 1.2m]$
*   $Friction \sim U[0.5, 1.5]$
*   $Delay \sim U[0ms, 40ms]$
*   The agent learns to accept "Sensor Noise" and "Sluggish Motors" as normal states.

### 🔹 Part 3: Visual DR

For Vision-based RL:
*   Randomize Textures (Walls, Floor).
*   Randomize Lighting (Spotlights, Ambient).
*   Randomize Camera Pose (Jitter XYZ).
*   The agent learns to ignore colors/textures and focus on *Shapes* and *Depth*.

---

## 💻 Implementation: DR Wrapper

We wrap a standard Gym Environment to add randomization on `reset()`.

### 🛠️ Project Structure
```text
day83_sim2real/
├── src/
│   ├── dr_wrapper.py
│   └── train_robust.py
└── textures/
    └── random_pattern.jpg
```

### 👨‍💻 Input Randomizer (`src/dr_wrapper.py`)

Using PyBullet (access to low-level physics).

```python
import gymnasium as gym
import numpy as np
import pybullet as p

class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.mass_range = [0.8, 1.2]
        self.friction_range = [0.5, 1.0]
        self.color_range = [0.0, 1.0]

    def reset(self, **kwargs):
        # 1. Randomize Dynamics
        robot_id = self.env.robot_id # Assuming PyBullet Env exposes this
        
        # Mass
        base_mass = 1.0 # Nominal
        new_mass = base_mass * np.random.uniform(*self.mass_range)
        p.changeDynamics(robot_id, -1, mass=new_mass)
        
        # Friction
        for joint in range(p.getNumJoints(robot_id)):
            fric = np.random.uniform(*self.friction_range)
            p.changeDynamics(robot_id, joint, lateralFriction=fric)
            
        # 2. Randomize Visuals (Color)
        color = np.random.uniform(*self.color_range, size=3)
        p.changeVisualShape(robot_id, -1, rgbaColor=[*color, 1])
        
        # 3. Add Latency (Simulated via Buffer)
        self.act_buffer = [] 
        
        return self.env.reset(**kwargs)

    def step(self, action):
        # Latency Simulation: Delay action by 1 step
        # (Simplified: In real PPO, we feed delayed observation)
        
        # Add Action Noise
        action += np.random.normal(0, 0.05, size=action.shape)
        
        obs, reward, done, trunc, info = self.env.step(action)
        
        # Add Observation Noise
        obs += np.random.normal(0, 0.01, size=obs.shape)
        
        return obs, reward, done, trunc, info
```

### 👨‍💻 Robust Training (`src/train_robust.py`)

Standard PPO training.
*   **Key:** Training takes longer!
*   Why? The problem is harder. The agent cannot "cheat" by memorizing a specific friction value. It must learn a robust gait.

---

## 🔬 Lab Exercise: "The Slippery Slope"

### 1. Lab Objectives
- **Env:** Inverted Pendulum (CartPole).
- **Control:** Train PPO *without* DR on `gravity=-9.8`.
- **Test:** Run on `gravity=-20.0` (Jupiter).
    - Result: Fails (cannot compensate).
- **Train:** Train PPO *with* DR on `gravity ~ U[-5, -25]`.
- **Test:** Run on `gravity=-20.0`.
    - Result: Succeeds. The agent learned to "React to falling speed" rather than "Memorize timing".

---

## 🚀 Project: "Shadow Hand Transfer"

**Goal:** Rubik's Cube manipulation (OpenAI style).
1.  **Task:** Rotate cube.
2.  **Randomization:**
    *   Cube Size: $\pm 5\%$.
    *   Cube Mass: $\pm 50\%$.
    *   Surface Friction: $0.1$ (Ice) to $1.5$ (Rubber).
    *   Gravitational Vector: Tilt the world by $5^{\circ}$.
3.  **Observation:** The policy develops a "Tight Grasp". It squeezes harder to compensate for potential slip. This behavior *emerges* from the randomization.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Policy Diverges"
*   **Cause:** Randomization range is too wide. Impossible tasks (e.g., Mass=100kg, Motor=1Nm).
*   **Fix:** **Curriculum Learning**. Start with small range $\pm 1\%$. Gradually expand to $\pm 20\%$.

#### 2. "Conservative Behavior"
*   **Symptom:** Robot moves very slowly.
*   **Cause:** Slow implies safety. If friction is unknown, moving fast is risky.
*   **Fix:** Add a "Velocity Reward" to force speed, or reduce the lower bound of friction randomization if it's unrealistic.

---

## ⚡ Optimization: Automatic Domain Randomization (ADR)

Tuning ranges $[min, max]$ is tedious.
*   **ADR:** The *algorithm* automatically expands the ranges.
*   If Agent succeeds on current range, Expand it.
*   If Agent fails, Shrink it.
*   Results in a curriculum that pushes the agent to the limit of solvability.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Visual Domain Randomization"?
    *   **A:** Rendering the simulation with crazy colors (Purple floor, Green sky) so the network ignores color and learns edge/depth features.
2.  **Q:** Why add Observation Noise?
    *   **A:** Real sensors have noise ($O_{real} = O_{true} + \epsilon$). If Sim is perfect ($O_{sim} = O_{true}$), the policy overfits to precision.
3.  **Q:** Is DR enough for everything?
    *   **A:** No. Complex contact dynamics (soft bodies, fluids) are hard to randomize if the sim physics engine cannot model them at all.

### Challenge Task
> **Task:** Latency Handling.
> 1. Wrap the env to buffer the last 5 observations.
> 2. Stack them: `Current State = [O_t, O_{t-1}, ... O_{t-5}]`.
> 3. This gives the LSTM/MLP context about *derivatives* (Acceleration) which helps estimate Mass/Friction online.

---

## 📚 Further Reading
- **OpenAI:** "Solving Rubik's Cube with a Robot Hand" (2019).
- **Domain Randomization Paper (Tobin et al.):** "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World".

---

**Day 83 Complete**
