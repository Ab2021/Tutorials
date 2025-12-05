# Day 26: Learning-Based Control (RL)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 4: Advanced Control & Dynamics

---

> **📝 Content Creator Instructions:**
> We used RL for Planning (high-level). Now we use it for Control (motor torques).
> - **Focus:** Continuous Control RL (DDPG, TD3, SAC) and Sim-to-Real for Quadrupeds.
> - **Code:** Training a MuJoCo Hopper/Walker using SAC.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Contrast** Policy Interaction methods (PPO) vs. Q-Learning methods (DQN, DDPG) for continuous control.
2.  **Explain** the Twin Delayed DDPG (TD3) algorithm and why it stabilizes training.
3.  **Implement** Soft Actor-Critic (SAC) to learn a robust locomotion gait.
4.  **Apply** Domain Randomization (Friction, Mass, Delay) to enable Sim-to-Real transfer.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU (Recommended).

### Software Environment
```bash
pip install gymnasium[mujoco] stable-baselines3 shimmy
```

### Prior Knowledge
- Reinforcement Learning Basics (Day 17).
- Neural Networks.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Continuous Action Spaces

In Day 17 (Grid Nav), action was discrete (Up/Down).
In Motor Control, action is continuous voltage $u \in [-12V, 12V]$.
*   **Problem:** DQN cannot output continuous values easily (requires $\arg \max_a Q(s,a)$).
*   **Solution:** Actor-Critic methods.
    *   **Actor ($\pi(s)$):** Outputs the action directly.
    *   **Critic ($Q(s,a)$):** Evaluates how good the action was.

### 🔹 Part 2: TD3 (Twin Delayed DDPG)

Deep Deterministic Policy Gradient (DDPG) is unstable (Q-value overestimation).
**TD3 Fixes:**
1.  **Clipped Double Q-Learning:** Use two Critics ($Q_1, Q_2$). Take $\min(Q_1, Q_2)$ to reduce overestimation.
2.  **Delayed Policy Updates:** Update Actor less frequently than Critic.
3.  **Target Policy Smoothing:** Add noise to the target action to make Q-function smoother.

### 🔹 Part 3: SAC (Soft Actor-Critic)

Currently the state-of-the-art for robotics.
**Max Entropy Framework:**
$$ J(\pi) = \sum E [R(s_t, a_t) + \alpha H(\pi(\cdot|s_t))] $$
*   Maximize Reward + Maximize Entropy (Randomness).
*   **Benefit:** The robot builds a diverse set of solutions. If one limb gets stuck, it has "backup plans" because it learned to solve the task in multiple ways.
*   **Robustness:** Highly robust to external disturbances.

---

## 💻 Implementation: Learning to Hop

We will use the standard `Hopper-v4` environment in Gym (MuJoCo physics).
**Goal:** Make a 1-legged robot hop forward as fast as possible.

### 🛠️ Project Structure
```text
day26_rl_control/
├── train_sac.py
├── evaluate.py
└── domain_randomization.py
```

### 👨‍💻 Code Implementation (`train_sac.py`)

```python
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import os

# 1. Create Env
env_name = "Hopper-v4"
env = gym.make(env_name, render_mode=None)

# 2. Define Controller (SAC)
# Policy: MlpPolicy (Dense NNs)
# Learning Rate: 3e-4 (Standard for Adam)
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    batch_size=256,
    buffer_size=1000000, # Replay Buffer
    ent_coef='auto',     # Auto-tune entropy alpha
    tensorboard_log="./sac_hopper_tensorboard/"
)

# 3. Callback for periodic evaluation
eval_env = gym.make(env_name)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./logs/',
    log_path='./logs/',
    eval_freq=5000,
    deterministic=True,
    render=False
)

# 4. Train
print("Starting Training (This takes 30-60 mins on GPU)...")
model.learn(total_timesteps=100000, callback=eval_callback)

model.save("sac_hopper_final")
```

### 👨‍💻 Code Implementation (`evaluate.py`)

```python
import gymnasium as gym
from stable_baselines3 import SAC

model = SAC.load("logs/best_model")
env = gym.make("Hopper-v4", render_mode="human")

obs, _ = env.reset()
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    
    if terminated or truncated:
        obs, _ = env.reset()
```

---

## 🔬 Lab Exercise: Sim-to-Real Sim

We cannot run RL on hardware directly (Robot will break during exploration).
We train in Sim, deploy to Real. But Sim $\ne$ Real.

### 1. Lab Objectives
- Implement **Domain Randomization** wrapper.
- Randomize: Link Mass ($ \pm 20\%$), Friction ($ \pm 50\%$), Motor Strength.
- Train SAC on this "Randomized" env.
- **Test:** Run the policy on a "Nominal" environment (representing Reality).
- **Observe:** The randomized policy is slightly lower performance peak, but **much** more stable when you mess with the physics parameters.

### 2. Wrapper Code (`domain_randomization.py`)

```python
import gymnasium as gym
import numpy as np

class RandomPhysicsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        # Access MuJoCo model
        model = self.env.unwrapped.model
        
        # Randomize Mass of Torso (Body 1)
        original_mass = model.body_mass[1]
        random_scale = np.random.uniform(0.8, 1.2)
        model.body_mass[1] = original_mass * random_scale
        
        # Randomize Friction
        model.geom_friction[:] = model.geom_friction[:] * np.random.uniform(0.5, 1.5)
        
        return self.env.reset(**kwargs)
```

---

## 🚀 Project: "Quadruped Locomotion"

**Goal:** Train a Spot-Mini clone (Ant-v4 or custom URDF) to walk.
1.  **Reward Shaping:**
    *   $r = v_x$ (Forward velocity).
    *   $r -= 0.001 ||u||^2$ (Energy penalty).
    *   $r -= 0.5 |y|$ (Lateral drift penalty).
    *   $r -= 0.5 |\omega_z|$ (Spin penalty).
2.  **Training:** Run SAC for 1M steps.
3.  **Result:** The robot discovers a "Trotting" gait automatically!
4.  **Challenge:** Add "Terrain" (Steps, Slopes) to the simulation.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Robot falls immediately"
*   **Cause:** Early exploration is random noise. 
*   **Fix:** Normal. Training takes time. Ensure `reset()` puts the robot in a valid upright state. If it starts lying down, it can't learn to walk.

#### 2. "Q-Value Divergence"
*   **Symptom:** Critic Loss goes to infinity.
*   **Cause:** Learning rate too high or Reward scaling too large (Reward=1000 per step).
*   **Fix:** Normalize rewards to range $[-1, 1]$ or decrease LR.

---

## ⚡ Optimization: Asymmetric Actor-Critic

In Sim, we know everything (Privileged Information). In Real, we only know sensors.
*   **Training:**
    *   **Critic:** Sees EVERYTHING (Ground friction, Ground height map, exact velocity).
    *   **Actor:** Sees ONLY SENSORS (IMU, Proprioception).
*   **Deployment:** We throw away the Critic. The Actor runs purely on sensor data.
*   *Result:* Better training because Critic guides Actor with ground truth.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why use Off-Policy (SAC) instead of On-Policy (PPO) for Robotics?
    *   **A:** Sample Efficiency. Robots are slow (Sim or Real). SAC reuses old data (Replay Buffer), learning much faster per real-world second than PPO.
2.  **Q:** What is "Exploration Noise"?
    *   **A:** Adding random jitter to actions during training to discover new behaviors.
3.  **Q:** Why Domain Randomization?
    *   **A:** It turns "System Identification Error" into "Noise". The NN learns to be robust to noise, thus robust to model mismatch.

### Challenge Task
> **Task:** Recovery Policy.
> 1. Push the robot over (Apply large force).
> 2. Train a policy specifically to "Get Up" from the ground.
> 3. Switch: If $|roll| > 45^\circ$, run Recovery Policy. Else, run Walking Policy.

---

## 📚 Further Reading
- **SAC Paper:** Haarnoja et al. (2018).
- **Learning to Walk:** "Minitaur" Paper (Google Brain).
- **RMA (Rapid Motor Adaptation):** Kumar et al. (CMU).

---

**Day 26 Complete**
