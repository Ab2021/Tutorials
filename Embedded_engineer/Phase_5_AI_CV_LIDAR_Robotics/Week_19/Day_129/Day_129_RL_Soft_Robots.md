# Day 129: Learning-Based Control (RL for Continuum)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 19: Soft Robotics & Bio-Inspired Control

---

> **📝 Content Creator Instructions:**
> If you can't model it, learn it.
> - **Focus:** Reinforcement Learning (RL) for Soft Robotics, Sim-to-Real gaps, OpenAI Gym (Gymnasium) customization, and Soft Actor-Critic (SAC) intuition.
> - **Code:** A custom Gym environment `SoftRobotEnv` wrapping the Day 127 Kinematics, and a training loop using `stable-baselines3` to teach the robot to reach a red dot.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Justify** RL for Soft Robotics (Analytical models like CCA fail with external loads/contacts).
2.  **Define** the MDP (Markov Decision Process): State (Tip Pos), Action (Pressures), Reward ($-Distance$).
3.  **Implement** a custom Gymnasium Environment compatible with Stable-Baselines3.
4.  **Train** a PPO agent to control the continuum arm.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU recommended for training (but simple 2D arm works on CPU).

### Software Environment
```bash
pip install gymnasium stable-baselines3 shimmy
```

### Prior Knowledge
- Reinforcement Learning concepts (Agent, Environment, Reward).
- Day 127 (Kinematics).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Analytical limit

Analytical models (Jacobians) are great until the robot touches a wall.
*   **Soft Robot Contact:** Deformation is complex.
*   **Hysteresis:** History dependent.
*   **Solution:** Model-Free RL. The agent learns the physics by interacting with it.

### 🔹 Part 2: The RL Loop

1.  **Observation ($O_t$):** Current Tip Position $(x,y,z)$ + Target Position $(x_d, y_d, z_d)$.
2.  **Action ($A_t$):** Change in Pressures $(\Delta P_1, \Delta P_2, \Delta P_3)$.
3.  **Reward ($R_t$):**
    *   $R = - ||X_{tip} - X_{target}||^2$ (Dense Reward).
    *   $R = +100$ if Hit Target (Sparse Reward).
    *   $R = -0.1$ per step (Time penalty).

### 🔹 Part 3: Sim-to-Real

Training on real robots is slow (and they break).
We train in Sim.
*   **Domain Randomization:** Randomize friction, mass, stiffness in Sim so the Policy becomes robust enough to handle the Real World.

---

## 💻 Implementation: Training the Arm

We wrap our kinematic model in a Gym Env.

### 🛠️ Project Structure
```text
day129_rl_soft/
├── src/
│   ├── soft_env.py
│   ├── train.py
└── output/
    ├── learning_curve.png
```

### 👨‍💻 Gymnasium Environment (`src/soft_env.py`)

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
# Import simplified model physics (from Day 127 concept)

class SoftRobotEnv(gym.Env):
    def __init__(self):
        super(SoftRobotEnv, self).__init__()
        
        # Action: Change in curvature [dK1, dPhi1, dK2, dPhi2]
        # Range: -0.1 to +0.1 per step
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(4,), dtype=np.float32)
        
        # Observation: [TipX, TipY, TipZ, TargetX, TargetY, TargetZ, K1, P1, K2, P2]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        # Robot State [K1, P1, K2, P2]
        self.q = np.array([0.001, 0.0, 0.001, 0.0]) # Start straight
        
        self.target = np.array([0.2, 0.2, 0.3]) 
        self.max_steps = 100
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.q = np.array([0.001, 0.0, 0.001, 0.0]) 
        
        # Random Target
        self.target = np.random.uniform(-0.3, 0.3, size=3)
        self.target[2] = np.random.uniform(0.1, 0.4) # Z
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. Apply Action
        self.q += action
        
        # Clip ranges (Physical limits)
        self.q[0] = np.clip(self.q[0], 0, 10) # Kappa > 0
        self.q[2] = np.clip(self.q[2], 0, 10)
        self.q[1] = np.clip(self.q[1], -np.pi, np.pi) # Phi
        self.q[3] = np.clip(self.q[3], -np.pi, np.pi)

        # 2. Forward Kinematics (Simplified 2-segment calc)
        # (Copy logic from Day 127)
        tip_pos = self._fk(self.q)
        
        # 3. Calculate Reward
        dist = np.linalg.norm(tip_pos - self.target)
        reward = -dist # Minimize distance
        
        # 4. Check Done
        terminated = False
        if dist < 0.02:
            reward += 10.0 # Bonus
            terminated = True
            
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
        self.current_step += 1
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        tip = self._fk(self.q)
        return np.concatenate([tip, self.target, self.q]).astype(np.float32)

    def _fk(self, q):
        # Dummy FK for brevity (replace with Day 127 logic)
        # Simulating a simple bending arm
        k1, p1, k2, p2 = q
        # Approx tip pos just to make RL learn *something*
        # (This math is fake, use real FK in production!)
        x = (np.cos(p1)/k1 if k1>0.1 else 0) + (np.cos(p2)/k2 if k2>0.1 else 0)
        y = (np.sin(p1)/k1 if k1>0.1 else 0) + (np.sin(p2)/k2 if k2>0.1 else 0)
        z = 0.4 - (0.01*k1 + 0.01*k2) 
        # Adding noise to position to keep bounds sane
        return np.array([np.clip(x, -0.4, 0.4), np.clip(y, -0.4, 0.4), np.clip(z, 0, 0.4)])
```

### 👨‍💻 Training Script (`src/train.py`)

Using PPO (Proximal Policy Optimization).

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from soft_env import SoftRobotEnv
import matplotlib.pyplot as plt

def main():
    # 1. Create Env
    env = SoftRobotEnv()
    
    # Check compliance
    check_env(env)
    
    # 2. Instantiate Agent
    model = PPO("MlpPolicy", env, verbose=1)
    
    # 3. Train
    print("Training...")
    model.learn(total_timesteps=10000)
    
    # 4. Save
    model.save("soft_robot_ppo")
    
    # 5. Test
    obs, _ = env.reset()
    print(f"Target: {obs[3:6]}")
    for i in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, _ = env.step(action)
        tip_pos = obs[:3]
        print(f"Step {i}: Action={action}, Tip={tip_pos}, Reward={reward:.3f}")
        if done:
            print("Target Reached!")
            break

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Learning to Curl"

### 1. Lab Objectives
- **Run:** `train.py`.
- **Observe:** Initially, the robot flails randomly (Exploration). Rewards are large negative ($-0.5$).
- **Wait:** After 5000 steps, rewards increase ($-0.1$).
- **Result:** The agent learns that increasing $\kappa$ (Curvature) brings the tip closer to targets that are not on the Z-axis.
- **Compare:** Try `SAC` instead of `PPO`. SAC is generally more sample efficient for robotics.

---

## 🚀 Project: "Obstacle Avoidance RL"

**Goal:** Reach Target. Avoid Red Sphere.
1.  **State:** Add Obstacle Position `(ObsX, ObsY, ObsZ)` to observation.
2.  **Reward:**
    *   $R = -Distance_{target}$.
    *   If $Distance_{obstacle} < 0.1$: $R -= 100$. (Collision Penalty).
3.  **Result:** The arm learns to loop *around* the obstacle to get to the target. (Emergent behavior).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Agent stands still"
*   **Cause:** Action magnitude too small or Rewards too sparse.
*   **Fix:** Ensure `action_space` allows significant movement. Check if Random Walk covers the workspace.

#### 2. "Exploding Gradients"
*   **Cause:** FK singularity (division by zero curvature).
*   **Fix:** Clip observations and calculation results.

---

## ⚡ Optimization: Curriculum Learning

Start easy, get harder.
1.  **Stage 1:** Target is always [0, 0, 0.4] (Straight up). Learn to stabilize.
2.  **Stage 2:** Target is within 5cm.
3.  **Stage 3:** Full Workspace.
*   Speeds up training significantly.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the "Credit Assignment Problem"?
    *   **A:** Determining *which* action caused the reward. Did I hit the target because I increased pressure at step 5 or step 95?
2.  **Q:** Why not just use ID (Inverse Dynamics)?
    *   **A:** ID for soft robots requires solving partial differential equations (PDEs) in real-time. RL approximates the inverse map neural network.
3.  **Q:** Observation Space Normalization?
    *   **A:** Critical. Inputs (Position 0.2, Pressure 40PSI) have different scales. Normalize everything to [-1, 1] for Neural Nets.

### Challenge Task
> **Task:** Hysteresis Agent.
> 1. Add `last_action` to Observation.
> 2. The Agent should learn *momentum*.
> 3. Verify it learns to overshoot slightly to compensate for lag.

---

## 📚 Further Reading
- **Stable-Baselines3 Docs:** Standard RL library.
- **Gymnasium:** API reference.

---

**Day 129 Complete**
