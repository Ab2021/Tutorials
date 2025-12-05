# Day 17: Machine Learning for Planning (RL)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 3: Advanced Navigation & Planning

---

> **📝 Content Creator Instructions:**
> Classic planning (A*) requires a map. Humans drive by "feel".
> - **Focus:** Reinforcement Learning (PPO, SAC) and Imitation Learning (BC).
> - **Code:** Training a DRL agent to navigate a maze using Ray RLLib / Stable-Baselines3.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Formulate** Navigation as a Markov Decision Process (MDP).
2.  **Train** a PPO (Proximal Policy Optimization) agent to reach a goal while avoiding obstacles.
3.  **Explain** the Simulation-to-Reality (Sim2Real) gap and Domain Randomization.
4.  **Implement** Behavior Cloning (Imitation Learning) from human expert demonstrations.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU (Training RL is slow on CPU).

### Software Environment
```bash
pip install stable-baselines3 shimmy gymnasium
pip install box2d-py # Physics engine
```

### Prior Knowledge
- Neural Networks (PyTorch).
- Basic Reward functions (+1 good, -1 bad).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Navigation as MDP

Markov Decision Process $(S, A, P, R, \gamma)$:
*   **State ($S$):** Lidar Scans (360 floats) + Goal Vector (2 floats).
*   **Action ($A$):** Linear Velocity $v$, Angular Velocity $\omega$.
*   **Reward ($R$):**
    *   $+10$: Reached Goal.
    *   $-100$: Collision.
    *   $-0.1$: Per time step (encourages speed).
    *   $-0.5$: Jerky motion (encourages smoothness).

### 🔹 Part 2: Reinforcement Learning Algorithms

#### 2.1 PPO (Proximal Policy Optimization)
*   **On-Policy:** Learns from data collected by current policy $\pi_{\theta}$.
*   **Key Idea:** Don't change the policy *too much* in one step (Clip ratio). Prevents catastrophic forgetting.
*   *Pros:* Stable, works well with continuous action spaces (Robotics).

#### 2.2 SAC (Soft Actor-Critic)
*   **Off-Policy:** Learns from Replay Buffer (old data).
*   **Key Idea:** Entropy Regularization. Maximize Reward + Randomness (Exploration).
*   *Pros:* Sample efficient (needs fewer interactions than PPO).

### 🔹 Part 3: Imitation Learning (BC)

Learning from scratch is hard.
**Behavior Cloning (BC):**
*   Collect dataset $D = \{(s_i, a_i)\}$ from Expert (Human Joystick).
*   Train NN (Supervised Learning): Minimize MSE $( \pi_\theta(s) - a_{expert} )^2$.
*   *Issue:* **Distribution Shift**. If robot deviates slightly, it sees a state it never saw in training, makes a mistake, deviates further -> Crash. (DAgger fixes this).

---

## 💻 Implementation: Training PPO Navigation

We will use `stable-baselines3` and a custom Gym Environment.

### 🛠️ Project Structure
```text
day17_rl/
├── envs/
│   └── robot_nav_env.py
├── train_ppo.py
└── eval_policy.py
```

### 👨‍💻 Code Implementation (`envs/robot_nav_env.py`)

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RobotNavEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Action: [v, w] (normalized -1 to 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # State: [Lidar (10 rays), GoalDist, GoalAngle]
        self.observation_space = spaces.Box(low=0, high=10, shape=(12,), dtype=np.float32)
        
        self.state = None
        self.goal = np.array([5.0, 5.0])
        self.pos = np.zeros(2)
        self.angle = 0.0
        
    def reset(self, seed=None):
        self.pos = np.zeros(2)
        self.angle = 0.0
        self.goal = np.random.uniform(-5, 5, size=(2,))
        return self._get_obs(), {}
        
    def step(self, action):
        # 1. Apply Dynamics (Unicycle)
        v = (action[0] + 1) / 2.0 * 1.0 # Map -1..1 to 0..1 m/s
        w = action[1] * 1.0 # -1..1 rad/s
        
        dt = 0.1
        self.pos[0] += v * np.cos(self.angle) * dt
        self.pos[1] += v * np.sin(self.angle) * dt
        self.angle += w * dt
        
        # 2. Compute Reward
        dist = np.linalg.norm(self.pos - self.goal)
        reward = -dist # Dense reward (guide to goal)
        
        terminated = False
        truncated = False
        
        if dist < 0.5:
            reward += 100
            terminated = True
            
        # 3. Get Obs (Simulate Lidar)
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, {}
        
    def _get_obs(self):
        # Fake Lidar (just random for template)
        lidar = np.random.uniform(0, 5, size=(10,))
        
        # Goal Relative
        dist = np.linalg.norm(self.pos - self.goal)
        relative_angle = np.arctan2(self.goal[1]-self.pos[1], self.goal[0]-self.pos[0]) - self.angle
        
        return np.concatenate([lidar, [dist, relative_angle]]).astype(np.float32)
```

### 👨‍💻 Code Implementation (`train_ppo.py`)

```python
from stable_baselines3 import PPO
from envs.robot_nav_env import RobotNavEnv

# 1. Create Env
env = RobotNavEnv()

# 2. Define Model
# MlpPolicy: 2 layers of 64 neurons
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_nav_tensorboard/")

# 3. Train
print("Training PPO...")
try:
    model.learn(total_timesteps=100000)
except KeyboardInterrupt:
    pass

# 4. Save
model.save("ppo_nav_model")
print("Model Saved.")
```

---

## 🔬 Lab Exercise: Obstacle Avoidance

### 1. Lab Objectives
- Modify the `_get_obs` to return *real* raycasts to circular obstacles.
- Add Reward Penalty: `if min(lidar) < 0.2: reward -= 50; terminated = True`.
- Train agent.
- **Visualize:** Watch the robot "learn" to turn away from walls. Initially it will crash repeatedly (Exploration).

### 2. Step-by-Step Guide
1.  Add `self.obstacles = [[2, 2, 0.5]]` (x, y, radius).
2.  Implement `raycast(pos, angle, obstacles)` function.
3.  Run training for 500k steps.

### 3. Expected Output
- **Episode 0:** Random spinning. Result: -50 reward.
- **Episode 1000:** Drives straight, hits wall. Result: -30 reward.
- **Episode 5000:** Turns away from wall, misses goal. Result: -10 reward.
- **Episode 10000:** Navigates around wall to goal. Result: +90 reward.

---

## 🚀 Project: "End-to-End Visual Navigation"

**Goal:** Train a robot to navigate a corridor using *only Camera Images* (Pixels).
**Method:**
1.  **Sim:** Isaac Gym / Gazebo.
2.  **Network:** CNN Encoder (ResNet) -> LSTM (Memory) -> Actor/Critic Heads.
3.  **Task:** "Reach the Red Box".
4.  **Sim2Real:** Apply Drywall textures to simulator walls to match reality.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Reward Hacking"
*   **Symptom:** Robot spins in circles forever.
*   **Cause:** Reward was `r += v` (encourage speed). Robot found that spinning in place creates high angular `v` (or just moving fast in circles) without hitting walls.
*   **Fix:** Ensure optimal behavior maximizes reward. Use `r = Progress_Towards_Goal`.

#### 2. "Sparse Reward Problem"
*   **Symptom:** Robot never finds the goal (+100), so gradients are zero.
*   **Fix:** Curriculum Learning. Start with Goal 1m away. Once solved, move Goal to 2m, etc.

---

## ⚡ Optimization: Vectorized Environments

Training on 1 robot is slow.
**Isaac Gym:** Simulates 4,096 robots in parallel on GPU.
*   PPO collects 4,096 steps per Sim-Step.
*   Training takes minutes instead of days.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Difference between RL and Traditional Planning?
    *   **A:** Planning requires a Model ($Map$, $Dynamics$). RL learns the Policy directly from Experience, often model-free.
2.  **Q:** What is the Credit Assignment Problem?
    *   **A:** If the robot crashes at step 100, was it because of the action at step 99 or step 1? (Usually step 1 set up the failure). RL uses $\gamma$ (discount factor) to handle this.

### Challenge Task
> **Task:** Implement "Curiosity" (Intrinsic Reward).
> 1. Add a module that predicts the next state $s_{t+1}$ given $s_t, a_t$.
> 2. Reward = Prediction Error.
> 3. Robot gets points for seeing "New things" (unpredictable states). It will explore the map without a goal!

---

## 📚 Further Reading
- **PPO Paper:** Schulman et al. (2017).
- **Sim-to-Real:** Domain Randomization (OpenAI).
- **Navigation:** "Learning to Navigate in Complex Environments" (Mirowski et al.).

---

**Day 17 Complete**
