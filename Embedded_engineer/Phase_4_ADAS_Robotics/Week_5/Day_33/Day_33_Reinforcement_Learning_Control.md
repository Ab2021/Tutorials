# Day 33: Reinforcement Learning for Control
## Phase 4: ADAS & Robotics Systems | Week 5: Path Planning & Decision Making

---

> **📝 Day 33 Focus:**
> Classical control (PID/MPC) requires a perfect model of the car and the road. What if the model is unknown or too complex? **Reinforcement Learning (RL)** allows an agent to learn how to drive by trial and error, maximizing a reward signal. Today, we move from "Programming" behavior to "Training" behavior.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the RL tuple $(S, A, R, S')$: State, Action, Reward, Next State.
2.  **Explain** the Exploration vs Exploitation dilemma ($\epsilon$-greedy).
3.  **Implement** Q-Learning (Tabular) for a simple grid-world driving task.
4.  **Understand** Deep Q-Networks (DQN) and how they scale to high-dimensional inputs (Images).
5.  **Train** an RL agent using `stable-baselines3` to solve a continuous control problem.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 18:** Neural Networks (for Deep RL).
-   **Probability:** Markov Decision Processes (MDP).

### Hardware Requirements
-   **GPU:** Recommended for Deep RL training.

### Software Stack
-   **Python Libraries:** `gymnasium` (formerly gym), `stable-baselines3`, `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The RL Loop

1.  **Agent** observes **State** $S_t$ (e.g., Camera image, Speed).
2.  Agent takes **Action** $A_t$ (e.g., Steer Left).
3.  **Environment** transitions to **Next State** $S_{t+1}$.
4.  Environment gives **Reward** $R_{t+1}$ (e.g., +1 for staying in lane, -100 for crash).
5.  Goal: Maximize **Cumulative Reward** $G_t = \sum \gamma^k R_{t+k+1}$.

### 🔹 Part 2: Q-Learning (Value-Based)

We want to learn a function $Q(S, A)$ that predicts the expected future reward of taking action $A$ in state $S$.

**Bellman Equation:**
$$ Q(S, A) \leftarrow Q(S, A) + \alpha [ R + \gamma \max_{a'} Q(S', a') - Q(S, A) ] $$

-   $\alpha$: Learning Rate.
-   $\gamma$: Discount Factor (0.99 means care about future, 0 means greedy).

**Table Lookup:** If $S$ is discrete (Grid), we store $Q$ in a table.

### 🔹 Part 3: Deep Q-Networks (DQN)

If $S$ is an image, the table is too big.
Use a **Neural Network** to approximate $Q(S, A)$.
-   Input: Image.
-   Output: Q-values for each action (Steer Left, Straight, Right).

**Key Tricks:**
1.  **Experience Replay:** Store transitions $(S, A, R, S')$ in a buffer and sample randomly to break correlation.
2.  **Target Network:** Use a separate, slowly updating network to calculate the target Q-values for stability.

---

## 💻 Implementation: RL Driving

We will implement two scripts:
1.  `q_learning_grid.py`: Tabular Q-Learning from scratch.
2.  `dqn_cartpole.py`: Deep RL using Stable Baselines (CartPole is the "Hello World" of control).

### 🛠️ Setup
Create `week5_day33`.

```bash
mkdir -p ~/ros2_ws/src/week5_day33
cd ~/ros2_ws/src/week5_day33
pip install gymnasium stable-baselines3 shimmy
touch q_learning_grid.py dqn_cartpole.py
```

### 👨‍💻 Code: Tabular Q-Learning (Grid World)

```python
import numpy as np
import random
import time
import os

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.agent_pos = [0, 0]
        self.goal_pos = [size-1, size-1]
        self.obstacles = [[1, 1], [2, 2], [3, 3]] # Diagonal wall
        
    def reset(self):
        self.agent_pos = [0, 0]
        return tuple(self.agent_pos)

    def step(self, action):
        # 0: Up, 1: Down, 2: Left, 3: Right
        x, y = self.agent_pos
        
        if action == 0: x = max(0, x-1)
        elif action == 1: x = min(self.size-1, x+1)
        elif action == 2: y = max(0, y-1)
        elif action == 3: y = min(self.size-1, y+1)
        
        self.agent_pos = [x, y]
        state = tuple(self.agent_pos)
        
        # Reward
        if state == tuple(self.goal_pos):
            return state, 10, True # Goal reached
        elif list(state) in self.obstacles:
            return state, -10, True # Crash
        else:
            return state, -1, False # Step cost

def run_q_learning():
    env = GridWorld()
    q_table = np.zeros((env.size, env.size, 4)) # State x Action
    
    # Hyperparameters
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    episodes = 500
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Epsilon-Greedy Action Selection
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3) # Explore
            else:
                action = np.argmax(q_table[state]) # Exploit
                
            next_state, reward, done = env.step(action)
            
            # Bellman Update
            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])
            
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state][action] = new_value
            
            state = next_state
            
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if episode % 50 == 0:
            print(f"Episode {episode}: Epsilon {epsilon:.2f}")

    print("Training Complete. Testing...")
    
    # Test Run
    state = env.reset()
    done = False
    path = [state]
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done = env.step(action)
        path.append(state)
        if len(path) > 20: break # Infinite loop guard
        
    print(f"Path: {path}")

if __name__ == "__main__":
    run_q_learning()
```

### 👨‍💻 Code: Deep RL (PPO) with Stable Baselines

We use **PPO (Proximal Policy Optimization)**, which is generally more stable than DQN for continuous control.

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

def train_ppo():
    # 1. Create Environment
    # MountainCar-v0: Drive a car up a hill (requires momentum)
    env_id = "MountainCar-v0"
    env = make_vec_env(env_id, n_envs=4)

    # 2. Instantiate the Agent
    # MlpPolicy: Multi-Layer Perceptron (Standard NN)
    model = PPO("MlpPolicy", env, verbose=1)

    # 3. Train
    print("Training PPO Agent...")
    model.learn(total_timesteps=100000)
    
    # 4. Save
    model.save("ppo_mountaincar")
    print("Model Saved.")

    # 5. Evaluate
    env = gym.make(env_id, render_mode="human")
    obs, _ = env.reset()
    
    print("Running Evaluation...")
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
            
    env.close()

if __name__ == "__main__":
    train_ppo()
```

---

## 🔬 Lab Exercise: Reward Shaping

### Lab Objectives
1.  Run `q_learning_grid.py`.
2.  **Observation:** The agent finds the shortest path avoiding obstacles.
3.  **Experiment:** Change the step cost from `-1` to `0`.
    -   *Result:* The agent might wander around aimlessly because there is no penalty for wasting time.
4.  **Experiment:** Change the crash penalty from `-10` to `-1`.
    -   *Result:* The agent might decide that crashing is a valid shortcut if the goal is close.
5.  **Lesson:** "You get what you incentivize." Reward Shaping is the hardest part of RL.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Agent doesn't learn
**Symptom:** Random behavior even after training.
**Cause:**
-   Hyperparameters (Learning rate too high/low).
-   Reward is too sparse (Goal is too hard to find randomly).
**Solution:** Use **Curriculum Learning** (Start with easy task, make it harder).

#### 2. Catastrophic Forgetting
**Symptom:** Agent learns, then suddenly performance drops to zero.
**Cause:** New experiences overwrite old important ones.
**Solution:** Increase Replay Buffer size. Lower learning rate.

#### 3. Simulation to Reality Gap (Sim2Real)
**Symptom:** Works in Gym, fails on real robot.
**Cause:** Simulator physics $\neq$ Real physics.
**Solution:** Domain Randomization (Randomize friction, mass, colors in Sim during training).

---

## ⚡ Optimization & Best Practices

### 1. Vectorized Environments
Train on 16 CPU cores simultaneously.
-   `make_vec_env(..., n_envs=16)`
-   Speeds up data collection by 16x.

### 2. Continuous Action Space
DQN only works for discrete actions (Left, Right).
For steering (Continuous angle), use **PPO** or **SAC (Soft Actor-Critic)**.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between Model-Based (MPC) and Model-Free (RL) control?
    *   **A:** MPC uses a physics model to predict. RL learns the policy directly from experience without necessarily knowing the physics.
2.  **Q:** What is $\epsilon$-greedy?
    *   **A:** A strategy where the agent explores (random action) with probability $\epsilon$ and exploits (best known action) with probability $1-\epsilon$.
3.  **Q:** Why do we use a Discount Factor $\gamma$?
    *   **A:** To value immediate rewards more than distant rewards, and to ensure the mathematical series converges for infinite horizons.

### Challenge Task
**Task:** Highway Overtaking RL.
1.  Use `highway-env` (pip install highway-env).
2.  Train a DQN agent to navigate traffic.
3.  Observe it learning to change lanes to maintain speed.

---

## 📚 Further Reading & References
-   [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/) - Best intro guide.
-   [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)

---

**Day 33 Complete** | Phase 4: ADAS & Robotics Systems | Week 5: Path Planning & Decision Making
