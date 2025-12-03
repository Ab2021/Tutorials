# Day 139: Reinforcement Learning for Control (DQN/PPO)
## Phase 4: ADAS & Robotics Systems | Week 20: Path Planning & Decision Making

---

> **📝 Day 139 Focus:**
> Classical control (PID/MPC) requires a physics model. **Reinforcement Learning (RL)** learns from experience. The agent tries actions, gets rewards (or crashes), and learns to maximize the score. It's the "End-to-End" approach to driving.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the RL components: Agent, Environment, State, Action, Reward.
2.  **Explain** the Q-Learning update rule and the Bellman Equation.
3.  **Implement** a Deep Q-Network (DQN) using PyTorch.
4.  **Train** an agent to solve the `CartPole` or `MountainCar` environment (OpenAI Gym).
5.  **Discuss** the challenges of RL in real-world driving (Sim-to-Real gap).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 120:** Neural Networks.
-   **Probability:** Markov Decision Process (MDP).

### Hardware Requirements
-   **GPU:** Recommended for training.

### Software Stack
-   **Python:** `gymnasium` (formerly gym), `torch`, `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The MDP (Markov Decision Process)

-   **State ($S_t$):** What the robot sees (Camera, Lidar).
-   **Action ($A_t$):** What the robot does (Steer, Gas).
-   **Reward ($R_t$):** Feedback (+1 for moving, -100 for crash).
-   **Goal:** Maximize Return $G_t = \sum \gamma^k R_{t+k}$.

### 🔹 Part 2: Q-Learning

We want to learn a function $Q(s, a)$ that predicts the expected return.
$$ Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)] $$
-   **Bellman Equation:** The value of this state is the immediate reward plus the value of the next best state.

### 🔹 Part 3: Deep Q-Network (DQN)

If the state space is huge (Images), we can't use a table. We use a Neural Network to approximate $Q(s, a)$.
-   **Experience Replay:** Store transitions $(s, a, r, s')$ in a buffer and sample randomly to break correlations.
-   **Target Network:** Use a frozen copy of the network to calculate targets for stability.

---

## 💻 Implementation: DQN on CartPole

**Scenario:**
-   Balance a pole on a cart.
-   State: [Cart Pos, Cart Vel, Pole Angle, Pole Vel].
-   Action: [Left, Right].
-   Reward: +1 for every step the pole stays up.

### 🛠️ Setup
Create `week20_day139` and `dqn_agent.py`.

```bash
mkdir -p ~/ros2_ws/src/week20_day139
cd ~/ros2_ws/src/week20_day139
pip install gymnasium[classic_control]
touch dqn_agent.py
```

### 👨‍💻 Code: DQN Implementation

```python
import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# --- 1. The Network ---
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- 2. The Agent ---
class Agent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.n_obs = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        
        self.policy_net = DQN(self.n_obs, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_obs, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)
        self.memory = deque(maxlen=10000)
        
        self.steps_done = 0
        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
            
        transitions = random.sample(self.memory, self.batch_size)
        # Transpose the batch
        batch = list(zip(*transitions))
        
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_state_batch = torch.cat(batch[3])
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Huber Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

def main():
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    
    num_episodes = 200 # Increase for better results
    durations = []
    
    print("Training...")
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        
        for t in range(1000):
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=agent.device)
            done = terminated or truncated
            
            if terminated:
                next_state = torch.zeros_like(state) # Zero state for terminal
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)
                
            agent.memory.append((state, action, reward, next_state))
            state = next_state
            
            agent.optimize_model()
            
            # Soft Update Target Net
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            tau = 0.005
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            agent.target_net.load_state_dict(target_net_state_dict)
            
            if done:
                durations.append(t + 1)
                print(f"Episode {i_episode}: Duration {t+1}")
                break
                
    print("Training Complete")
    plt.plot(durations)
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Learning Curve

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   Episodes 0-50: Duration is low (~10-20 steps). The pole falls immediately.
    -   Episodes 100+: Duration increases (~100-200 steps). The agent learns to balance.
3.  **Experiment:**
    -   Change `gamma` to 0.1 (Short-sighted).
    -   **Result:** The agent fails. It only cares about the immediate +1 reward, not the future crash.
    -   **Lesson:** Tuning Hyperparameters ($\gamma, \alpha, \epsilon$) is the hardest part of RL.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Catastrophic Forgetting
**Symptom:** Agent learns well, then suddenly drops to zero performance.
**Cause:** Neural Network overfits to recent data in the buffer.
**Solution:** Increase Replay Buffer size. Lower Learning Rate.

#### 2. Sparse Rewards
**Symptom:** Agent never learns anything.
**Cause:** Reward is 0 everywhere except the goal. Random exploration never finds the goal.
**Solution:** **Reward Shaping**. Give partial rewards for getting closer to the goal.

---

## ⚡ Optimization & Best Practices

### 1. PPO (Proximal Policy Optimization)
DQN is unstable. PPO is the modern standard (used by OpenAI).
-   **Policy Gradient:** Optimizes the policy directly (probabilities) rather than Q-values.
-   **Clipped Objective:** Prevents large updates that destroy the policy.

### 2. Sim-to-Real
Training on a real car is dangerous (crashes).
-   **Domain Randomization:** Train in sim with random colors, friction, and lighting.
-   **Result:** The agent learns to be robust enough to handle the real world.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is Exploration vs Exploitation?
    *   **A:** Exploration: Trying random actions to find new strategies ($\epsilon$). Exploitation: Using the best known strategy to maximize reward ($1-\epsilon$).
2.  **Q:** Why do we need a Target Network?
    *   **A:** To stabilize training. If we update the network using its own predictions as targets, the target moves constantly (chasing your own tail), leading to oscillation.
3.  **Q:** Can RL replace MPC?
    *   **A:** In theory, yes. In practice, MPC is safer and more explainable. RL is often used for high-level decisions (Behavior) or complex maneuvers where modeling is hard.

### Challenge Task
**Task:** MountainCar.
1.  Change env to `gym.make("MountainCar-v0")`.
2.  Reward is -1 per step. Goal is to reach the top.
3.  DQN struggles here because it needs momentum (go left to go right).
4.  Try increasing `epsilon` decay to encourage more exploration.

---

## 📚 Further Reading & References
-   [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)
-   [Deep Learning for Self-Driving Cars (MIT)](https://deeplearning.mit.edu/)

---

**Day 139 Complete** | Phase 4: ADAS & Robotics Systems | Week 20: Path Planning & Decision Making
