# Day 20: Phase 2 Review & Project

## 1. Phase 2 Review: Value-Based Deep RL
We have covered the evolution of Deep Q-Learning:
1.  **DQN:** The breakthrough. NN + Replay Buffer + Target Net.
2.  **Double DQN:** Fixing overestimation bias.
3.  **Dueling DQN:** Separating Value and Advantage.
4.  **PER:** Prioritizing important samples.
5.  **Noisy Nets:** Better exploration.
6.  **Distributional RL:** Learning the full return distribution.
7.  **Rainbow:** Combining everything.
8.  **Advanced Topics:** DRQN (Memory), NAF (Continuous), h-DQN (Hierarchy), IRL (Rewards).

## 2. Project: LunarLander-v2 with Rainbow
Your task is to implement a simplified **Rainbow DQN** agent to solve `LunarLander-v2`.

### Requirements
1.  **Environment:** `gymnasium.make("LunarLander-v2")`.
2.  **Agent:** Implement a class `RainbowAgent` that includes:
    *   **Double DQN** update rule.
    *   **Dueling** Network Architecture.
    *   **Prioritized Experience Replay** (use a library like `cpprb` or implement a simple SumTree).
    *   (Optional) Noisy Nets or N-step returns.
3.  **Training:**
    *   Train for 500 episodes.
    *   Goal: Average reward > 200 over 100 episodes.
4.  **Visualization:**
    *   Plot the Learning Curve (Reward vs Episode).
    *   Record a video of the trained agent landing safely.

## 3. Code Skeleton
```python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DuelingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Implement Dueling Architecture
        pass

class RainbowAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = DuelingNetwork(state_dim, action_dim)
        self.target_net = DuelingNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        # Initialize Replay Buffer
        pass
        
    def act(self, state, epsilon=0.0):
        # Epsilon-greedy action selection
        pass
        
    def update(self, batch_size):
        # 1. Sample from buffer (with priorities)
        # 2. Compute Double DQN Loss
        # 3. Update weights
        # 4. Update priorities
        pass

# Training Loop
env = gym.make("LunarLander-v2")
agent = RainbowAgent(env.observation_space.shape[0], env.action_space.n)

for episode in range(500):
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.act(state, epsilon=0.1)
        next_state, reward, done, _, _ = env.step(action)
        # Store in buffer
        agent.update(batch_size=64)
        state = next_state
```

### Key Takeaways
*   Implementing these algorithms builds intuition.
*   Debugging RL is an art (check gradients, check buffer statistics).
