# Day 11: Deep Q-Networks (DQN)

## 1. From Q-Learning to Deep Q-Learning
Tabular Q-Learning stores $Q(s, a)$ in a table. This fails for high-dimensional states (like pixels).
**DQN** replaces the table with a Neural Network $Q(s, a; \theta)$ that approximates the Q-values.
*   **Input:** State $s$ (e.g., 4 stacked frames of a game).
*   **Output:** Q-values for all actions $a \in A$.
*   **Loss:** Mean Squared Error between Prediction and Target.
    $$ L(\theta) = \mathbb{E} [ (R + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2 ] $$

## 2. Key Innovations of DQN
Naive Deep RL is unstable due to correlations in data and moving targets. DQN introduced two tricks to fix this:

### A. Experience Replay
*   Instead of training on the current step tuple $(s, a, r, s')$, we store it in a **Replay Buffer** $D$.
*   We sample a **random batch** from $D$ to train the network.
*   **Benefit:** Breaks temporal correlations (data becomes i.i.d.) and allows reusing data (sample efficiency).

### B. Target Network
*   The target $y = r + \gamma \max Q(s', a'; \theta)$ depends on the weights $\theta$. If we update $\theta$, the target moves, leading to "chasing your own tail".
*   **Solution:** Use a separate **Target Network** with weights $\theta^-$.
*   $\theta^-$ is a copy of $\theta$ that is frozen and updated only every $C$ steps (e.g., every 1000 steps).
*   **Benefit:** Stabilizes the learning target.

## 3. Code Example: Minimal DQN in PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict()) # Sync
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.action_dim = action_dim
        
    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_t)
            return q_values.argmax().item()
            
    def train(self):
        if len(self.memory) < self.batch_size: return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Current Q
        curr_q = self.q_net(states).gather(1, actions)
        
        # Target Q (using Target Net)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        loss = nn.MSELoss()(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
```

### Key Takeaways
*   DQN = Q-Learning + Neural Net + Replay Buffer + Target Net.
*   It was the first algorithm to reach human-level performance on Atari.
