# Day 75: Multi-Agent Reinforcement Learning (MARL) Basics
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing Simple MARL (Independent Q-Learning)

We will simulate a Grid World where two agents try to meet.
**Approach:** Independent Q-Learning (IQL). Each agent treats the other as part of the environment.

```python
import numpy as np
import random

GRID_SIZE = 5
ACTIONS = [(0,1), (0,-1), (1,0), (-1,0)] # R, L, D, U

class QAgent:
    def __init__(self, name, start_pos):
        self.name = name
        self.pos = start_pos
        self.q_table = {} # (x, y) -> [q_val_0, q_val_1, ...]
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def get_action(self):
        if self.pos not in self.q_table:
            self.q_table[self.pos] = np.zeros(4)
            
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[self.pos])

    def update(self, action, reward, next_pos):
        if next_pos not in self.q_table:
            self.q_table[next_pos] = np.zeros(4)
            
        old_q = self.q_table[self.pos][action]
        next_max = np.max(self.q_table[next_pos])
        
        # Bellman Equation
        new_q = old_q + self.lr * (reward + self.gamma * next_max - old_q)
        self.q_table[self.pos][action] = new_q
        self.pos = next_pos

# Environment
def step(agent1, agent2, a1, a2):
    # Move Agent 1
    d1 = ACTIONS[a1]
    n1 = (max(0, min(GRID_SIZE-1, agent1.pos[0] + d1[0])),
          max(0, min(GRID_SIZE-1, agent1.pos[1] + d1[1])))
    
    # Move Agent 2
    d2 = ACTIONS[a2]
    n2 = (max(0, min(GRID_SIZE-1, agent2.pos[0] + d2[0])),
          max(0, min(GRID_SIZE-1, agent2.pos[1] + d2[1])))
    
    # Reward: +10 if they meet, -1 step cost
    reward = -1
    done = False
    if n1 == n2:
        reward = 10
        done = True
        
    return n1, n2, reward, done

# Training Loop
a1 = QAgent("A1", (0,0))
a2 = QAgent("A2", (4,4))

for episode in range(1000):
    a1.pos = (0,0)
    a2.pos = (4,4)
    
    for t in range(20):
        act1 = a1.get_action()
        act2 = a2.get_action()
        
        n1, n2, r, done = step(a1, a2, act1, act2)
        
        # Update both
        a1.update(act1, r, n1)
        a2.update(act2, r, n2)
        
        if done:
            break
            
print("Training Complete. Agents learned to meet.")
```

### PettingZoo (Gym for MARL)

Just as OpenAI Gym is the standard for RL, **PettingZoo** is the standard for MARL.

```python
from pettingzoo.butterfly import pistonball_v6

env = pistonball_v6.env()
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        # Random policy
        action = env.action_space(agent).sample() 
        
    env.step(action)
```

### Shared Experience Replay

In IQL, agents learn slowly.
In **Shared Experience**, agents share a replay buffer.
*   Agent A tries something.
*   Agent B learns from Agent A's experience.
*   This assumes agents are homogeneous (same capabilities).

### Summary

*   **IQL:** Simple, but unstable.
*   **PettingZoo:** The library to use.
*   **Coordination:** Hardest part. Agents might learn to run in circles if not rewarded correctly.
