# Day 18: Hierarchical RL (h-DQN)

## 1. The Problem of Long Horizons
Standard RL struggles when the goal is very far away (e.g., "Go to the airport, catch a plane, fly to London").
The sequence of atomic actions (step left, step right) is too long.
**Hierarchical RL (HRL)** breaks the problem into **sub-goals** (Temporal Abstraction).

## 2. Hierarchical-DQN (h-DQN) Architecture
h-DQN (Kulkarni et al., 2016) uses two levels of hierarchy:

### A. Meta-Controller (High Level)
*   **Input:** State $s$.
*   **Output:** A **Goal** $g$ (e.g., "Reach the door").
*   **Timescale:** Operates slowly (picks a goal every $N$ steps or when the previous goal is reached).
*   **Reward:** Extrinsic reward $R_{ext}$ (from the environment).
*   **Objective:** Maximize cumulative extrinsic reward.

### B. Controller (Low Level)
*   **Input:** State $s$ AND Goal $g$.
*   **Output:** Atomic Action $a$ (e.g., "Move Joystick Left").
*   **Timescale:** Operates at every step.
*   **Reward:** Intrinsic reward $R_{int}(s, g)$ (e.g., +1 if state $s$ matches goal $g$, else 0).
*   **Objective:** Maximize cumulative intrinsic reward (reach the goal).

## 3. Training
Both levels are trained simultaneously using Q-Learning (DQN).
*   **Meta-Controller:** Learns $Q_{meta}(s, g)$.
*   **Controller:** Learns $Q_{controller}(s, a, g)$.

## 4. Code Example: Two-Level Architecture
```python
import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()
        # Input takes both state and goal
        self.fc = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=1)
        return self.fc(x)

class MetaController(nn.Module):
    def __init__(self, state_dim, num_goals):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_goals) # Output is a discrete goal ID
        )
    def forward(self, state):
        return self.fc(state)

# Interaction Loop
# 1. Meta-Controller picks goal
goal = meta_controller(state).argmax()

# 2. Controller executes for N steps
for t in range(N):
    action = controller(state, goal_embedding[goal]).argmax()
    next_state, reward, done = env.step(action)
    
    # 3. Calculate Intrinsic Reward (Did we reach goal?)
    r_int = 1.0 if check_goal(next_state, goal) else 0.0
    
    # Update Controller with r_int
    # Update Meta-Controller with accumulated extrinsic reward
```

### Key Takeaways
*   HRL decomposes tasks into sub-problems.
*   **Temporal Abstraction:** Decisions are made at different time scales.
*   Allows solving much harder, sparse-reward problems (like Montezuma's Revenge).
