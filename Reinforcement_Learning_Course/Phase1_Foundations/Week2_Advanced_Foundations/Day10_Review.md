# Day 10: RL Foundations Review & Mini-Project

## 1. Phase 1 Review
We have covered the pillars of Reinforcement Learning:
1.  **MDPs:** The problem statement ($S, A, P, R, \gamma$).
2.  **Bellman Equations:** The recursive structure of value functions.
3.  **Dynamic Programming:** Solving MDPs when the model is known (Planning).
4.  **Monte Carlo:** Learning from complete episodes (Model-Free, High Variance).
5.  **TD Learning:** Learning from steps (Bootstrapping, Low Variance).
6.  **Function Approximation:** Scaling to large state spaces.
7.  **Bandits & Exploration:** The core dilemma of RL.

## 2. Mini-Project: GridWorld Benchmark
Your task is to build a comprehensive RL benchmark on a custom GridWorld.

### Requirements
1.  **Environment:** Create a 5x5 GridWorld with:
    *   Start state (0,0).
    *   Goal state (4,4) with reward +10.
    *   Obstacles (walls) and Pits (reward -10).
    *   Step cost -0.1 (to encourage shortest path).
2.  **Agents:** Implement the following algorithms as classes:
    *   `ValueIterationAgent` (Planning).
    *   `MonteCarloAgent` (First-visit MC Control).
    *   `SARSAAgent` (On-policy TD).
    *   `QLearningAgent` (Off-policy TD).
3.  **Comparison:**
    *   Train each agent for 1000 episodes.
    *   Plot the **Cumulative Reward per Episode** (smoothed).
    *   Visualize the final **Value Function** (heatmap) and **Policy** (arrows).

## 3. Code Skeleton
```python
import numpy as np
import matplotlib.pyplot as plt

class GridEnvironment:
    def __init__(self):
        # Define grid, start, goal, obstacles
        pass
    def step(self, action):
        # Return next_state, reward, done
        pass
    def reset(self):
        pass

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = {}
        pass
    def act(self, state):
        pass
    def update(self, state, action, reward, next_state):
        pass

# Main Loop
env = GridEnvironment()
agent = QLearningAgent(actions=[0,1,2,3])
rewards = []

for episode in range(1000):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    rewards.append(total_reward)

plt.plot(rewards)
plt.show()
```
