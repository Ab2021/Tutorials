# Lab 01: Multi-Armed Bandit

## Difficulty
🟢 Easy

## Problem Statement
Implement the Epsilon-Greedy algorithm to solve the Multi-Armed Bandit problem.
- `n_arms`: Number of slot machines.
- `true_probs`: True probability of winning for each arm (hidden).
- Agent must balance exploration (epsilon) and exploitation.

## Starter Code
```python
import numpy as np

class BanditAgent:
    def __init__(self, n_arms, epsilon):
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        
    def select_arm(self):
        # TODO: Implement epsilon-greedy
        pass
        
    def update(self, arm, reward):
        # TODO: Update Q-values
        pass
```
