# Lab 03: Value Iteration

## Difficulty
🟡 Medium

## Problem Statement
Implement Value Iteration to find the optimal value function for the GridWorld.
`V(s) = max_a sum_s' P(s'|s,a) [R(s,a,s') + gamma * V(s')]`

## Starter Code
```python
import numpy as np

def value_iteration(env, gamma=0.99, theta=1e-6):
    V = np.zeros((env.size, env.size))
    while True:
        delta = 0
        # TODO: Iterate over all states
        # TODO: Update V[s]
        if delta < theta:
            break
    return V
```
