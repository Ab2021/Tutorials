# Lab 04: Policy Iteration

## Difficulty
🟡 Medium

## Problem Statement
Implement Policy Iteration:
1. **Policy Evaluation**: Calculate V for current policy.
2. **Policy Improvement**: Update policy to be greedy with respect to V.
Repeat until stable.

## Starter Code
```python
def policy_iteration(env, gamma=0.99):
    policy = np.ones((env.size, env.size, 4)) / 4  # Uniform random
    while True:
        # TODO: Evaluate Policy
        # TODO: Improve Policy
        pass
    return policy
```
