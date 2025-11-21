# Day 3: Dynamic Programming (DP)

## 1. Planning vs. Learning
*   **Planning:** We have the full model of the environment ($P$ and $R$). We use it to compute the optimal policy. This is what DP does.
*   **Learning:** We don't know the model. We learn from interaction. (This comes later with Monte Carlo and TD).

## 2. Policy Iteration
Policy Iteration consists of two alternating steps:
1.  **Policy Evaluation:** Calculate $V_{\pi}(s)$ for the current policy $\pi$.
    $$ V_{k+1}(s) = \sum_{s', r} P(s', r | s, \pi(s)) [r + \gamma V_k(s')] $$
    Repeat until convergence ($V_k \approx V_{\pi}$).
2.  **Policy Improvement:** Generate a better policy $\pi'$ by acting greedily with respect to $V_{\pi}$.
    $$ \pi'(s) = \arg\max_a \sum_{s', r} P(s', r | s, a) [r + \gamma V_{\pi}(s')] $$

We repeat these steps until the policy stops changing.

## 3. Value Iteration
Policy Iteration can be slow because it requires full evaluation in every step.
**Value Iteration** combines evaluation and improvement into a single update step (using the Bellman Optimality Equation directly).
$$ V_{k+1}(s) = \max_a \sum_{s', r} P(s', r | s, a) [r + \gamma V_k(s')] $$
Once $V$ converges to $V_*$, we extract the optimal policy.

## 4. Code Example: Value Iteration
```python
import numpy as np

# GridWorld Setup
size = 4
gamma = 0.9
goal = (3, 3)
states = [(i, j) for i in range(size) for j in range(size)]
V = {s: 0.0 for s in states}
actions = [0, 1, 2, 3] # Up, Right, Down, Left

def get_transitions(state, action):
    x, y = state
    if state == goal: return [(1.0, state, 0)] # Terminal
    
    if action == 0: x = max(0, x - 1)
    elif action == 1: y = min(size - 1, y + 1)
    elif action == 2: x = min(size - 1, x + 1)
    elif action == 3: y = max(0, y - 1)
    
    next_state = (x, y)
    reward = 1 if next_state == goal else 0
    return [(1.0, next_state, reward)] # Prob, Next, Reward

# Value Iteration Loop
theta = 1e-4 # Convergence threshold
while True:
    delta = 0
    new_V = V.copy()
    for s in states:
        if s == goal: continue
        
        q_values = []
        for a in actions:
            q_val = 0
            for prob, next_s, r in get_transitions(s, a):
                q_val += prob * (r + gamma * V[next_s])
            q_values.append(q_val)
        
        best_value = max(q_values)
        delta = max(delta, abs(best_value - V[s]))
        new_V[s] = best_value
        
    V = new_V
    if delta < theta:
        break

print("Optimal Value Function:")
for i in range(size):
    print([f"{V[(i, j)]:.2f}" for j in range(size)])
```

### Key Takeaways
*   DP assumes a known model.
*   Policy Iteration: Evaluate -> Improve.
*   Value Iteration: Directly iterate on Bellman Optimality.
