# Day 2: The Bellman Equations

## 1. Value Functions
To solve an MDP, we need to know "how good" a state is.
*   **State-Value Function $V_{\pi}(s)$:** The expected return starting from state $s$ and following policy $\pi$ thereafter.
    $$ V_{\pi}(s) = \mathbb{E}_{\pi} [G_t | S_t = s] $$
*   **Action-Value Function $Q_{\pi}(s, a)$:** The expected return starting from state $s$, taking action $a$, and *then* following policy $\pi$.
    $$ Q_{\pi}(s, a) = \mathbb{E}_{\pi} [G_t | S_t = s, A_t = a] $$

## 2. The Bellman Expectation Equation
The value of a state is the immediate reward plus the discounted value of the next state. This recursive relationship is the **Bellman Equation**.

### For $V_{\pi}(s)$:
$$ V_{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s', r} P(s', r | s, a) [r + \gamma V_{\pi}(s')] $$
*   Average over actions (policy $\pi$).
*   Average over dynamics (transition $P$).

### For $Q_{\pi}(s, a)$:
$$ Q_{\pi}(s, a) = \sum_{s', r} P(s', r | s, a) [r + \gamma V_{\pi}(s')] $$
$$ Q_{\pi}(s, a) = \sum_{s', r} P(s', r | s, a) [r + \gamma \sum_{a'} \pi(a'|s') Q_{\pi}(s', a')] $$

## 3. Code Example: Calculating V for a Random Policy
Using the GridWorld from Day 1, let's calculate $V(s)$ for a uniform random policy ($\pi(a|s) = 0.25$ for all 4 actions).

```python
import numpy as np

# GridWorld Setup (Simplified)
size = 3
gamma = 0.9
states = [(i, j) for i in range(size) for j in range(size)]
goal = (2, 2)
V = {s: 0.0 for s in states} # Initialize V(s) = 0

def get_transitions(state, action):
    # Returns (next_state, reward)
    x, y = state
    if state == goal: return (state, 0) # Terminal state stays
    
    if action == 0: x = max(0, x - 1)
    elif action == 1: y = min(size - 1, y + 1)
    elif action == 2: x = min(size - 1, x + 1)
    elif action == 3: y = max(0, y - 1)
    
    next_state = (x, y)
    reward = 1 if next_state == goal else 0
    return next_state, reward

# Bellman Update (One Iteration)
def bellman_update(V, state):
    if state == goal: return 0
    
    value_sum = 0
    for action in [0, 1, 2, 3]:
        next_state, reward = get_transitions(state, action)
        # Bellman Expectation: r + gamma * V(s')
        value_sum += 0.25 * (reward + gamma * V[next_state])
        
    return value_sum

# Run for 100 iterations
for _ in range(100):
    new_V = V.copy()
    for s in states:
        new_V[s] = bellman_update(V, s)
    V = new_V

print("Value Function:")
for i in range(size):
    row = [f"{V[(i, j)]:.2f}" for j in range(size)]
    print(row)
```

### Key Takeaways
*   $V(s)$ decomposes into immediate reward + discounted future value.
*   We can iteratively update $V$ using this relationship (Policy Evaluation).
