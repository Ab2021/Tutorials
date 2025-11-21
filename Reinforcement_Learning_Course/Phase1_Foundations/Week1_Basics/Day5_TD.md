# Day 5: Temporal Difference (TD) Learning

## 1. Combining MC and DP
TD Learning is the "best of both worlds":
*   **Like MC:** It learns from raw experience (model-free).
*   **Like DP:** It updates estimates based on other estimates (bootstrapping).
TD updates the value function *after every step*, rather than waiting for the end of the episode.

## 2. TD(0) Prediction
$$ V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)] $$
*   **TD Target:** $R_{t+1} + \gamma V(S_{t+1})$ (The estimated return).
*   **TD Error ($\delta_t$):** $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ (Difference between target and current estimate).

## 3. SARSA (On-Policy Control)
SARSA stands for **S**tate, **A**ction, **R**eward, **S**tate', **A**ction'.
It updates the Q-value based on the action *actually taken* by the current policy.
$$ Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma Q(S', A') - Q(S, A)] $$
*   The agent chooses $A'$ using its current policy (e.g., $\epsilon$-greedy).
*   Safe: It learns the value of the "safe" policy (including exploration risks).

## 4. Q-Learning (Off-Policy Control)
Q-Learning updates the Q-value based on the *best possible* action in the next state.
$$ Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \max_{a} Q(S', a) - Q(S, A)] $$
*   The agent might take action $A'$ (exploration), but it updates assuming it took the greedy action.
*   Aggressive: It learns the optimal value $Q_*$ directly.

## 5. Code Example: Q-Learning vs SARSA on CliffWalking
In "Cliff Walking", falling off the cliff gives -100 reward.
*   **Q-Learning** learns the optimal path (right along the edge).
*   **SARSA** learns a safer path (further from the edge) to avoid falling due to $\epsilon$-greedy exploration.

```python
import numpy as np

# Simplified Environment
# 0: Up, 1: Right, 2: Down, 3: Left
# Grid 4x12. Start (3,0), Goal (3,11). Cliff (3, 1..10).

def step(state, action):
    x, y = state
    if action == 0: x = max(0, x-1)
    elif action == 1: y = min(11, y+1)
    elif action == 2: x = min(3, x+1)
    elif action == 3: y = max(0, y-1)
    
    next_state = (x, y)
    reward = -1
    done = False
    
    # Cliff check
    if 3 == x and 1 <= y <= 10:
        reward = -100
        next_state = (3, 0) # Reset to start
    elif next_state == (3, 11):
        done = True
        
    return next_state, reward, done

# Q-Learning Update
def q_learning(episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    Q = {} # Sparse dict
    for _ in range(episodes):
        state = (3, 0)
        done = False
        while not done:
            # Epsilon-Greedy
            if np.random.rand() < epsilon: action = np.random.randint(4)
            else: 
                qs = [Q.get((state, a), 0) for a in range(4)]
                action = np.argmax(qs)
            
            next_state, reward, done = step(state, action)
            
            # Update
            old_q = Q.get((state, action), 0)
            next_max = max([Q.get((next_state, a), 0) for a in range(4)])
            new_q = old_q + alpha * (reward + gamma * next_max - old_q)
            Q[(state, action)] = new_q
            
            state = next_state
    return Q

# Run
Q_table = q_learning()
print("Q-Learning finished. Value of Start (Right):", Q_table.get(((3,0), 1), 0))
```

### Key Takeaways
*   TD bootstraps (updates from guess).
*   SARSA is safer during training (on-policy).
*   Q-Learning is optimal (off-policy).
