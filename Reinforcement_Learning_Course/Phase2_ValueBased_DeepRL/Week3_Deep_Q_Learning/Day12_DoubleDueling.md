# Day 12: Double DQN & Dueling DQN

## 1. The Problem: Overestimation Bias
Standard Q-Learning (and DQN) uses the `max` operator to calculate the target:
$$ Y_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta^-) $$
*   **Issue:** If the Q-values have random noise, taking the `max` tends to pick the largest noise, leading to systematic **overestimation** of values.
*   **Analogy:** If you have 10 clocks and all are slightly wrong (random error), the one that is "fastest" is likely ahead of the true time. If you always trust the fastest clock, you will always be early.

## 2. Double DQN (DDQN)
DDQN decouples **action selection** from **action evaluation**.
1.  **Selection:** Use the *current* network $\theta$ to pick the best action.
    $$ a^* = \arg\max_a Q(S_{t+1}, a; \theta) $$
2.  **Evaluation:** Use the *target* network $\theta^-$ to evaluate that action.
    $$ Y_t = R_{t+1} + \gamma Q(S_{t+1}, a^*; \theta^-) $$
This simple change significantly reduces overestimation.

## 3. Dueling DQN
Dueling DQN changes the *architecture* of the neural network.
Instead of outputting $Q(s, a)$ directly, it splits into two streams:
1.  **Value Stream $V(s)$:** How good is the state? (Scalar).
2.  **Advantage Stream $A(s, a)$:** How much better is action $a$ than the average action? (Vector).

$$ Q(s, a) = V(s) + (A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a')) $$
*   **Benefit:** The agent can learn $V(s)$ (which states are good/bad) without learning the effect of every action. This is useful in states where actions don't matter much (e.g., driving on a straight empty road).

## 4. Code Example: Dueling Architecture
```python
import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Feature Extractor (Shared)
        self.features = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        feats = self.features(x)
        values = self.value_stream(feats)
        advantages = self.advantage_stream(feats)
        
        # Aggregation Layer (Mean subtraction for stability)
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

# Test
net = DuelingDQN(state_dim=4, action_dim=2)
input_state = torch.randn(1, 4)
output_q = net(input_state)
print("Q-Values:", output_q)
```

### Key Takeaways
*   **Double DQN:** Fixes math (overestimation).
*   **Dueling DQN:** Fixes architecture (better generalization).
*   Both are standard improvements over vanilla DQN.
