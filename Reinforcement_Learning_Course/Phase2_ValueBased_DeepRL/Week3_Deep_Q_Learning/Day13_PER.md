# Day 13: Prioritized Experience Replay (PER)

## 1. The Problem with Uniform Sampling
Standard DQN samples transitions uniformly from the replay buffer.
*   **Inefficiency:** Many transitions are "boring" (e.g., walking in a straight line with no reward). The agent already predicts them well.
*   **Idea:** We should focus on "surprising" transitions where the agent has a lot to learn.

## 2. Priority Metric: TD Error
We use the **TD Error** $\delta$ as a proxy for how "surprising" or valuable a transition is.
$$ \delta = |R + \gamma \max Q(S', a') - Q(S, a)| $$
*   High $|\delta|$ $\implies$ High prediction error $\implies$ High priority.
*   Low $|\delta|$ $\implies$ We already know this $\implies$ Low priority.

## 3. Stochastic Prioritization
If we strictly sample the highest errors (Greedy Prioritization), we risk overfitting to a small subset of noisy transitions.
Instead, we sample with probability proportional to priority:
$$ P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha} $$
*   $p_i = |\delta_i| + \epsilon$ (small constant to ensure non-zero prob).
*   $\alpha$: Controls how much prioritization we use (0 = Uniform, 1 = Greedy).

## 4. Bias Correction (Importance Sampling)
Prioritized sampling changes the data distribution. The agent sees "hard" samples more often than they actually occur. This introduces **bias**.
To fix this, we use **Importance Sampling Weights**:
$$ w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta $$
*   We scale the loss by $w_i$.
*   If $P(i)$ is high (frequently sampled), $w_i$ is low (downweight the loss).
*   $\beta$: Anneals from $\beta_0$ to 1 over training.

## 5. Code Example: SumTree (Concept)
Implementing PER efficiently requires a **SumTree** (Segment Tree) data structure to sample in $O(\log N)$ time.

```python
import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        
    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        
    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change
            
    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree): break
            
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
    
    def total(self):
        return self.tree[0]

# Usage
# buffer = SumTree(1000)
# buffer.add(priority=1.5, data=(s, a, r, s'))
# s = random.uniform(0, buffer.total())
# idx, p, data = buffer.get(s)
```

### Key Takeaways
*   PER speeds up learning by focusing on hard examples.
*   Requires $O(\log N)$ data structure (SumTree).
*   Requires Importance Sampling to correct bias.
