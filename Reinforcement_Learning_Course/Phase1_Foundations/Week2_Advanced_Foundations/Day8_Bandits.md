# Day 8: Multi-Armed Bandits

## 1. The Simplest RL Problem
The **Multi-Armed Bandit** problem is a special case of RL with only **one state**.
*   You have $k$ actions (arms).
*   Each action gives a reward drawn from a probability distribution.
*   Goal: Maximize total reward over a time horizon $T$.
*   Challenge: You don't know the reward distributions. You must **explore** to find the best arm and **exploit** to get rewards.

## 2. Action-Value Methods
We estimate the value of each action $a$:
$$ Q_t(a) \approx \frac{\text{Sum of rewards when } a \text{ taken}}{\text{Number of times } a \text{ taken}} $$
By the Law of Large Numbers, $Q_t(a) \to q_*(a)$ as $N_t(a) \to \infty$.

## 3. Exploration vs. Exploitation
*   **Greedy:** Always choose $A_t = \arg\max Q_t(a)$. (Maximizes immediate reward, but might get stuck on a sub-optimal arm).
*   **$\epsilon$-Greedy:**
    *   Prob $1-\epsilon$: Choose greedy action.
    *   Prob $\epsilon$: Choose random action.
    *   Ensures all arms are visited infinitely often (asymptotically optimal).

## 4. Code Example: 10-Armed Bandit Testbed
Comparing Greedy vs $\epsilon$-Greedy.

```python
import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k=10):
        self.k = k
        # True values of arms (unknown to agent)
        self.q_true = np.random.randn(k) 
        self.q_est = np.zeros(k)
        self.action_count = np.zeros(k)
        
    def step(self, action):
        # Reward is Gaussian around true value
        return np.random.randn() + self.q_true[action]
    
    def act(self, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.q_est)
            
    def update(self, action, reward):
        self.action_count[action] += 1
        # Incremental average update
        self.q_est[action] += (1/self.action_count[action]) * (reward - self.q_est[action])

def run_experiment(epsilon, steps=1000):
    bandit = Bandit()
    rewards = []
    for _ in range(steps):
        action = bandit.act(epsilon)
        reward = bandit.step(action)
        bandit.update(action, reward)
        rewards.append(reward)
    return rewards

# Run
rewards_greedy = run_experiment(epsilon=0.0)
rewards_eps01 = run_experiment(epsilon=0.1)

print(f"Avg Reward (Greedy): {np.mean(rewards_greedy):.2f}")
print(f"Avg Reward (Eps=0.1): {np.mean(rewards_eps01):.2f}")
```

### Key Takeaways
*   Bandits isolate the exploration-exploitation trade-off.
*   Greedy usually fails to find the optimal arm.
*   $\epsilon$-Greedy is a simple, effective baseline.
