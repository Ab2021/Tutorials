# Day 21: REINFORCE (Monte Carlo Policy Gradient)

## 1. The Policy Gradient Idea
In Value-Based RL, we learned $Q(s, a)$ and picked $a = \arg\max Q$.
In **Policy-Based RL**, we parameterize the policy $\pi_\theta(a|s)$ directly (e.g., a neural network that outputs action probabilities).
We want to maximize the expected return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$.
We can use **Gradient Ascent**: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$.

## 2. The Log-Derivative Trick
How do we differentiate an expectation?
$$ \nabla_\theta \mathbb{E}[R] = \mathbb{E} [\nabla_\theta \log \pi_\theta(a|s) \cdot R(\tau)] $$
*   $\nabla_\theta \log \pi_\theta(a|s)$ is the **Score Function**. It tells us how to change $\theta$ to make action $a$ more likely.
*   $R(\tau)$ is the **Return**.
*   **Intuition:** If the return $R$ is high, push $\theta$ in the direction that increases the probability of the actions we took. If $R$ is low, push in the opposite direction.

## 3. REINFORCE Algorithm
REINFORCE is the Monte Carlo version of Policy Gradient.
1.  Run an episode to generate trajectory $\tau = (s_0, a_0, r_1, ..., s_T)$.
2.  Calculate the return $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}$ for each step.
3.  Update $\theta$:
    $$ \theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t $$

## 4. Code Example: REINFORCE in PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1) # Output probabilities
        )
        
    def forward(self, x):
        return self.fc(x)

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        
        # Sample action from distribution
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        # Save log_prob for backprop
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
        
    def update(self):
        R = 0
        returns = []
        # Calculate discounted returns (backwards)
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        # Normalize returns (Baselines reduce variance)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R) # Negative because Gradient Descent
            
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Reset
        self.saved_log_probs = []
        self.rewards = []
```

### Key Takeaways
*   Learns stochastic policies.
*   Can handle continuous action spaces (using Gaussian output).
*   **High Variance:** Because it uses the full Monte Carlo return $G_t$.
