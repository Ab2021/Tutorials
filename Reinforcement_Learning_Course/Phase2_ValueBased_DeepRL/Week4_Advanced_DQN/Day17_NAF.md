# Day 17: Continuous Control with Q-Learning (NAF)

## 1. The Problem with Continuous Actions
DQN works great for Atari (Discrete Actions: Left, Right, Jump).
But for Robotics (Joint Torques: $-1.0$ to $+1.0$), DQN fails.
*   **Reason:** The target $y = r + \gamma \max_a Q(s', a)$ requires finding the global maximum of $Q(s', a)$ with respect to $a$.
*   If $a$ is continuous, this is a non-convex optimization problem at every step! (Too slow).

## 2. Normalized Advantage Functions (NAF)
NAF (Gu et al., 2016) is a clever trick to apply Q-Learning to continuous spaces.
It restricts the Q-function to a specific **quadratic form** so that the maximum is easy to find analytically.
$$ Q(s, a) = V(s) + A(s, a) $$
$$ A(s, a) = -\frac{1}{2} (a - \mu(s))^T P(s) (a - \mu(s)) $$
*   $V(s)$: Value function (Scalar).
*   $\mu(s)$: The action that maximizes Q (Vector).
*   $P(s)$: A positive-definite matrix (State-dependent covariance).

## 3. Why it works
Since $P(s)$ is positive-definite, the term $(a - \mu(s))^T P(s) (a - \mu(s))$ is always $\ge 0$.
Therefore, $A(s, a) \le 0$.
The maximum occurs when $a = \mu(s)$, giving $A(s, \mu(s)) = 0$.
$$ \max_a Q(s, a) = V(s) $$
**Result:** We can perform the max operation instantly! The optimal action is just the output of the $\mu(s)$ network.

## 4. Code Example: NAF Network
```python
import torch
import torch.nn as nn

class NAF(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        
        self.base = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        
        # Output 1: Value V(s)
        self.v_head = nn.Linear(128, 1)
        
        # Output 2: Action Mean mu(s)
        self.mu_head = nn.Linear(128, action_dim)
        
        # Output 3: Matrix L (Cholesky decomposition of P)
        # P = L * L^T
        self.l_head = nn.Linear(128, action_dim * (action_dim + 1) // 2)
        
    def forward(self, state, action=None):
        feat = self.base(state)
        V = self.v_head(feat)
        mu = torch.tanh(self.mu_head(feat)) # Bound actions
        
        # Construct L matrix (Lower Triangular)
        L_entries = self.l_head(feat)
        L = torch.zeros((state.size(0), self.action_dim, self.action_dim))
        # Fill L (omitted for brevity, involves indexing)
        # Diagonal of L must be positive (exponentiate)
        
        P = torch.bmm(L, L.transpose(1, 2))
        
        if action is None:
            return mu # Return greedy action
            
        # Calculate Q
        diff = (action - mu).unsqueeze(2) # (B, A, 1)
        # A = -0.5 * diff^T * P * diff
        adv = -0.5 * torch.bmm(torch.bmm(diff.transpose(1, 2), P), diff).squeeze(2)
        
        Q = V + adv
        return Q, V, mu
```

### Key Takeaways
*   NAF allows Q-Learning in continuous spaces.
*   It assumes the Q-function is quadratic (unimodal).
*   Precursor to more general Actor-Critic methods (DDPG, SAC).
