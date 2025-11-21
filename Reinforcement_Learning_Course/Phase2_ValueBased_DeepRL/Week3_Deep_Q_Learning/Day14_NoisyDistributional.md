# Day 14: Noisy Nets & Distributional RL

## 1. Noisy Nets for Exploration
We previously discussed $\epsilon$-greedy exploration.
**Noisy Nets** replace the standard linear layers ($y = Wx + b$) with noisy linear layers:
$$ y = (\mu_W + \sigma_W \odot \epsilon_W) x + (\mu_b + \sigma_b \odot \epsilon_b) $$
*   $\mu$: Learnable mean parameters.
*   $\sigma$: Learnable standard deviation parameters.
*   $\epsilon$: Random noise sampled from $N(0, 1)$.
*   **Effect:** The network learns *how much* to explore. If $\sigma \to 0$, it becomes deterministic. If $\sigma$ is high, it explores.

## 2. Distributional RL (C51)
Standard DQN learns the **expected value** $Q(s, a) = \mathbb{E}[R]$.
But the return is a random variable $Z(s, a)$. Two states might have the same mean but different variances (Risk).
**C51 (Categorical DQN)** learns the full distribution of returns.
*   **Support:** Divide the range of returns $[V_{min}, V_{max}]$ into 51 atoms (bins).
*   **Output:** The network outputs a softmax probability distribution $p_i(s, a)$ over these atoms.
*   **Loss:** Kullback-Leibler (KL) Divergence between the predicted distribution and the target distribution.

## 3. Why Distributional?
1.  **Risk Sensitivity:** Knowing the variance allows us to distinguish between "safe" and "risky" actions.
2.  **Better Learning:** Learning the distribution provides a richer training signal (auxiliary task) than just the mean. It stabilizes learning.

## 4. Code Example: Noisy Linear Layer
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
    def reset_noise(self):
        epsilon_in = torch.randn(self.in_features)
        epsilon_out = torch.randn(self.out_features)
        # Factorized Gaussian Noise (for efficiency)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

# Usage in DQN
class NoisyDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.noisy1 = NoisyLinear(128, 128)
        self.noisy2 = NoisyLinear(128, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.noisy1(x))
        return self.noisy2(x)
```

### Key Takeaways
*   Noisy Nets automate exploration tuning.
*   Distributional RL learns the full shape of returns, not just the mean.
