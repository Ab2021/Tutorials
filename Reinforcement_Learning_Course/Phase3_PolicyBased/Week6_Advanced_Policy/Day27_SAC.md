# Day 27: Soft Actor-Critic (SAC)

## 1. Maximum Entropy RL
Standard RL maximizes expected return: $J = \mathbb{E}[\sum r_t]$.
**SAC** also maximizes the policy's entropy:
$$ J = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t (r_t + \alpha H(\pi(\cdot|s_t)))\right] $$
where $H(\pi) = -\sum \pi(a|s) \log \pi(a|s)$ is the entropy.
*   **High Entropy:** Policy is more random (exploration).
*   **Low Entropy:** Policy is more deterministic (exploitation).
*   $\alpha$ controls the temperature (tradeoff).

## 2. Why Entropy Maximization?
*   **Exploration:** The agent is incentivized to try diverse actions.
*   **Robustness:** The policy learns to succeed in multiple ways (not overfitting to a single solution).
*   **Sample Efficiency:** Better exploration → faster learning.

## 3. SAC Architecture
SAC uses:
*   **Stochastic Actor:** Outputs a Gaussian distribution $\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$.
*   **Two Soft Q-Critics:** $Q_{\phi_1}(s, a)$ and $Q_{\phi_2}(s, a)$ (like TD3).
*   **Automatic Temperature Tuning:** Learns $\alpha$ to match a target entropy.

## 4. Reparameterization Trick
To backpropagate through the stochastic policy, SAC uses the **reparameterization trick**:
$$ a_\theta(s, \epsilon) = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$
*   The noise $\epsilon$ is external, so we can differentiate through $\mu$ and $\sigma$.
*   This allows us to optimize the policy using gradient descent.

## 5. Code Sketch: SAC Update
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class  GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
    
    def forward(self, state):
        x = self.fc(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)  # Numerical stability
        std = log_std.exp()
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)  # Squash to [-1, 1]
        
        # Compute log prob with squashing correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

def sac_update(actor, critic1, critic2, critic1_target, critic2_target, 
               actor_optimizer, critic_optimizer, alpha_optimizer, replay_buffer, alpha):
    states, actions, rewards, next_states, dones = replay_buffer.sample(256)
    
    # Update Critics (Soft Q-Function)
    with torch.no_grad():
        next_actions, next_log_probs = actor.sample(next_states)
        target_q1 = critic1_target(next_states, next_actions)
        target_q2 = critic2_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
        target_q = rewards + 0.99 * (1 - dones) * target_q
    
    current_q1 = critic1(states, actions)
    current_q2 = critic2(states, actions)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
    
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    # Update Actor
    new_actions, log_probs = actor.sample(states)
    q1_new = critic1(states, new_actions)
    q2_new = critic2(states, new_actions)
    q_new = torch.min(q1_new, q2_new)
    
    actor_loss = (alpha * log_probs - q_new).mean()
    
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    # Update Temperature (Alpha)
    target_entropy = -action_dim  # Heuristic: -dim(A)
    alpha_loss = -(alpha * (log_probs + target_entropy).detach()).mean()
    
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()
    
    # Soft update target networks
    tau = 0.005
    for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

### Key Takeaways
*   SAC is state-of-the-art for continuous control.
*   Entropy maximization provides automatic exploration.
*   More sample-efficient than TD3 or DDPG.
