# Day 25: Deep Deterministic Policy Gradient (DDPG)

## 1. Continuous Action Spaces
Most policy gradients (REINFORCE, A2C, PPO) work with **discrete** actions using softmax.
For **continuous** actions (e.g., robot joint torques, car steering angles), we need a different approach.
**DDPG** combines:
*   **Deterministic Policy:** $a = \mu_\theta(s)$ (outputs a single action, not a distribution).
*   **Q-Learning:** Uses a critic $Q_\phi(s, a)$ to guide the policy.

## 2. Actor-Critic for Continuous Control
*   **Actor:** Learns a deterministic policy $\mu_\theta(s): \mathcal{S} \rightarrow \mathcal{A}$.
*   **Critic:** Learns the Q-function $Q_\phi(s, a)$.
*   **Policy Gradient:** 
    $$ \nabla_\theta J \approx \mathbb{E}[\nabla_a Q(s, a)|_{a=\mu(s)} \nabla_\theta \mu_\theta(s)] $$
    *   This is the **Deterministic Policy Gradient (DPG)** theorem.
    *   We differentiate $Q$ w.r.t. actions, then chain-rule through the policy.

## 3. Key Components from DQN
DDPG is "DQN for continuous control":
*   **Replay Buffer:** Store $(s, a, r, s')$ tuples, sample mini-batches.
*   **Target Networks:** Slow-moving targets $\mu'$ and $Q'$ for stability.
    $$ y = r + \gamma Q'(s', \mu'(s')) $$
*   **Soft Updates:** 
    $$ \theta' \leftarrow \tau \theta + (1 - \tau) \theta' $$
    where $\tau \approx 0.001$ (much slower than periodic hard updates).

## 4. Exploration in Deterministic Policies
A deterministic policy has no inherent exploration.
**Solution:** Add noise to actions during training.
$$ a = \mu_\theta(s) + \mathcal{N}(0, \sigma) $$
*   Often use **Ornstein-Uhlenbeck (OU) noise** for temporally correlated exploration.
*   Decay $\sigma$ over time.

## 5. Code Example: DDPG Update
```python
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, action_dim), nn.Tanh()
        )
        self.max_action = max_action
    
    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, 1)
        )
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

def ddpg_update(actor, critic, actor_target, critic_target, 
                actor_optimizer, critic_optimizer, replay_buffer, tau=0.001):
    # Sample batch
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size=64)
    
    # Update Critic
    with torch.no_grad():
        next_actions = actor_target(next_states)
        target_q = rewards + 0.99 * (1 - dones) * critic_target(next_states, next_actions)
    
    current_q = critic(states, actions)
    critic_loss = nn.MSELoss()(current_q, target_q)
    
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    # Update Actor
    actor_loss = -critic(states, actor(states)).mean()
    
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    # Soft update target networks
    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

### Key Takeaways
*   DDPG is off-policy and sample-efficient.
*   Foundation for modern continuous control (TD3, SAC).
*   Sensitive to hyperparameters (learning rates, noise).
