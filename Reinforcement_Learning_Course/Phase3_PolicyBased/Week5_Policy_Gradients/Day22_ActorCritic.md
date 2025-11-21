# Day 22: Actor-Critic Methods (A2C)

## 1. The Best of Both Worlds
*   **Value-Based (DQN):** Low variance, but biased. Hard to handle continuous actions.
*   **Policy-Based (REINFORCE):** Unbiased, handles continuous actions, but High Variance.
*   **Actor-Critic:** Combines them.
    *   **Actor ($\pi_\theta$):** Learns the policy (controls the agent).
    *   **Critic ($V_\phi$):** Learns the value function (evaluates the agent).

## 2. The Update Rule
Instead of using the Monte Carlo return $G_t$ (high variance), we use the Critic to estimate the return.
$$ \nabla J \approx \nabla \log \pi_\theta(a|s) \cdot A(s, a) $$
where $A(s, a)$ is the **Advantage**:
$$ A(s, a) = r + \gamma V_\phi(s') - V_\phi(s) $$
*   This is the **TD Error**!
*   The Critic reduces variance by bootstrapping.
*   The Actor learns from the Critic's feedback, not just the raw reward.

## 3. A2C (Advantage Actor-Critic)
A2C is the synchronous, deterministic version of A3C (Asynchronous Advantage Actor-Critic).
*   **Architecture:** A single neural network with two heads:
    1.  **Policy Head:** Outputs action probabilities (softmax).
    2.  **Value Head:** Outputs scalar value $V(s)$.
*   **Parallel Environments:** Run $N$ environments in parallel to gather diverse data and break correlations (replaces Replay Buffer).

## 4. Code Example: A2C Update
```python
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.common = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.common(x)
        probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return probs, value

def update_a2c(agent, optimizer, transitions, gamma=0.99):
    # transitions: list of (s, a, r, s', done)
    states, actions, rewards, next_states, dones = zip(*transitions)
    
    # Convert to tensors
    states = torch.stack(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    masks = 1 - torch.tensor(dones, dtype=torch.float32)
    
    # Get Actor and Critic outputs
    probs, values = agent(states)
    _, next_values = agent(torch.stack(next_states))
    
    # Calculate Returns (TD Target)
    # Target = r + gamma * V(s')
    targets = rewards + gamma * next_values.squeeze() * masks
    
    # Calculate Advantage
    # Adv = Target - V(s)
    advantages = targets - values.squeeze()
    
    # 1. Critic Loss (MSE)
    critic_loss = advantages.pow(2).mean()
    
    # 2. Actor Loss (Policy Gradient)
    # We detach advantage to stop gradients flowing back from Actor to Critic
    dist = torch.distributions.Categorical(probs)
    log_probs = dist.log_prob(actions)
    actor_loss = -(log_probs * advantages.detach()).mean()
    
    # 3. Entropy Loss (Bonus for exploration)
    entropy_loss = -dist.entropy().mean()
    
    total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### Key Takeaways
*   Actor-Critic is the standard for modern RL.
*   A2C uses parallel workers to stabilize training.
*   Entropy regularization prevents premature convergence.
