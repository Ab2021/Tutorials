# Day 26: Twin Delayed DDPG (TD3)

## 1. The Three Improvements Over DDPG
TD3 (Fujimoto et al., 2018) addresses DDPG's instability with three key modifications:

### 1.1 Clipped Double Q-Learning
Use **two** independent critic networks $Q_{\phi_1}$ and $Q_{\phi_2}$.
For the target, take the **minimum**:
$$ y = r + \gamma \min_{i=1,2} Q'_{\phi_i}(s', a') $$
*   This reduces **overestimation bias** (like Double DQN).
*   The minimum provides a conservative estimate.

### 1.2 Delayed Policy Updates
Update the actor **less frequently** than the critics.
*   Update critics every step.
*   Update actor every $d$ steps (typically $d=2$).
*   **Why?** The actor needs stable Q-values. Updating too frequently with noisy Q-estimates leads to poor policies.

### 1.3 Target Policy Smoothing
Add **noise** to the target action to smooth out Q-value estimates:
$$ a' = \mu'_\theta(s') + \epsilon, \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c) $$
*   This prevents the policy from exploiting errors in the Q-function.
*   Makes Q-values more robust to small perturbations.

## 2. Code Example: TD3 Update
```python
import torch
import torch.nn as nn

def td3_update(actor, critic1, critic2, actor_target, critic1_target, critic2_target,
               actor_optimizer, critic_optimizer, replay_buffer, 
               step, policy_freq=2, policy_noise=0.2, noise_clip=0.5):
    # Sample batch
    states, actions, rewards, next_states, dones = replay_buffer.sample(256)
    
    # Target policy smoothing
    noise = torch.randn_like(actions) * policy_noise
    noise = noise.clamp(-noise_clip, noise_clip)
    next_actions = (actor_target(next_states) + noise).clamp(-max_action, max_action)
    
    # Compute target Q using minimum of two critics
    target_q1 = critic1_target(next_states, next_actions)
    target_q2 = critic2_target(next_states, next_actions)
    target_q = torch.min(target_q1, target_q2)
    target_q = rewards + (1 - dones) * 0.99 * target_q
    
    # Update both critics
    current_q1 = critic1(states, actions)
    current_q2 = critic2(states, actions)
    critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
    
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    # Delayed policy update
    if step % policy_freq == 0:
        # Update actor
        actor_loss = -critic1(states, actor(states)).mean()
        
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        # Soft update target networks
        tau = 0.005
        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

### Key Takeaways
*   TD3 is more stable and performs better than DDPG.
*   The three tricks are simple but effective.
*   State-of-the-art for deterministic continuous control.
