# Day 23: Proximal Policy Optimization (PPO)

## 1. The Problem with Vanilla Policy Gradients
In standard Policy Gradients, we update $\theta$ using:
$$ \theta \leftarrow \theta + \alpha \nabla J(\theta) $$
*   **Problem:** If the learning rate is too large, a single bad update can destroy the policy.
*   We have no control over how much the policy changes between updates.
*   This makes training unstable and sample-inefficient.

## 2. Trust Region Idea
We want to constrain the policy update so that the new policy $\pi_{\theta_{new}}$ doesn't deviate too much from the old policy $\pi_{\theta_{old}}$.
We measure the difference using **KL Divergence**:
$$ D_{KL}(\pi_{\theta_{old}} || \pi_{\theta_{new}}) < \delta $$
This ensures the policy changes smoothly and doesn't take catastrophic steps.

## 3. PPO-Clip: The Practical Solution
PPO approximates the trust region constraint using a **clipped objective**.
Define the **probability ratio**:
$$ r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)} $$
The PPO loss is:
$$ L^{CLIP}(\theta) = \mathbb{E}[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)] $$
*   $\epsilon$ is typically 0.2.
*   If $r_t > 1 + \epsilon$, the new policy assigns much higher probability to the action. We clip it.
*   If $r_t < 1 - \epsilon$, the new policy assigns much lower probability. We clip it.
*   This prevents large policy updates.

## 4. Code Example: PPO Update
```python
import torch
import torch.nn as nn
import torch.optim as optim

def ppo_update(policy, optimizer, states, actions, log_probs_old, advantages, returns, clip_epsilon=0.2, epochs=10):
    """
    PPO update with clipped objective.
    
    Args:
        policy: Actor-Critic network
        optimizer: Optimizer
        states: Batch of states
        actions: Batch of actions taken
        log_probs_old: Log probabilities under old policy
        advantages: GAE advantages
        returns: Discounted returns (for critic)
        clip_epsilon: Clipping parameter (typically 0.2)
        epochs: Number of optimization epochs on same batch
    """
    for _ in range(epochs):
        # Get current policy outputs
        action_probs, values = policy(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Calculate ratio
        ratio = torch.exp(log_probs - log_probs_old)
        
        # PPO Clipped Objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic Loss (MSE)
        critic_loss = (returns - values.squeeze()).pow(2).mean()
        
        # Entropy Bonus
        entropy_loss = -entropy.mean()
        
        # Total Loss
        loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
        
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
```

### Key Takeaways
*   PPO is the most popular RL algorithm (used by OpenAI, DeepMind).
*   Simple to implement and stable.
*   Can reuse data for multiple epochs (more sample-efficient than A2C).
*   The clipping constraint prevents destructive updates.
