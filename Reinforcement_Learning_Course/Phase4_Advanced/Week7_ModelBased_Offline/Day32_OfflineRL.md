# Day 32: Offline Reinforcement Learning

## 1. The Offline RL Problem
**Standard RL:** Agent interacts with the environment, collects data, learns.
**Offline RL (Batch RL):** Learn from a **fixed dataset** $\mathcal{D}$ collected by some behavior policy $\pi_\beta$. No further interaction.

**Motivation:**
*   Real-world scenarios: medical treatment, autonomous driving (can't  explore dangerously).
*   Leverage existing datasets (logs, demonstrations).

## 2. The Challenge: Distribution Shift
The learned policy $\pi$ may visit states not in $\mathcal{D}$.
*   **Out-of-Distribution (OOD) Actions:** $Q(s, a)$ for unseen $(s, a)$ can be wildly overestimated.
*   **Extrapolation Error:** The value function extrapolates poorly to OOD regions.

## 3. Conservative Q-Learning (CQL)
**CQL** (Kumar et al., 2020) addresses overestimation by learning a **conservative** Q-function.
Add a penalty to the Q-learning objective:
$$ \min_Q \mathbb{E}_{s \sim \mathcal{D}} [\log \sum_a \exp(Q(s, a))] - \mathbb{E}_{(s,a) \sim \mathcal{D}} [Q(s, a)] + \text{Bellman Error} $$
*   The first term encourages Q-values for all actions to be low.
*   The second term encourages Q-values for actions in the dataset to be accurate.
*   This creates a **lower bound** on the true Q-value.

## 4. Behavioral Cloning (BC)
The simplest offline approach: **supervised learning** on the dataset.
$$ \min_\pi \mathbb{E}_{(s, a) \sim \mathcal{D}} [-\log \pi(a|s)] $$
*   Imitates the behavior policy.
*   **Problem:** Limited by the quality of the dataset. Cannot improve beyond $\pi_\beta$.

## 5. Implicit Q-Learning (IQL)
**IQL** (Kostrikov et al., 2021) learns a policy that improves upon the dataset **without** querying OOD actions.
*   Learns an **expectile regression** value function (not max).
*   Extracts a policy via **advantage-weighted regression**.
*   State-of-the-art offline RL method.

## 6. Code Sketch: CQL Loss
```python
import torch
import torch.nn as nn

def cql_loss(q_network, policy, batch, alpha=1.0):
    states, actions, rewards, next_states, dones = batch
    
    # Standard Q-learning target
    with torch.no_grad():
        next_actions = policy(next_states)
        target_q = rewards + 0.99 * (1 - dones) * q_network(next_states, next_actions)
    
    # Bellman error
    current_q = q_network(states, actions)
    bellman_error = F.mse_loss(current_q, target_q)
    
    # CQL penalty: log-sum-exp of Q-values
    # Sample random actions for current states
    random_actions = torch.rand_like(actions)
    q_random = q_network(states, random_actions)
    
    # Dataset actions
    q_data = q_network(states, actions)
    
    # CQL loss: encourage low Q for random actions, accurate Q for data actions
    cql_penalty = torch.logsumexp(q_random, dim=1).mean() - q_data.mean()
    
    total_loss = bellman_error + alpha * cql_penalty
    return total_loss
```

### Key Takeaways
*   Offline RL learns from fixed datasets without environment interaction.
*   **Challenge:** Distribution shift and OOD extrapolation errors.
*   **CQL:** Learns conservative Q-functions to avoid overestimation.
*   **IQL:** Avoids querying OOD actions entirely using expectile regression.
