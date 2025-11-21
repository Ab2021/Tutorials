# Day 19: Inverse Reinforcement Learning (IRL)

## 1. The Problem: Where do rewards come from?
So far, we assumed the reward function $R(s, a)$ is given (e.g., +1 for winning).
In real life (e.g., driving), defining the reward is hard.
*   "Don't crash" (Easy).
*   "Drive smoothly" (Hard to quantify).
*   "Don't be aggressive" (Very hard).
**Inverse RL** aims to learn the reward function $R$ from a set of **Expert Demonstrations** $\tau_E = \{ (s_1, a_1), (s_2, a_2), ... \}$.

## 2. Behavioral Cloning (BC) vs. IRL
*   **Behavioral Cloning:** Supervised learning. Train $\pi(s) \to a$ to mimic the expert.
    *   *Problem:* Compounding errors. If the agent drifts slightly off-track, it enters states it has never seen and fails.
*   **IRL:** Learn the *intent* (Reward Function) of the expert, then use RL to optimize that reward.
    *   *Benefit:* Robust. If the agent drifts, it knows *why* it should go back (to maximize reward).

## 3. Maximum Entropy IRL
The problem is ill-posed: $R(s)=0$ explains any behavior (optimally).
We assume the expert acts **probabilistically** based on the value of the trajectory:
$$ P(\tau) \propto e^{R(\tau)} $$
We try to find a reward function $R_\theta$ such that the expected feature counts of the agent match the expert, while maximizing the entropy of the distribution (being as random as possible while matching constraints).

## 4. Code Example: Linear Reward Function
Assume $R(s) = w^T \phi(s)$, where $\phi(s)$ are features (e.g., speed, distance to lane).

```python
import numpy as np

def expert_feature_counts(demonstrations, feature_fn):
    # Calculate average feature counts of expert
    feature_sum = 0
    for traj in demonstrations:
        for state in traj:
            feature_sum += feature_fn(state)
    return feature_sum / len(demonstrations)

def max_ent_irl(expert_features, env, iterations=100):
    # Initialize weights randomly
    w = np.random.normal(size=expert_features.shape)
    
    for i in range(iterations):
        # 1. Compute Reward R = w * features
        # 2. Solve MDP with this R to get Policy pi
        # 3. Calculate expected feature counts of pi
        agent_features = calc_expected_counts(env, w)
        
        # 4. Update w to minimize difference (Gradient Ascent)
        # Gradient = Expert_Features - Agent_Features
        grad = expert_features - agent_features
        w += 0.01 * grad
        
    return w
```

### Key Takeaways
*   IRL infers the "Why" behind behavior.
*   Solves the reward engineering problem.
*   Computationally expensive (requires solving MDP in the inner loop).
