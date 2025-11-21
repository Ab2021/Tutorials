# Day 25 Deep Dive: TD3 and SAC

## 1. DDPG's Limitations
DDPG suffers from:
*   **Overestimation Bias:** Like DQN, the critic overestimates Q-values.
*   **High Variance:** Sensitive to hyperparameters.
*   **Target Policy Smoothing:** The deterministic target can lead to sharp Q-value estimates.

## 2. TD3: Twin Delayed DDPG
TD3 (Fujimoto et al., 2018) improves DDPG with three tricks:
*   **Clipped Double Q-Learning:** Use two critics, take the minimum for the target.
    $$ y = r + \gamma \min_{i=1,2} Q'_i(s', \mu'(s')) $$
*   **Delayed Policy Updates:** Update the actor less frequently than the critic (e.g., every 2 critic updates).
*   **Target Policy Smoothing:** Add noise to the target action.
    $$ a' = \mu'(s') + \text{clip}(\epsilon, -c, c) $$
    This smooths out the Q-value landscape.

## 3. SAC: Soft Actor-Critic
SAC (Haarnoja et al., 2018) is the state-of-the-art for continuous control.
**Key Idea:** Maximize return AND entropy.
$$ J(\pi) = \sum \mathbb{E}[r_t + \alpha H(\pi(\cdot|s_t))] $$
*   $\alpha$ is the temperature hyperparameter (controls exploration).
*   SAC learns a **stochastic** policy (unlike DDPG's deterministic policy).
*   Uses **automatic temperature tuning** to adjust $\alpha$ dynamically.
*   More stable and robust than DDPG/TD3.

## 4. Comparison: DDPG vs. TD3 vs. SAC
| Feature | DDPG | TD3 | SAC |
|---------|------|-----|-----|
| Policy Type | Deterministic | Deterministic | Stochastic |
| Exploration | OU Noise | Gaussian Noise | Entropy Maximization |
| Stability | Moderate | High | Very High |
| Sample Efficiency | Good | Better | Best |
| Complexity | Low | Medium | High |
| Use Case | Research | Production | State-of-Art |

## 5. When to Use What?
*   **DDPG:** Simple baseline, educational purposes.
*   **TD3:** Production robotics, when deterministic policies are acceptable.
*   **SAC:** State-of-the-art, when sample efficiency and stability are critical.
