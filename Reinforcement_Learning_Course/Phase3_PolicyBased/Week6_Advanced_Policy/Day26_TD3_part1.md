# Day 26 Deep Dive: Ablation Studies

## 1. What Makes TD3 Work?
Fujimoto et al. conducted thorough ablation studies to understand each component's contribution.

### Experiment: Removing Each Component
| Configuration | Mujoco Score (Avg.) |
|---------------|---------------------|
| Full TD3 | **3500** |
| TD3 w/o Clipped Double-Q | 2800 |
| TD3 w/o Delayed Policy | 2900 |
| TD3 w/o Target Smoothing | 3100 |
| DDPG (Baseline) | 2500 |

**Key Findings:**
*   All three components contribute significantly.
*   **Clipped Double-Q** has the largest impact (addresses overestimation).
*   **Delayed Policy Updates** prevents the actor from overfitting to noisy Q-estimates.
*   **Target Smoothing** makes Q-values more robust.

## 2. Hyperparameter Sensitivity
TD3 is **less sensitive** to hyperparameters than DDPG:
*   **Policy Noise ($\sigma$):** 0.1-0.3 works well.
*   **Noise Clip ($c$):** 0.3-0.5.
*   **Policy Update Frequency ($d$):** 2-3.
*   **Learning Rate:** 3e-4 for both actor and critics.
*   **Discount ($\gamma$):** 0.99.

## 3. When TD3 Fails
*   **Sparse Rewards:** TD3 (like all off-policy methods) struggles with very sparse rewards. Consider reward shaping.
*   **High-Dimensional Action Spaces:** Performance degrades as action dimensionality increases.
*   **Non-Stationary Environments:** TD3 assumes the environment is stationary. Rapidly changing dynamics can cause issues.

## 4. Comparison with SAC
*   **TD3:** Deterministic policy, simpler, faster.
*   **SAC:** Stochastic policy, entropy maximization, more sample-efficient, but more complex.
*   **When to use TD3:** When you need a deterministic policy or want simplicity.
*   **When to use SAC:** When sample efficiency and exploration are critical.
