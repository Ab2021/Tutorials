# Day 30 Deep Dive: Debugging Policy Gradients

## 1. Common Failure Modes
*   **Vanishing Gradients:** Policy becomes too deterministic (entropy → 0). Add entropy regularization.
*   **Exploding Gradients:** Large probability ratios. Use gradient clipping and PPO's clip.
*   **Reward Scaling:** If rewards vary widely, normalize them or use value normalization.
*   **Learning Rate:** Too high → instability. Too low → slow learning. Use adaptive schedules.

## 2. Hyperparameter Tuning Guide
| Hyperparameter | Typical Range | What It Controls |
|----------------|---------------|------------------|
| Learning Rate | 1e-4 to 3e-4 | Update step size |
| Discount (γ) | 0.95 to 0.999 | Future vs. immediate reward |
| GAE Lambda (λ) | 0.90 to 0.99 | Bias-variance tradeoff |
| Clip Epsilon (ε) | 0.1 to 0.3 | Policy update constraint |
| Entropy Coefficient | 0.001 to 0.01 | Exploration |
| Num Epochs | 3 to 10 | Data reuse |
| Batch Size | 64 to 256 | Gradient stability |

## 3. Visualization Tips
*   **Learning Curves:** Plot episode reward, value loss, policy loss, entropy over time.
*   **KL Divergence:** Monitor $D_{KL}(\pi_{old} || \pi_{new})$ to ensure stable updates.
*   **Explained Variance:** Measure how well the value function predicts returns.
*   **Gradient Norms:** Track gradient magnitudes to detect exploding/vanishing gradients.

## 4. Advanced Tricks
*   **Value Function Clipping:** Clip value updates like policy updates.
*   **Observation Normalization:** Normalize states using running mean/std.
*   **Reward Clipping:** Clip rewards to [-1, 1] (for Atari games).
*   **Learning Rate Annealing:** Linearly decay LR to 0 over training.
