# Day 33 Deep Dive: Sampled MuZero and EfficientZero

## 1. Sampled MuZero
Standard MuZero plans over all actions at each node (expensive for large action spaces).
**Sampled MuZero:** Only simulates a **subset** of promising actions.
*   Use the policy network to sample likely actions.
*   Reduces computational cost significantly.

## 2. EfficientZero: Sample-Efficient MuZero
**EfficientZero** (Ye et al., 2021) achieves human-level Atari with only **100k frames** (vs. 50M for Rainbow).

**Improvements:**
*   **Self-Supervised Consistency Loss:** Predict future latent states.
*   **Off-Policy Correction:** Reuse old data more effectively.
*   **Prioritized Experience Replay:** Focus on important transitions.

## 3. Gumbel MuZero
Uses **Gumbel Top-k trick** for action selection:
*   More principled than sampling from the policy.
*   Improves exploration and sample efficiency.

## 4. Applications Beyond Games
*   **Video Compression (YouTube):** MuZero optimizes bitrate allocation.
*   **Robotics:** Planning for manipulation tasks.
*   **Recommender Systems:** Sequential recommendation as RL.
