# Day 31 Deep Dive: World Models and Dreamer

## 1. World Models Architecture
**World Models** (Ha & Schmidhuber, 2018) learns a compressed representation of the environment.

### Components:
1.  **Vision (V):** VAE that encodes observations $o_t$ into latent $z_t$.
2.  **Memory (M):** RNN/LSTM that predicts $z_{t+1}$ from $z_t, a_t$.
3.  **Controller (C):** Small policy that maps $(z_t, h_t) \rightarrow a_t$.

**Training:**
*   Train V and M on collected data (unsupervised).
*   Train C entirely in the **dream** (latent space), not the real environment.

### Advantages:
*   Policy training is fast (no real environment interaction).
*   The model is compact (latent space is much smaller than pixel space).

## 2. Dreamer: State-of-the-Art MBRL
**Dreamer** (Hafner et al., 2020) improves upon World Models:
*   **Recurrent State-Space Model (RSSM):** Learns latent dynamics.
*   **Actor-Critic in Imagination:** Trains policy and value function entirely in the latent space.
*   **Multi-Step Imagination:** Uses long imagination rollouts for learning.

**Dreamer v3** (2023) achieves near-perfect scores on Atari with only 200k frames (vs. 50M for Rainbow DQN).

## 3. Model Predictive Control (MPC)
Instead of learning a policy, use the model for **online planning**:
1.  At each step, simulate multiple action sequences using the model.
2.  Choose the sequence with the highest predicted return.
3.  Execute the first action, replan at the next step.

**Example: MPC in Robotics:**
*   Plan trajectories for robotic manipulation.
*   Replan at each timestep to handle model errors.

## 4. Handling Model Errors
*   **Ensemble Models:** Train multiple models, use disagreement as uncertainty.
*   **Pessimism:** Use the worst-case model prediction (conservative planning).
*   **Model-Based Value Expansion (MVE):** Use the model for short rollouts, value function for long-term.
