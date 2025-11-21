# Day 31 Interview Questions: Model-Based RL

## Q1: What is the difference between Model-Free and Model-Based RL?
**Answer:**
*   **Model-Free (DQN, PPO):** Learns the policy or value function directly from experience. Sample-inefficient but robust (no model bias).
*   **Model-Based (Dyna-Q, Dreamer):** Learns a model of the environment dynamics $T(s'|s, a)$ and $R(s, a)$, then uses it for planning or imagination. Sample-efficient but can suffer from model errors.

## Q2: How does Dyna-Q work?
**Answer:**
Dyna-Q integrates three components:
1. **Direct RL:** Update Q-values from real experience using TD learning.
2. **Model Learning:** Store or learn a model $T(s, a) \rightarrow (r, s')$ from real transitions.
3. **Planning:** Sample random previously visited $(s, a)$, simulate $(r, s')$ using the model, update Q-values with simulated experience.

This allows many Q-value updates per real environment step, improving sample efficiency.

## Q3: What are the main challenges in Model-Based RL?
**Answer:**
1. **Model Error:** If the model is inaccurate, planning will be suboptimal. Errors compound over long rollouts.
2. **High-Dimensional Observations:** Modeling pixel-level dynamics is difficult. Solution: Learn latent models.
3. **Exploration:** Need diverse data to learn an accurate model.
4. **Computational Cost:** Planning or simulation can be expensive.

## Q4: What is a World Model?
**Answer:**
**World Models** (Ha & Schmidhuber, 2018) learns a compressed representation of the environment:
*   **VAE (V):** Encodes observations into a compact latent space.
*   **RNN (M):** Predicts next latent state from current latent and action.
*   **Controller (C):** Small policy trained entirely in the latent "dream" world.

The policy never sees real pixels during training, only the learned latent representation.

## Q5: How does Dreamer improve upon World Models?
**Answer:**
**Dreamer** uses:
*   **RSSM (Recurrent State-Space Model):** More powerful latent dynamics model.
*   **Actor-Critic in Imagination:** Trains policy and value function on long imagined rollouts (15-50 steps).
*   **Multi-Step Learning:** Uses imagined trajectories for better credit assignment.

Dreamer v3 achieves state-of-the-art performance on Atari with 100x less data than Rainbow DQN.

## Q6: What is Model Predictive Control (MPC)?
**Answer:**
MPC uses the model for **online planning** instead of learning a policy:
1. At each timestep, simulate many action sequences using the model.
2. Choose the sequence with the highest predicted return.
3. Execute only the first action.
4. Replan at the next step (closed-loop control).

**Advantages:** Handles model errors via replanning. **Disadvantages:** Computationally expensive.

## Q7: How do we handle model errors?
**Answer:**
*   **Ensemble Models:** Train multiple models, use their disagreement as uncertainty. Plan conservatively when uncertainty is high.
*   **Pessimism:** Use the worst-case prediction from the ensemble.
*   **Model-Based Value Expansion (MVE):** Use the model for short rollouts (k-steps), then use a learned value function for long-term estimation.
*   **Online Adaptation:** Continuously update the model with new data.
