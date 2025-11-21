# Day 19 Deep Dive: Generative Adversarial Imitation Learning (GAIL)

## 1. The Cost of IRL
Standard MaxEnt IRL requires solving the MDP (finding the optimal policy for the current reward) in the inner loop of the optimization.
This is extremely slow. It works for GridWorlds, but not for Mujoco/Atari.

## 2. GAIL: Merging GANs and IRL
Ho & Ermon (2016) showed that we can skip the explicit reward learning step.
We can frame Imitation Learning as a **Minimax Game** similar to GANs (Generative Adversarial Networks).
*   **Generator ($\pi_\theta$):** The policy tries to generate trajectories that look like the expert's.
*   **Discriminator ($D_\phi$):** Tries to distinguish between expert state-action pairs $(s, a)_E$ and agent state-action pairs $(s, a)_\pi$.

## 3. The Algorithm
1.  Sample expert data $\tau_E$.
2.  Sample agent data $\tau_\pi$ by running the current policy.
3.  **Train Discriminator:** Minimize cross-entropy loss to classify Expert vs. Agent.
    $$ L_D = - \mathbb{E}_E [\log D(s, a)] - \mathbb{E}_\pi [\log (1 - D(s, a))] $$
4.  **Train Policy (Generator):** Use TRPO/PPO to maximize the "reward" given by the discriminator.
    $$ R(s, a) = - \log (1 - D(s, a)) $$
    *   If $D$ is confused ($D \approx 0.5$), reward is high.
    *   If $D$ knows it's the agent ($D \approx 0$), reward is low.

## 4. Why GAIL is powerful
*   It scales to large environments (continuous control).
*   It learns a policy directly from data without needing to manually define a reward function.
*   It avoids the compounding error problem of Behavioral Cloning.
