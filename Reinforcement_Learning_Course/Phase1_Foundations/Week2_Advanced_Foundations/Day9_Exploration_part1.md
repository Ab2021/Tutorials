# Day 9 Deep Dive: Advanced Exploration

## 1. Count-Based Exploration
In tabular settings, we can simply count visits $N(s)$.
$$ R_{int}(s) = \frac{\beta}{\sqrt{N(s)}} $$
This gives high reward to rarely visited states.
**Challenge:** In continuous/large spaces, we never visit the exact same state twice. $N(s)$ is always 0 or 1.
**Solution:** Pseudo-counts (using density models) or hashing (SimHash).

## 2. Random Network Distillation (RND)
A popular method for continuous spaces (used in Montezuma's Revenge).
*   **Target Network ($T$):** Fixed, randomly initialized neural network. Maps $s \to \mathbb{R}^k$.
*   **Predictor Network ($\hat{T}$):** Trained to predict the output of $T$.
*   **Intrinsic Reward:** $r_{int} = ||T(s) - \hat{T}(s)||^2$.
*   **Logic:**
    *   If $s$ is familiar, $\hat{T}$ has trained on it and predicts $T(s)$ well. Low reward.
    *   If $s$ is novel, $\hat{T}$ has high error. High reward.
*   **Benefit:** No need to model the complex environment dynamics, just a random projection.

## 3. Noisy Nets
Instead of adding noise to the *action* ($\epsilon$-greedy), add noise to the *weights* of the network.
$$ y = (w + \sigma \odot \epsilon) x $$
*   $\epsilon \sim N(0, 1)$.
*   $\sigma$ is a learnable parameter.
*   **Effect:** The agent samples a specific "personality" (set of weights) for an entire episode. This leads to **consistent** exploration (e.g., "Try going left for this whole episode") rather than jittery exploration ("Left, Right, Left, Right").

## 4. Curiosity-Driven Learning
Predict the next state $S_{t+1}$ given $S_t, A_t$.
$$ r_{int} = ||\text{Model}(S_t, A_t) - S_{t+1}||^2 $$
*   **Problem:** The "TV Problem". If the agent finds a TV showing static noise, the next state is unpredictable. The error is high. The agent gets addicted to watching TV.
*   **Solution:** Predict only features relevant to the agent (Inverse Dynamics Features).
