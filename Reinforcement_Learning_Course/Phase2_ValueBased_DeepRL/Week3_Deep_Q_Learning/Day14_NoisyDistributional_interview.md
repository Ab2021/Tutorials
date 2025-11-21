# Day 14 Interview Questions: Noisy Nets & Distributional RL

## Q1: What is the main advantage of Noisy Nets over $\epsilon$-greedy exploration?
**Answer:**
Noisy Nets provide **state-dependent, temporally consistent** exploration.
*   **$\epsilon$-greedy:** Adds random noise to actions at every step. This causes "jittery" behavior (e.g., left, right, left, right) which cancels out progress.
*   **Noisy Nets:** Adds noise to the *weights*. For a fixed sample of noise (usually held constant for an episode or a few steps), the agent commits to a specific "personality" or strategy. This allows it to explore deep into the environment with consistent actions.

## Q2: Explain the basic idea of C51 (Categorical DQN).
**Answer:**
C51 models the *distribution* of returns $Z(s, a)$ instead of just the expected return $Q(s, a)$.
It discretizes the range of possible returns into 51 fixed "atoms" (bins) and learns a probability mass function over these atoms. The Bellman update involves projecting the target distribution onto these fixed atoms.

## Q3: Why is Distributional RL considered "Risk-Sensitive"?
**Answer:**
Standard RL maximizes expected value $\mathbb{E}[Z]$.
*   Action A: 50% chance of 0, 50% chance of 100. Mean = 50.
*   Action B: 100% chance of 50. Mean = 50.
Standard RL sees A and B as equal.
Distributional RL knows the variance. A risk-averse agent (using IQN, for example) could choose B (lower variance), while a risk-seeking agent could choose A (higher potential upside).

## Q4: What is the difference between Aleatoric and Epistemic uncertainty in RL?
**Answer:**
*   **Aleatoric:** Inherent randomness in the environment (e.g., a coin flip, stochastic transition). Distributional RL models this.
*   **Epistemic:** Uncertainty due to lack of knowledge (e.g., "I haven't visited this state before"). Exploration methods (like Noisy Nets or Count-Based) target this.

## Q5: How does Quantile Regression DQN (QR-DQN) differ from C51?
**Answer:**
*   **C51:** Fixes the locations (x-axis) and learns the probabilities (y-axis).
*   **QR-DQN:** Fixes the probabilities (y-axis, e.g., uniform quantiles) and learns the locations (x-axis).
QR-DQN eliminates the need to tune the support range $[V_{min}, V_{max}]$ and generally performs better.
