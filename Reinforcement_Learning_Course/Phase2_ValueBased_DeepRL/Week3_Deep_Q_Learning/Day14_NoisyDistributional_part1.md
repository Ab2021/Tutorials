# Day 14 Deep Dive: Advanced Distributional RL

## 1. Quantile Regression DQN (QR-DQN)
C51 has a limitation: it uses fixed bins (atoms) for the support (e.g., -10 to 10). If the rewards are outside this range, it fails.
**QR-DQN** flips the problem:
*   Instead of fixing the *values* (x-axis) and learning probabilities (y-axis),
*   It fixes the *probabilities* (quantiles) and learns the *values*.
*   **Output:** The network outputs $N$ values $q_1, q_2, ..., q_N$ representing the locations of quantiles (e.g., 10th percentile, 20th percentile...).
*   **Loss:** Quantile Regression Loss (asymmetric L1 loss).

## 2. Implicit Quantile Networks (IQN)
QR-DQN uses a fixed number of quantiles (e.g., 200).
**IQN** takes this further:
*   It takes a random probability $\tau \sim U(0, 1)$ as **input** to the network (along with state $s$).
*   It outputs the value of the return at that specific quantile $Z_\tau(s, a)$.
*   **Benefit:** We can query the distribution at *any* resolution.
*   **Risk-Sensitive Policies:**
    *   If we want to be safe, we query low $\tau$ (worst-case scenarios).
    *   If we want to gamble, we query high $\tau$ (best-case scenarios).

## 3. Why does Distributional RL work so well?
It's not just about risk.
*   **Auxiliary Task:** Predicting the distribution forces the network to learn more robust features about the environment dynamics (e.g., "this action usually gives 0 but sometimes gives 100").
*   **Non-Linearity:** The expectation operator $\mathbb{E}$ is linear. The Bellman operator for distributions is more complex and might smooth out optimization landscapes.
