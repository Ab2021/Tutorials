# Day 1 Deep Dive: The Nuances of MDPs

## 1. Why do we need a Discount Factor ($\gamma$)?
The discount factor $\gamma \in [0, 1]$ serves two primary purposes:

### A. Mathematical Convergence
In an infinite horizon task (where the episode never ends), the sum of rewards could be infinite:
$$ G_t = \sum_{k=0}^{\infty} R_{t+k+1} $$
If $R_t = 1$ for all $t$, then $G_t = \infty$. This makes it impossible to compare policies (is one infinity better than another?).
By adding $\gamma < 1$, the sum becomes a **geometric series**:
$$ G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \le R_{max} \sum_{k=0}^{\infty} \gamma^k = \frac{R_{max}}{1-\gamma} $$
This ensures the return is always finite.

### B. Uncertainty and Financial Analogy
*   **Uncertainty:** We are less confident about predictions far into the future. Discounting reflects this uncertainty.
*   **Interest Rate:** In economics, money today is worth more than money tomorrow. Similarly, a reward today is better than a reward tomorrow because the agent might die or the episode might end stochastically.

## 2. Finite vs. Infinite Horizon
*   **Finite Horizon:** The episode has a fixed length $T$ (e.g., play a game for 1000 steps). The optimal policy might be **non-stationary** (dependent on time). For example, in the last step, you might take a risky action you wouldn't take at step 1.
*   **Infinite Horizon:** The agent goes on forever (or until a terminal state is reached, which might never happen). The optimal policy is usually **stationary** (depends only on state, not time).

## 3. Partially Observable MDPs (POMDPs)
In the real world, we often don't see the full state $S_t$. We see an **observation** $O_t$ which is a noisy or incomplete view of $S_t$.
*   **Example:** A robot with a camera doesn't know what's behind a wall, but the "state" of the world includes what's behind the wall.
*   **Solution:** The agent must maintain a **Belief State** $b(s)$, which is a probability distribution over possible states given the history of observations.
*   Mathematically, a POMDP is a tuple $(S, A, P, R, \Omega, O)$, where $\Omega$ is the set of observations and $O$ is the observation function.

## 4. Model-Based vs. Model-Free RL
*   **Model-Based:** The agent learns (or is given) the transition probability $P(s'|s,a)$ and reward function $R(s,a)$. It can then "plan" (think ahead) without acting. (e.g., Chess engines).
*   **Model-Free:** The agent doesn't know $P$ or $R$. It learns purely by trial and error, estimating values directly from experience. (e.g., Q-Learning).
