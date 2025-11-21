# Day 4 Interview Questions: Monte Carlo Methods

## Q1: Why can't Monte Carlo methods be used for continuous (non-terminating) tasks?
**Answer:**
MC methods require the episode to terminate to calculate the return $G_t$. In a continuous task, the return would be an infinite sum (or require arbitrary truncation), and the update step $V(S_t) \leftarrow V(S_t) + \alpha (G_t - V(S_t))$ could never happen until the end of time.

## Q2: What is the difference between On-Policy and Off-Policy learning?
**Answer:**
*   **On-Policy:** The agent learns the value of the policy it is currently executing ($\pi \approx \mu$). It attempts to improve the policy that generates the data. (e.g., SARSA).
*   **Off-Policy:** The agent learns the value of a target policy $\pi$ (usually optimal) while following a different behavior policy $\mu$. (e.g., Q-Learning). This allows learning from historical data or human demonstrations.

## Q3: Why does Monte Carlo have high variance?
**Answer:**
The return $G_t = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{T-1} R_T$ depends on many random events:
1.  Stochastic transitions $P(s'|s,a)$ at every step.
2.  Stochastic rewards $R(s,a)$ at every step.
3.  Stochastic policy $\pi(a|s)$ at every step.
These random variables accumulate over the length of the episode, leading to a high variance in the observed return $G_t$.

## Q4: Does Monte Carlo require the Markov Property?
**Answer:**
Strictly speaking, for **prediction** (estimating expected return), MC does *not* require the Markov property. The average return is an unbiased estimate of the value of the *history* observed.
However, if we aggregate these histories into "states" that are not Markovian (aliasing), our value function $V(s)$ will be an average over different real situations, which might not be useful for **control** (choosing optimal actions).

## Q5: What is the "GLIE" property?
**Answer:**
GLIE stands for **Greedy in the Limit with Infinite Exploration**.
For an MC Control algorithm to converge to the optimal policy:
1.  **Infinite Exploration:** Every state-action pair must be visited infinitely often.
2.  **Greedy in Limit:** The policy must converge to the greedy policy (e.g., $\epsilon \to 0$ over time).
