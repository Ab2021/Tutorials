# Day 5 Interview Questions: TD Learning

## Q1: What is the fundamental difference between SARSA and Q-Learning?
**Answer:**
*   **SARSA (On-Policy):** Updates the Q-value using the action $A'$ *actually taken* by the current policy (which might be random/exploratory). It learns the value of the "safe" policy.
*   **Q-Learning (Off-Policy):** Updates the Q-value using the *best possible* action $a'$ (greedy), regardless of what the agent actually does. It learns the value of the optimal policy directly.

## Q2: In the "Cliff Walking" example, why does SARSA take the longer, safer path?
**Answer:**
Because SARSA is on-policy. It knows that if it walks close to the cliff, its $\epsilon$-greedy policy might randomly choose to jump off the cliff (exploration). Therefore, the "value" of walking near the cliff is low (high risk of -100). It prefers the safer path where random actions don't lead to disaster.
Q-Learning assumes it will act optimally in the future (no random jumps), so it sees the edge path as safe.

## Q3: What is the TD Error?
**Answer:**
The TD Error $\delta_t$ is the difference between the **TD Target** (estimated return after one step) and the **Current Estimate**.
$$ \delta_t = [R_{t+1} + \gamma V(S_{t+1})] - V(S_t) $$
It represents the "surprise" or the new information gained from the step.

## Q4: Why is TD learning often faster than Monte Carlo?
**Answer:**
1.  **Updates per Step:** TD updates after every step, whereas MC waits for the end of the episode.
2.  **Lower Variance:** TD targets depend on only one step of randomness, making the learning signal more stable, allowing for larger learning rates.

## Q5: Can TD learning handle continuous state spaces?
**Answer:**
Tabular TD (using a table for Q-values) cannot. However, TD methods can be combined with **Function Approximation** (e.g., Neural Networks) to handle continuous states. This is the basis of Deep Q-Networks (DQN).
