# Day 3 Interview Questions: Dynamic Programming

## Q1: What is the main difference between Policy Iteration and Value Iteration?
**Answer:**
*   **Policy Iteration:** Alternates between *complete* Policy Evaluation (solving for $V_{\pi}$ exactly) and Policy Improvement. It usually takes fewer iterations to converge but each iteration is computationally expensive.
*   **Value Iteration:** Combines evaluation and improvement into a single step. It truncates the evaluation step to just one update. It takes more iterations but each iteration is faster.

## Q2: What is the computational complexity of one sweep of Value Iteration?
**Answer:**
$O(|S|^2 |A|)$.
For each state ($|S|$), we iterate over all actions ($|A|$), and for each action, we sum over all possible next states ($|S|$).
If the transition matrix is sparse (max branching factor $b$), it becomes $O(|S| \cdot |A| \cdot b)$.

## Q3: What is "Bootstrapping" in the context of RL/DP?
**Answer:**
Bootstrapping means updating an estimate based on other estimates.
In DP, $V(s)$ is updated using the current estimated value of $V(s')$.
This contrasts with Monte Carlo methods (Day 4), which update estimates based on actual observed returns (no bootstrapping).

## Q4: Why is DP considered "Planning" and not "Learning"?
**Answer:**
DP requires access to the full model of the environment ($P(s'|s,a)$ and $R(s,a)$). It computes the solution without necessarily interacting with the environment.
"Learning" usually implies the agent doesn't know the model and must discover it (or the optimal policy) through trial-and-error interaction.

## Q5: Does Asynchronous DP converge?
**Answer:**
Yes, provided that every state is visited/updated infinitely often. In-place updates often converge faster than synchronous sweeps because information propagates more quickly through the state space.
