# Day 10 Interview Questions: Phase 1 Review

## Q1: Summarize the difference between Model-Based and Model-Free RL.
**Answer:**
*   **Model-Based:** The agent has access to (or learns) the transition dynamics $P(s'|s,a)$ and reward function $R(s,a)$. It uses **Planning** (e.g., DP, MCTS) to find the optimal policy.
*   **Model-Free:** The agent does not know $P$ or $R$. It learns purely from **Experience** (samples of transitions) using methods like Monte Carlo or TD Learning.

## Q2: What is the "Credit Assignment Problem"?
**Answer:**
The challenge of determining which past action is responsible for a current reward.
*   **Temporal:** The reward might come many steps after the action (delayed reward).
*   **Structural:** Which component of the action vector caused the reward?
Eligibility Traces and TD learning help solve the temporal aspect by propagating credit back in time.

## Q3: Why is Off-Policy learning harder than On-Policy?
**Answer:**
Off-policy learning involves two distributions: the behavior policy $\mu$ (generating data) and the target policy $\pi$ (being learned).
1.  **Variance:** Importance sampling ratios can have infinite variance.
2.  **Instability:** The "Deadly Triad" (FA + Bootstrapping + Off-Policy) can cause divergence.
However, it is more sample-efficient (can reuse old data/experience replay).

## Q4: Explain the Bias-Variance Tradeoff in RL.
**Answer:**
*   **Monte Carlo:** Unbiased (converges to true mean), High Variance (depends on full episode noise).
*   **TD(0):** Biased (bootstraps from current estimate), Low Variance (depends on 1-step noise).
*   **TD($\lambda$):** Interpolates between the two.

## Q5: What is the difference between $V(s)$ and $Q(s,a)$?
**Answer:**
*   $V(s)$: How good is it to be in state $s$? (Averaged over all possible actions).
*   $Q(s,a)$: How good is it to take action $a$ in state $s$?
We need $Q$ for model-free control because we can't do the lookahead $\sum P(s'|s,a)V(s')$ without the model $P$.
