# Day 13 Interview Questions: Prioritized Experience Replay

## Q1: Why does Prioritized Experience Replay introduce bias?
**Answer:**
The standard Q-learning update rule assumes that samples are drawn from the underlying distribution of the environment (or the replay buffer's uniform distribution).
By over-sampling "hard" transitions (high TD error), we change the expectation operator $\mathbb{E}$. The agent thinks these rare, hard events happen much more frequently than they actually do. This biases the gradient estimate.

## Q2: How do we correct this bias?
**Answer:**
We use **Importance Sampling (IS)** weights.
$$ w_i = \frac{1}{P(i)^\beta} $$
We multiply the loss of each sample by $w_i$.
*   If a sample has high priority $P(i)$ (oversampled), $w_i$ is small (downweight the update).
*   If a sample has low priority (undersampled), $w_i$ is large (upweight the update).
This restores the correct expected value of the gradient.

## Q3: What is the time complexity of sampling in PER?
**Answer:**
$O(\log N)$, where $N$ is the buffer size.
This is achieved using a **SumTree** (Segment Tree). A naive implementation (iterating through the list to find the cumulative sum interval) would be $O(N)$, which is too slow for large buffers ($N=10^6$).

## Q4: What happens if we set $\alpha=0$ in PER?
**Answer:**
$$ P(i) \propto p_i^\alpha $$
If $\alpha=0$, then $P(i) \propto 1$. This effectively makes the probabilities uniform. PER becomes standard Uniform Experience Replay.

## Q5: Why do we add a small constant $\epsilon$ to the priority $p_i = |\delta_i| + \epsilon$?
**Answer:**
To ensure that no transition has zero probability of being sampled.
If the TD error is exactly zero (e.g., the agent predicted perfectly once), we don't want to never see that sample again. The environment might be stochastic, or the Q-function might change later, making that sample valuable again.
