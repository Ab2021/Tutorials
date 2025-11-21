# Day 6 Interview Questions: N-Step TD & Eligibility Traces

## Q1: What is the main benefit of N-Step TD over 1-Step TD?
**Answer:**
N-Step TD propagates reward information faster.
In 1-Step TD, a reward at step $T$ only affects the state $S_{T-1}$ in the first episode. It takes another episode for that value to propagate to $S_{T-2}$, and so on.
In N-Step TD, the reward affects $S_{T-n}, ..., S_{T-1}$ immediately. This speeds up learning, especially in environments with sparse, delayed rewards.

## Q2: What happens when $\lambda = 1$ in TD($\lambda$)?
**Answer:**
It becomes equivalent to **Monte Carlo** methods (specifically, offline $\lambda$-return matches MC).
The agent waits until the end of the episode (effectively) to update, as the trace decays only by $\gamma$ (not by $\lambda$).
*   Note: In online TD($\lambda$), it's not *exactly* MC because updates happen at every step, but the *total* update over the episode is equivalent to MC.

## Q3: Explain the "Backward View" of Eligibility Traces.
**Answer:**
The Backward View is a mechanistic implementation. Instead of looking forward in time (which requires waiting), we look backward.
When a TD error $\delta_t$ occurs (surprise), we ask: "Which states contributed to us getting here?"
The eligibility trace $E_t(s)$ answers this. We update all states in proportion to their trace: $V(s) \leftarrow V(s) + \alpha \delta_t E_t(s)$.

## Q4: Why do we need to "cut the trace" in Watkin's Q($\lambda$)?
**Answer:**
Because Q-Learning is off-policy. We are learning the value of the *greedy* policy.
If we take a non-greedy (exploratory) action, the chain of "optimal decisions" is broken. Future rewards from that point onwards result from a sub-optimal policy, so they shouldn't be used to update the value of the optimal policy. Therefore, we zero out the traces.

## Q5: What is the computational complexity of using Eligibility Traces?
**Answer:**
*   **Naive:** $O(|S|)$ per step, because we update traces for all states.
*   **Sparse/Lazy:** Can be $O(k)$ where $k$ is the number of active features (or recently visited states), making it efficient for large state spaces.
