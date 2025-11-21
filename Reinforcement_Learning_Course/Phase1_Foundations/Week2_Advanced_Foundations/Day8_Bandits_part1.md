# Day 8 Deep Dive: Advanced Bandit Algorithms

## 1. Optimistic Initial Values
Instead of initializing $Q(a) = 0$, what if we set $Q(a) = +5$ (assuming max reward is 1)?
*   **Mechanism:** The agent tries an arm, gets reward 1. It updates $Q(a)$ down to, say, 4. This is still "disappointing" compared to the +5 of other arms.
*   **Result:** It switches to other arms immediately.
*   **Effect:** It forces the agent to try every arm at least once early on. It's a simple, effective exploration trick for stationary problems.

## 2. Upper Confidence Bound (UCB)
$\epsilon$-Greedy explores randomly. UCB explores **intelligently** by favoring arms with high uncertainty.
$$ A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right] $$
*   **$Q_t(a)$:** Exploitation (current estimate).
*   **$\sqrt{\frac{\ln t}{N_t(a)}}$:** Exploration (uncertainty).
    *   If $N_t(a)$ is small (rarely visited), the term is large.
    *   As $t$ increases, the term grows (log t), ensuring we don't stop exploring entirely too soon.

## 3. Thompson Sampling
A Bayesian approach.
*   **Idea:** Maintain a probability distribution for the mean reward of each arm (e.g., a Beta distribution for click-through rates).
*   **Action:** Sample a value from each arm's distribution. Pick the arm with the highest *sampled* value.
*   **Effect:** If an arm is uncertain (wide distribution), it might produce a high sample by chance, leading to exploration. If it's known to be bad (narrow distribution around low value), it won't be picked.
*   **Property:** Probability Matching. The probability of choosing an arm equals the probability that it is actually the best arm.

## 4. Contextual Bandits
In the standard bandit problem, the problem is the same every step.
In **Contextual Bandits**, the agent sees a **Context** (State) $S_t$ before acting.
*   Reward depends on Action AND Context: $R(S_t, A_t)$.
*   **Difference from MDP:** The action does *not* affect the next state $S_{t+1}$. States are i.i.d. or external.
*   **Use Case:** Recommender Systems (Context = User Profile, Action = Recommended Movie).
