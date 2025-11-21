# Day 17 Interview Questions: Continuous Control with Q-Learning

## Q1: Why is standard DQN not suitable for continuous action spaces?
**Answer:**
DQN requires computing $\max_a Q(s, a)$ to calculate the target.
In discrete spaces, we can just iterate over all $k$ actions.
In continuous spaces, finding the global maximum of a general non-linear function $Q(s, a)$ is an expensive optimization problem (e.g., using Gradient Ascent or Cross-Entropy Method) that must be solved *at every step* of training. This is computationally prohibitive.

## Q2: How does NAF solve the maximization problem?
**Answer:**
NAF restricts the architecture of the Q-network to be **quadratic** in the action $a$.
$$ Q(s, a) = V(s) - \frac{1}{2}(a - \mu(s))^T P(s) (a - \mu(s)) $$
Because of this specific structure, we know analytically that the maximum occurs at $a = \mu(s)$, and the maximum value is $V(s)$. This avoids the need for iterative optimization.

## Q3: What is the main limitation of NAF?
**Answer:**
**Unimodality.** Since it models $Q$ as a quadratic, it assumes there is only one peak (optimal action).
If the environment requires a multi-modal policy (e.g., avoiding an obstacle by going either fully Left or fully Right), NAF might average the two and choose to go Straight (into the obstacle).
More general Actor-Critic methods (like SAC) or Distributional RL can handle multi-modality better.

## Q4: What is the role of the matrix $P(s)$ in NAF?
**Answer:**
$P(s)$ represents the **curvature** (Hessian) of the Q-function around the peak.
It tells us how sensitive the value is to deviations from the optimal action.
*   Large $P(s)$: The peak is sharp. Small deviations cause a large drop in value (Risky).
*   Small $P(s)$: The peak is flat. Deviations don't matter much (Safe/Indifferent).

## Q5: How do we ensure $P(s)$ is positive-definite?
**Answer:**
We don't have the network output $P$ directly. Instead, we output a lower-triangular matrix $L$ with positive diagonal elements (using exponentiation).
Then we compute $P = L L^T$. This construction guarantees that $P$ is positive-definite (or semi-definite) by definition.
