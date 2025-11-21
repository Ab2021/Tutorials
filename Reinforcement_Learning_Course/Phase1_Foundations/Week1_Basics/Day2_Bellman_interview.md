# Day 2 Interview Questions: Bellman Equations

## Q1: What is the difference between the Bellman Expectation Equation and the Bellman Optimality Equation?
**Answer:**
*   **Expectation Equation:** Describes the value of a state for a *specific, fixed* policy $\pi$. It uses an average over actions ($\sum \pi(a|s)$).
*   **Optimality Equation:** Describes the value of a state for the *optimal* policy $\pi_*$. It uses the maximum over actions ($\max_a$).

## Q2: Can we solve the Bellman Optimality Equation using Linear Algebra (Matrix Inversion)?
**Answer:**
No. The Optimality Equation contains a `max` operator ($V_*(s) = \max_a ...$), which makes the system of equations **non-linear**.
The Expectation Equation, however, is linear ($V = R + \gamma PV$) and *can* be solved via matrix inversion ($V = (I - \gamma P)^{-1} R$) for small state spaces.

## Q3: Why does Value Iteration converge?
**Answer:**
It converges because the Bellman Optimality Operator is a **contraction mapping** with respect to the max-norm ($L_\infty$), provided the discount factor $\gamma < 1$. By the Banach Fixed Point Theorem, repeated application of a contraction mapping is guaranteed to converge to a unique fixed point (the optimal value function).

## Q4: What is the relationship between $V(s)$ and $Q(s, a)$?
**Answer:**
1.  $V(s) = \max_a Q(s, a)$ (for the optimal policy).
2.  $Q(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) V(s')$.
Basically, $V(s)$ is the value of being in a state, and $Q(s, a)$ is the value of taking a specific action in that state and *then* following the policy.

## Q5: If $\gamma = 1$, does the Bellman Equation still work?
**Answer:**
Not necessarily.
*   If the horizon is infinite and rewards are always positive, the value could be infinite, and the contraction property doesn't hold.
*   If the task is episodic (guaranteed to end), $\gamma=1$ is fine (e.g., a game of Chess where you win +1 or lose -1).
