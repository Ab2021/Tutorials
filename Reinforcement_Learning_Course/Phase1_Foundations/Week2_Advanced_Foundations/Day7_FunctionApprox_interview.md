# Day 7 Interview Questions: Function Approximation

## Q1: Why is Function Approximation necessary in modern RL?
**Answer:**
Tabular methods require memory proportional to the number of states $|S|$. For real-world problems like robotics (continuous state) or Go ($10^{170}$ states), storing a table is impossible. FA allows us to parameterize the value function with a much smaller number of weights $d \ll |S|$ and generalize to unseen states.

## Q2: What is the "Deadly Triad"?
**Answer:**
The combination of:
1.  **Function Approximation**
2.  **Bootstrapping**
3.  **Off-Policy Learning**
When these three are present, many RL algorithms (like Q-Learning with Linear FA) are not guaranteed to converge and can diverge.

## Q3: Why is Gradient Descent in TD learning called "Semi-Gradient"?
**Answer:**
Because we ignore the dependence of the TD target on the weights.
The target $R + \gamma \hat{v}(S', \mathbf{w})$ changes as $\mathbf{w}$ changes. True gradient descent would account for this derivative. Semi-gradient methods treat the target as a fixed constant for the purpose of the update, which simplifies the update rule.

## Q4: Does Linear Q-Learning converge?
**Answer:**
Not necessarily. Because it combines all elements of the Deadly Triad, it can diverge. However, in practice, with low learning rates and careful feature engineering, it often works. Algorithms like **Gradient Q-Learning (GQ)** were developed to fix this theoretical issue.

## Q5: What are "Tile Coding" and "Coarse Coding"?
**Answer:**
They are methods to create binary feature vectors $\phi(s)$ for continuous spaces.
*   **Tile Coding:** Overlays multiple shifted grids (tilings) over the state space. The active features are the tiles containing the current state. It provides local generalization and is computationally efficient.
