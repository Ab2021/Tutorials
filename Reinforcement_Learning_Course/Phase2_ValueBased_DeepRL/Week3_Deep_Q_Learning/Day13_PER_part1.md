# Day 13 Deep Dive: Implementing PER Efficiently

## 1. The SumTree Data Structure
To sample with probability $P(i)$, we can't just iterate through the buffer (that would be $O(N)$). We need $O(\log N)$.
A **SumTree** is a binary tree where parent nodes are the sum of their children.
*   **Leaf Nodes:** Store the priorities $p_i$.
*   **Root Node:** Stores the total priority sum $\sum p_i$.
*   **Sampling:**
    1.  Pick a random number $s \in [0, \text{TotalSum}]$.
    2.  Traverse down:
        *   If $s < \text{LeftChild.Value}$: Go Left.
        *   Else: $s \leftarrow s - \text{LeftChild.Value}$, Go Right.
    3.  This reaches a leaf node in $\log_2 N$ steps.

## 2. Annealing Bias Correction ($\beta$)
The importance sampling weights are:
$$ w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta $$
*   **Early Training:** We care more about learning *something* than learning the *exact* distribution. Bias is acceptable. $\beta$ starts small (e.g., 0.4).
*   **Late Training:** We need the Q-values to converge to the true values. Unbiased gradients are critical. $\beta$ anneals to 1.0.

## 3. Rank-Based Prioritization
Instead of using raw TD error $|\delta|$ (which can be unstable and sensitive to outliers), we can use the **Rank** of the transition in the sorted list of errors.
$$ P(i) = \frac{1}{\text{rank}(i)} $$
*   **Pros:** Robust to outlier errors.
*   **Cons:** Harder to maintain a sorted structure efficiently. Most implementations stick to Proportional Prioritization (using $|\delta|$).

## 4. New Priorities
When a new transition $(s, a, r, s')$ enters the buffer, we don't know its TD error yet (we haven't passed it through the network).
*   **Heuristic:** Assign it **Max Priority** (current max in the tree).
*   **Reason:** Ensures all new data is sampled at least once immediately.
