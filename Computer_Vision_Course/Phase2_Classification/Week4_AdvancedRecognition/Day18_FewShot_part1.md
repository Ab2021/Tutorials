# Day 18 Deep Dive: Transductive vs Inductive

## 1. Inductive vs Transductive
*   **Inductive:** Classify query samples one by one. The model builds a rule from Support and applies it to Query.
*   **Transductive:** Classify all query samples together. The model can use the distribution of the *unlabeled* Query set to help classification.
    *   **Example:** If we know the query set is balanced (5 classes, 10 images each), we can enforce this constraint.
    *   **Performance:** Transductive methods usually outperform Inductive ones in FSL benchmarks.

## 2. SimpleShot (2019)
**Paper:** "SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning".
**Finding:** You don't need complex meta-learning!
1.  Train a standard ResNet backbone on all base classes (standard classification).
2.  **L2 Normalize** feature vectors.
3.  At test time, compute mean of support examples (prototypes).
4.  Classify query by Nearest Neighbor.
*   **Result:** Beats complex meta-learning methods like MAML and Matching Nets.
*   **Lesson:** A good feature extractor is 90% of the battle.

## 3. Matching Networks
**Idea:** Attention + Memory.
*   Uses an LSTM to embed the support set (Context).
*   Classifies query based on cosine similarity to support examples (weighted sum of labels).
    $$ \hat{y} = \sum_{i=1}^k a(x, x_i) y_i $$
*   Where $a$ is an attention kernel (softmax of cosine distance).

## 4. Reptile
**Simplified MAML.**
*   Don't compute second-order derivatives (Hessian) like MAML.
*   **Algorithm:**
    1.  Sample task.
    2.  Train on task for $k$ steps (SGD) to get $\theta'$.
    3.  Update initialization: $\theta \leftarrow \theta + \epsilon (\theta' - \theta)$.
*   Moves initialization towards the trained weights.

## Summary
The field has swung from complex meta-learning algorithms (MAML) back to simple, strong baselines (SimpleShot, Transfer Learning). Good features matter more than clever adaptation algorithms.
