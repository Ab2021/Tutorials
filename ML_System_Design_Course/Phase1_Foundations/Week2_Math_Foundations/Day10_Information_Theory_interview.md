# Day 10: Information Theory - Interview Questions

> **Topic**: Quantifying Information
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is Entropy (Shannon Entropy)?
**Answer:**
*   Measure of uncertainty or randomness in a distribution.
*   $H(X) = - \sum p(x) \log p(x)$.
*   High Entropy = Uniform distribution (Max uncertainty). Low Entropy = Deterministic (Min uncertainty).

### 2. What is the unit of Entropy?
**Answer:**
*   **Bits** (if log base 2).
*   **Nats** (if log base e).

### 3. Explain the intuition behind "Surprise" and Information.
**Answer:**
*   Low probability event = High surprise = High information.
*   "Sun rose today" (Prob $\approx$ 1) -> Zero info.
*   "Aliens landed" (Prob $\approx$ 0) -> Huge info.
*   Info = $-\log p(x)$.

### 4. What is Cross-Entropy? How is it used as a Loss Function?
**Answer:**
*   Measures difference between two distributions P (True) and Q (Predicted).
*   $H(P, Q) = - \sum P(x) \log Q(x)$.
*   In ML, P is one-hot (0, 1, 0). Minimizing Cross-Entropy maximizes the probability assigned to the correct class.

### 5. What is KL Divergence (Kullback-Leibler)?
**Answer:**
*   "Relative Entropy". Distance between P and Q.
*   $D_{KL}(P || Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$.
*   Expected extra bits needed to encode P using code optimized for Q.

### 6. Is KL Divergence symmetric? Why or why not?
**Answer:**
*   **No**. $D_{KL}(P || Q) \neq D_{KL}(Q || P)$.
*   Fitting Q to P (Forward KL) vs Fitting P to Q (Reverse KL) gives different results (Mean-seeking vs Mode-seeking).

### 7. What is Mutual Information?
**Answer:**
*   Amount of information obtained about X by observing Y.
*   $I(X; Y) = H(X) - H(X|Y)$.
*   Reduction in uncertainty. 0 if independent.

### 8. How is Mutual Information related to Entropy and Conditional Entropy?
**Answer:**
*   $I(X; Y) = H(X) + H(Y) - H(X, Y)$.
*   Intersection of the two entropy circles in a Venn diagram.

### 9. What is Information Gain? Where is it used in ML?
**Answer:**
*   Synonym for Mutual Information in Decision Trees.
*   $IG = Entropy(Parent) - WeightedAverage(Entropy(Children))$.
*   Used to decide the best split.

### 10. Explain the relationship between Cross-Entropy and Log-Likelihood.
**Answer:**
*   Minimizing Cross-Entropy is mathematically equivalent to Maximizing Log-Likelihood (for classification).
*   $NLL = - \sum \log(PredictedProb)$.

### 11. What is Perplexity?
**Answer:**
*   $2^{Entropy}$.
*   Measure of how confused a model is.
*   In NLP: "The weighted average branching factor". Perplexity 10 means model is as confused as picking from 10 words uniformly.

### 12. What is the entropy of a fair coin toss?
**Answer:**
*   $P(H)=0.5, P(T)=0.5$.
*   $-0.5 \log_2(0.5) - 0.5 \log_2(0.5) = 0.5 + 0.5 = 1$ bit.

### 13. What is the entropy of a biased coin (P(H)=0.9)? Is it higher or lower than a fair coin?
**Answer:**
*   **Lower**.
*   Less uncertainty. We are pretty sure it will be Heads.
*   $-0.9 \log(0.9) - 0.1 \log(0.1) \approx 0.47$ bits.

### 14. What is Joint Entropy?
**Answer:**
*   Uncertainty of two variables happening together.
*   $H(X, Y) = - \sum \sum p(x,y) \log p(x,y)$.

### 15. What is Conditional Entropy?
**Answer:**
*   Uncertainty of X remaining after knowing Y.
*   $H(X|Y)$.

### 16. Explain the concept of "Bit" in Information Theory.
**Answer:**
*   Amount of info needed to distinguish between 2 equally likely options.

### 17. Why do we minimize Cross-Entropy instead of maximizing Accuracy during training?
**Answer:**
*   **Accuracy** is discrete and non-differentiable (step function). Gradients are zero almost everywhere.
*   **Cross-Entropy** is smooth and differentiable. Provides gradients even when prediction is correct but not confident (e.g., 0.51 vs 0.99).

### 18. What is the Gini Impurity? How does it compare to Entropy?
**Answer:**
*   $1 - \sum p_i^2$.
*   Approximation of Entropy. Computationally cheaper (no logs).
*   Behaves very similarly for Decision Trees.

### 19. What is Differential Entropy (for continuous variables)?
**Answer:**
*   Extension of entropy to PDFs.
*   Can be negative! (Unlike discrete entropy).
*   Depends on coordinate scaling.

### 20. How is Information Theory used in Decision Trees?
**Answer:**
*   To build the tree greedily.
*   At each node, iterate all features. Calculate Information Gain. Pick feature with Max Gain.
