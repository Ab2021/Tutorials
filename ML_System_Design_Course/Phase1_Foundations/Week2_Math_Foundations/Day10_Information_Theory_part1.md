# Day 10 (Part 1): Advanced Information Theory

> **Phase**: 6 - Deep Dive
> **Topic**: Quantifying Uncertainty
> **Focus**: Mutual Information, Gini vs Entropy, and Compression
> **Reading Time**: 60 mins

---

## 1. Mutual Information (MI)

Correlation measures linear relationships. MI measures *any* relationship.

### 1.1 Definition
$I(X; Y) = H(X) - H(X|Y)$.
*   "How much does knowing Y reduce uncertainty about X?"
*   $I(X; Y) = KL(P(X,Y) || P(X)P(Y))$.

### 1.2 Feature Selection
*   Select features with high MI with the target variable.
*   Detects non-linear dependencies that Pearson Correlation misses (e.g., $y = x^2$ on $[-1, 1]$ has 0 correlation but high MI).

---

## 2. Entropy in Trees: Gini vs. Entropy

Decision Trees split nodes to maximize Information Gain.

### 2.1 Gini Impurity
*   $G = 1 - \sum p_i^2$.
*   Interpretation: Probability of misclassifying a randomly chosen element if it were randomly labeled according to the distribution.
*   **Pros**: Faster to compute (no log).

### 2.2 Entropy (Shannon)
*   $H = - \sum p_i \log p_i$.
*   **Comparison**: Gini and Entropy are very similar shapes. Gini peaks at 0.5. Entropy peaks at 1.0 (base 2).
*   **Result**: Rarely makes a difference in accuracy. Gini is default in Sklearn for speed.

---

## 3. Perplexity

The standard metric for LLMs.

### 3.1 Definition
*   $PP(P) = 2^{H(P)}$.
*   Interpretation: The "weighted average branching factor".
*   If Perplexity = 10, the model is as confused as if it had to choose uniformly among 10 words.

---

## 4. Tricky Interview Questions

### Q1: Why is KL Divergence not a distance metric?
> **Answer**:
> 1.  **Asymmetry**: $KL(P||Q) \neq KL(Q||P)$.
> 2.  **Triangle Inequality**: Does not hold.
> *   **Forward KL** ($P \log P/Q$): "Mean seeking". Used in MLE.
> *   **Reverse KL** ($Q \log Q/P$): "Mode seeking". Used in Variational Inference.

### Q2: What is the Entropy of a Gaussian?
> **Answer**: $\frac{1}{2} \log(2\pi e \sigma^2)$.
> *   Depends only on variance $\sigma^2$. Mean $\mu$ does not affect entropy (shifting the curve doesn't change uncertainty).
> *   Gaussian has the *maximum entropy* for a fixed variance.

### Q3: Explain Cross-Entropy vs. KL Divergence in Optimization.
> **Answer**:
> $H(P, Q) = H(P) + KL(P||Q)$.
> *   Since $H(P)$ (Entropy of true labels) is constant (0 for one-hot), minimizing Cross-Entropy is mathematically identical to minimizing KL Divergence.

---

## 5. Practical Edge Case: Zero Probability
*   **Problem**: If model predicts $Q(x) = 0$ but true $P(x) > 0$, then $\log Q(x) = -\infty$. Loss explodes.
*   **Fix**: Label Smoothing. Instead of target `[0, 1, 0]`, use `[0.05, 0.9, 0.05]`. Prevents model from being overconfident and driving weights to infinity.

