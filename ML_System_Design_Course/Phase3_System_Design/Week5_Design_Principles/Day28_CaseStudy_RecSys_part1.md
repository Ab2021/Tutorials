# Day 28 (Part 1): Advanced Recommendation Systems

> **Phase**: 6 - Deep Dive
> **Topic**: The Art of Ranking
> **Focus**: Negative Sampling, DCN, and ALS
> **Reading Time**: 60 mins

---

## 1. Negative Sampling

In Implicit Feedback (Clicks), we only have Positives. How to get Negatives?

### 1.1 Random Negatives
*   Sample random items user didn't click.
*   **Easy**, but uninformative.

### 1.2 Hard Negatives
*   Items the user *saw* but didn't click (Impressions).
*   Items ranked high by the model but not clicked.
*   **Danger**: False Negatives (User would have liked it, but didn't see it).

---

## 2. Model Architectures

### 2.1 Matrix Factorization (ALS)
*   **Alternating Least Squares**.
*   Fix User vectors, solve Item vectors (OLS). Fix Items, solve Users. Repeat.
*   **Parallelizable**: Spark implementation is standard.

### 2.2 DCN (Deep Cross Network)
*   **Goal**: Learn explicit feature interactions (like `City x Category`) without manual engineering.
*   **Cross Layer**: $x_{l+1} = x_0 x_l^T w_l + b_l + x_l$.
*   **Result**: Polynomial degree interactions.

---

## 3. Tricky Interview Questions

### Q1: How to handle "Position Bias" in training data?
> **Answer**:
> *   Items at Rank 1 get clicked more.
> *   **Fix**: Add `Position` as a feature during training.
> *   **Inference**: Set `Position = 0` (or a fixed value) for all items to predict relevance unbiased by position.

### Q2: Explain "Surprise" in RecSys.
> **Answer**:
> *   Recommending what user already knows (Milk) is high accuracy but low value.
> *   **Serendipity**: Recommending something unexpected but liked. Hard to optimize.

### Q3: WALS (Weighted ALS)?
> **Answer**:
> *   Weight positives ($w=1$) and negatives ($w=0.01$) differently.
> *   Crucial for implicit feedback where we have billions of "unknowns" (negatives).

---

## 4. Practical Edge Case: Popularity Bias
*   **Problem**: Model recommends Harry Potter to everyone.
*   **Fix**: Downsample popular items in training. Or use "Popularity" as a feature so model learns to factor it out.

