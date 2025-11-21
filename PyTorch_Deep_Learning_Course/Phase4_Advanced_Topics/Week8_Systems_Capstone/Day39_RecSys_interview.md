# Day 39: Recommender Systems - Interview Questions

> **Phase**: 4 - Advanced Topics
> **Week**: 8 - Systems & Capstone
> **Topic**: Retrieval, Ranking, and Architecture

### 1. What is the difference between Retrieval and Ranking?
**Answer:**
*   **Retrieval**: Fast selection of ~1000 candidates from millions of items. High Recall. Simple model (Two-Tower, ANN).
*   **Ranking**: Precise scoring of the 1000 candidates. High Precision. Complex model (DeepFM, DCN).

### 2. What is "Matrix Factorization"?
**Answer:**
*   Decomposing the User-Item interaction matrix $Y$ into two low-rank matrices $U$ and $V$.
*   $Y \approx U V^T$.
*   Captures latent factors.

### 3. Why use "Two-Tower" architecture?
**Answer:**
*   Decouples User and Item processing.
*   Allows pre-computing Item embeddings and indexing them in a Vector DB (FAISS).
*   Inference is just a Nearest Neighbor search ($O(\log N)$).

### 4. What is "Cold Start" problem?
**Answer:**
*   New User or New Item has no history.
*   **Solution**: Use content features (Meta-data, Images, Description) instead of ID embeddings. Use Bandits (Explore).

### 5. What is "Implicit" vs "Explicit" feedback?
**Answer:**
*   **Explicit**: Ratings (1-5 stars), Likes. Rare.
*   **Implicit**: Clicks, Watch time, Purchase. Abundant but noisy (Click $\neq$ Like).

### 6. What is "Wide & Deep"?
**Answer:**
*   Architecture combining a Linear model (Wide) for memorization and a DNN (Deep) for generalization.
*   Standard in industry (Google Play).

### 7. What is "FM" (Factorization Machine)?
**Answer:**
*   Model that captures 2nd-order interactions between features.
*   $w_i x_i + w_j x_j + <v_i, v_j> x_i x_j$.
*   Solves sparsity problem of polynomial regression.

### 8. What is "Hard Negative Mining"?
**Answer:**
*   Training on negatives that are difficult for the model (high predicted score but actually negative).
*   Improves model discrimination.

### 9. What is "NDCG"?
**Answer:**
*   Normalized Discounted Cumulative Gain.
*   Ranking metric.
*   Rewards correct items at the top of the list more than at the bottom.

### 10. What is "Bias" in RecSys?
**Answer:**
*   **Position Bias**: Items at top get clicked more regardless of relevance.
*   **Popularity Bias**: Popular items are recommended too often.
*   **Selection Bias**: We only observe data for shown items.

### 11. What is "SASRec"?
**Answer:**
*   Sequential Recommender using Self-Attention.
*   Predicts next item based on sequence of past items.
*   SOTA for sequential tasks.

### 12. What is "DCN" (Deep & Cross Network)?
**Answer:**
*   Network that explicitly learns feature crosses of arbitrary order.
*   Efficient (Linear complexity).

### 13. How to handle "Multiple Objectives"?
**Answer:**
*   Optimize for Clicks, Watch Time, Shares simultaneously.
*   **MMOE (Multi-Gate Mixture-of-Experts)**: Shared experts with task-specific gates.
*   Weighted sum of losses.

### 14. What is "ANN" (Approximate Nearest Neighbor)?
**Answer:**
*   Algorithms (HNSW, IVF) to find nearest vectors in sub-linear time.
*   Crucial for Retrieval stage.

### 15. What is "Session-based Recommendation"?
**Answer:**
*   Recommending based on the current anonymous session (short-term intent) rather than long-term user history.
*   Uses RNNs/Transformers.

### 16. What is "Content-based Filtering"?
**Answer:**
*   Recommending items similar to items the user liked, based on features (Genre, Actor).
*   Does not need other users' data.

### 17. What is "Collaborative Filtering"?
**Answer:**
*   Recommending based on similar users.
*   "Users who bought X also bought Y".

### 18. What is "Pointwise" vs "Pairwise" vs "Listwise" loss?
**Answer:**
*   **Pointwise**: Predict score for one item (MSE/BCE).
*   **Pairwise**: Predict if Item A > Item B (BPR, MarginRanking).
*   **Listwise**: Optimize the whole list order (LambdaRank).

### 19. What is "BPR" (Bayesian Personalized Ranking)?
**Answer:**
*   Pairwise loss function.
*   Maximizes the difference between positive and negative item scores.
*   $\sum \ln \sigma(x_{uij})$.

### 20. How to debug a RecSys model?
**Answer:**
*   Check offline metrics (AUC) vs online metrics (CTR).
*   Check diversity of recommendations.
*   Check for popularity bias (is it just recommending top 10 items?).
