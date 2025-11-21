# Day 28: Case Study: Recommendation System - Interview Questions

> **Topic**: System Design Mock
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Design a Video Recommendation System (YouTube/TikTok).
**Answer:**
*   **Goal**: Maximize Watch Time / Engagement.
*   **Architecture**: Candidate Generation (Funnel) -> Ranking -> Re-ranking.

### 2. Explain the Candidate Generation (Retrieval) stage.
**Answer:**
*   **Input**: Millions of videos. **Output**: Hundreds.
*   **Methods**:
    *   **Collaborative Filtering**: Matrix Factorization.
    *   **Two-Tower NN**: User Encoder, Item Encoder. Dot product search (ANN).
    *   **Graph**: Random Walks.

### 3. Explain the Ranking stage.
**Answer:**
*   **Input**: Hundreds. **Output**: Dozens.
*   **Model**: Heavy Neural Net (Deep & Cross, DLRM).
*   **Features**: User history, Video metadata, Context (Time, Device).
*   **Objective**: Predict p(Click), p(Watch > 30s).

### 4. What is Matrix Factorization?
**Answer:**
*   Decompose User-Item Interaction Matrix $R$ into $U \times V^T$.
*   Learns latent factors (embeddings).
*   Dot product approximates rating.

### 5. How do you handle Implicit Feedback?
**Answer:**
*   Explicit (Stars) is rare. Implicit (Clicks, Watch time) is abundant.
*   Treat Click = 1, No Click = 0 (with Negative Sampling).
*   Weighted Matrix Factorization (ALS).

### 6. What is Negative Sampling?
**Answer:**
*   We have only positives (clicks).
*   Treat unobserved items as negatives.
*   Don't use ALL unobserved (too many). Sample a few.
*   **Hard Negatives**: Items shown but not clicked.

### 7. Explain "Two-Tower" Architecture.
**Answer:**
*   Query Tower -> User Embedding.
*   Item Tower -> Item Embedding.
*   Training: Maximize dot product for positive pairs.
*   Serving: Pre-compute Item Embeddings. Index in FAISS. Compute User Embedding live. Query FAISS.

### 8. How do you optimize for multiple objectives (Clicks vs Watch Time)?
**Answer:**
*   **Multi-Task Learning (MTL)**. Shared bottom, separate heads.
*   **Combination**: $Score = w_1 \cdot P(Click) + w_2 \cdot P(Watch)$.
*   Or $Score = P(Click) \times E[WatchTime]$.

### 9. What is the "Cold Start" problem in RecSys?
**Answer:**
*   New User/Item has no history.
*   **User**: Use demographics, location, popular items.
*   **Item**: Use content features (Title, Thumbnail) via CNN/BERT. Exploration (Bandits).

### 10. Explain "Exploration vs Exploitation".
**Answer:**
*   **Exploit**: Show what we know they like. (Safe, short-term).
*   **Explore**: Show new things. (Risk, long-term learning).
*   **Bandits**: Thompson Sampling / UCB.

### 11. How do you evaluate a RecSys offline?
**Answer:**
*   **Precision@K**, **Recall@K**.
*   **NDCG** (Normalized Discounted Cumulative Gain) - accounts for rank order.
*   **MRR** (Mean Reciprocal Rank).

### 12. How do you evaluate a RecSys online?
**Answer:**
*   **A/B Test**.
*   Metrics: Time Spent, CTR, Retention, Revenue.

### 13. What is "Position Bias"? How to fix?
**Answer:**
*   Items at top get clicked more just because they are at top.
*   **Fix**: Add Position as a feature during training. During inference, set Position = 0 (or default).

### 14. Explain Wide & Deep Learning.
**Answer:**
*   **Wide**: Memorization (Cross features). Linear model.
*   **Deep**: Generalization (Embeddings). MLP.
*   Combines both.

### 15. How do you handle real-time updates?
**Answer:**
*   User clicks video.
*   Update User Embedding immediately (RNN/Transformer over history).
*   Refresh recommendations.

### 16. What is DLRM (Deep Learning Recommendation Model)?
**Answer:**
*   Meta's architecture.
*   Handles categorical features (Embeddings) and numerical features (MLP).
*   Feature Interaction via Dot Product.

### 17. How do you deal with "Filter Bubbles"?
**Answer:**
*   User gets stuck in a niche.
*   **Fix**: Force Diversity. Re-ranking rules. Exploration.

### 18. What is "Re-ranking"?
**Answer:**
*   Post-processing step.
*   Apply business logic: Diversity, Remove duplicates, Demote clickbait, Ad insertion.

### 19. How do you scale ANN (Approximate Nearest Neighbor) search?
**Answer:**
*   **HNSW** (Hierarchical Navigable Small World).
*   **IVF** (Inverted File).
*   **Quantization** (PQ).
*   Libraries: FAISS, ScaNN, Milvus.

### 20. What features would you use for a Movie RecSys?
**Answer:**
*   **User**: Age, Gender, Past Genres, Past Ratings.
*   **Movie**: Genre, Director, Actors, Year, Embedding of Plot.
*   **Context**: Time of day, Weekend vs Weekday.
