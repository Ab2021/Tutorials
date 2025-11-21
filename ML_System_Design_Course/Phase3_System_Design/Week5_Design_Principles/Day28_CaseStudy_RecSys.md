# Day 28: Case Study - Recommendation System (Netflix/YouTube)

> **Phase**: 3 - System Design
> **Week**: 5 - Design Principles (Case Studies)
> **Focus**: The Funnel Architecture
> **Reading Time**: 60 mins

---

## 1. The Problem

**Goal**: Recommend $k$ items from a catalog of $N$ (millions) to a user, maximizing watch time/engagement.
**Constraint**: Latency < 200ms.

---

## 2. The Architecture: The Funnel

We cannot score 10 Million items with a heavy neural net in 200ms. We use a multi-stage funnel.

### Stage 1: Candidate Generation (Retrieval)
*   **Goal**: Select Top 1000 relevant items from 10 Million. High Recall, Low Precision. Fast.
*   **Methods**:
    *   **Collaborative Filtering**: Matrix Factorization (ALS).
    *   **Two-Tower DNN**: User Tower encodes user -> vector $U$. Item Tower encodes item -> vector $I$. Compute Dot Product using **ANN (Approximate Nearest Neighbor)** index (FAISS/ScaNN).
    *   **Heuristics**: "Trending Now", "Watch it Again".

### Stage 2: Ranking
*   **Goal**: Reorder the Top 1000 to find the Top 10. High Precision. Slower.
*   **Model**: Heavy Deep Learning model (Multi-Task Learning).
*   **Features**: User history, Item details, Context (Time, Device), Interaction features (User X Item).

### Stage 3: Re-Ranking (Business Logic)
*   **Diversity**: Don't show 10 action movies. Shuffle to show genres.
*   **Freshness**: Boost new content.
*   **Filters**: Remove watched items.

---

## 3. Deep Dive: Two-Tower Model

*   **User Tower**: Inputs (ID embedding, Watch History average, Demographics) -> Dense Layers -> User Vector ($d=64$).
*   **Item Tower**: Inputs (ID embedding, Title text embedding, Genre) -> Dense Layers -> Item Vector ($d=64$).
*   **Training**: Softmax Cross-Entropy. Maximize dot product for positive pairs (User, Watched Item).
*   **Serving**: Precompute all Item Vectors and store in FAISS. At runtime, compute User Vector and query FAISS.

---

## 4. Interview Preparation

### System Design Questions

**Q1: How do you handle "New Items" (Cold Start) in the Two-Tower model?**
> **Answer**:
> *   **Content-Based**: The Item Tower uses content features (Title, Description, Video Frames). Even if the ID is new, the content features allow the model to generate a meaningful vector close to similar items.
> *   **Exploration**: Boost new items in the Re-Ranking phase to gather interaction data.

**Q2: Why use Multi-Task Learning in Ranking?**
> **Answer**: We care about multiple objectives: Click, Watch Time, Like, Share.
> *   Training a single model with multiple heads (one for each task) allows the shared layers to learn robust representations.
> *   Final Score = $w_1 \cdot P(\text{Click}) + w_2 \cdot P(\text{Like}) + w_3 \cdot \text{PredictedWatchTime}$.

**Q3: How do you update the ANN index?**
> **Answer**:
> *   **Batch**: Rebuild the index every night. (New items appear next day).
> *   **Streaming**: Some ANN libraries (HNSW) support incremental updates. Or use a Hybrid index (Static Main Index + Small Dynamic Index).

---

## 5. Further Reading
- [Deep Neural Networks for YouTube Recommendations (Paper)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)
- [System Design for Recommendations (Eugene Yan)](https://eugeneyan.com/writing/system-design-for-discovery/)
