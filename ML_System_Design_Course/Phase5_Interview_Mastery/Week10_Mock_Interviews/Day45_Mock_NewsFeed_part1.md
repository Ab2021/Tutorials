# Day 45 (Part 1): Advanced Mock - News Feed

> **Phase**: 6 - Deep Dive
> **Topic**: Social Networks
> **Focus**: Diversity, Ads, and Real-Time
> **Reading Time**: 60 mins

---

## 1. Diversity & Serendipity

Don't just show 10 posts from the same user.

### 1.1 Maximal Marginal Relevance (MMR)
*   Score = $\lambda \cdot \text{Relevance} - (1-\lambda) \cdot \text{Similarity(History)}$.
*   Penalize items similar to what is already in the feed.

### 1.2 Rules Engine
*   "No more than 2 posts from same author."
*   "At least 1 video per 10 posts."
*   Apply during **Re-ranking**.

---

## 2. Ad Injection

### 2.1 The Auction
*   Run generalized second-price auction (GSP).
*   **eCPM** = $p(\text{Click}) \times \text{Bid}$.
*   Insert Ad if eCPM > Organic Content Value.

### 2.2 Calibration
*   Ad clicks are rare. $p(\text{Click})$ must be calibrated (Isotonic Regression) to be comparable to Bid ($).

---

## 3. Tricky Interview Questions

### Q1: Pull vs Push for Feed?
> **Answer**:
> *   **Push (Fan-out on Write)**: Good for small graph. Fast read. Bad for Celebrities (10M writes).
> *   **Pull (Fan-out on Read)**: Good for Celebrities. Slow read.
> *   **Hybrid**: Push for normal users. Pull for celebrities.

### Q2: How to handle "Edit Post"?
> **Answer**:
> *   Feed is immutable log? No.
> *   Update the underlying DB. Cache invalidation (Write-through).

### Q3: Real-time updates (Comments)?
> **Answer**:
> *   **WebSockets**: Keep connection open. Push new comments.
> *   **Long Polling**: Client asks "Any new?" Server holds request until yes.

---

## 4. Practical Edge Case: Feed Pagination
*   **Offset**: `LIMIT 10 OFFSET 100`. Slow.
*   **Cursor**: `WHERE id < last_seen_id LIMIT 10`. Fast. Use Cursor-based pagination.

