# Day 45: Mock Interview: News Feed - Interview Questions

> **Topic**: Social Media Feed (Facebook/Twitter/LinkedIn)
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Design a News Feed.
**Answer:**
*   **Goal**: Maximize Engagement.
*   **Scale**: 1B DAU.
*   **Flow**: Retrieval -> Ranking -> Serving.

### 2. How do you generate the Feed? (Pull vs Push).
**Answer:**
*   **Pull (Fan-out on load)**: Query all friends' posts, merge, sort. Slow for user.
*   **Push (Fan-out on write)**: When user posts, push ID to all followers' pre-computed feed lists. Fast read. Slow write for celebs.
*   **Hybrid**: Push for normal users. Pull for celebs.

### 3. What features would you use?
**Answer:**
*   **User**: Age, Interests, Past CTR.
*   **Post**: Age (Recency), Type (Video/Image), Topic.
*   **Interaction**: User-Post affinity (History).

### 4. How do you handle "Recency"?
**Answer:**
*   Decay function in score: $Score = w_1 \cdot Relevance + w_2 / (Time + 1)$.
*   Or include "Time since published" as feature.

### 5. How do you handle "Diversity"?
**Answer:**
*   Don't show 10 posts from same author.
*   **MMR (Maximal Marginal Relevance)**.
*   Rules: "Max 2 posts per author per page".

### 6. How do you handle Ads?
**Answer:**
*   Ads are just another type of post.
*   Ranked alongside organic content.
*   Score = $Bid \times p(Click)$.
*   Constraint: "One ad every 5 posts".

### 7. What is the Ranking Model?
**Answer:**
*   Logistic Regression (Baseline).
*   GBDT (Facebook used this for years).
*   Neural Net (DLRM) for massive sparse features.

### 8. How do you evaluate the Feed?
**Answer:**
*   **Online**: Time Spent, Daily Active Users.
*   **Offline**: NDCG, AUC.

### 9. How do you handle "Viral Posts"?
**Answer:**
*   Cache them heavily (CDN).
*   Protect DB from hot keys.

### 10. How do you store the Feed?
**Answer:**
*   **Redis**: For active users (List of Post IDs).
*   **Cassandra/HBase**: For posts content.

### 11. What is "EdgeRank"?
**Answer:**
*   Old FB algorithm.
*   Affinity $\times$ Weight $\times$ Time Decay.

### 12. How do you handle "Unseen" posts?
**Answer:**
*   Track "Last Seen Post ID".
*   Only fetch posts > Last Seen.

### 13. How do you handle "Comments" and "Likes" updates?
**Answer:**
*   Eventual consistency.
*   Update counters asynchronously.
*   Client polls or uses WebSocket for real-time.

### 14. What if the user has no friends? (Cold Start).
**Answer:**
*   Show "Popular Global" content.
*   Ask for interests.
*   Import contacts.

### 15. How do you detect Clickbait?
**Answer:**
*   NLP model on title.
*   High CTR but Low Dwell Time (Bounce).
*   Downrank.

### 16. How do you handle "Video" vs "Text"?
**Answer:**
*   Calibrate scores.
*   Video might have higher engagement but takes more space.
*   Value model: $V = \alpha \cdot Click + \beta \cdot WatchTime$.

### 17. What is "Negative Feedback"?
**Answer:**
*   "Hide Post", "Report".
*   Strong negative signal.
*   Filter out similar content immediately.

### 18. How do you scale to 1B users?
**Answer:**
*   Sharding by User ID.
*   Geo-replication.

### 19. How do you handle "Friend Recommendation" (People You May Know)?
**Answer:**
*   Graph traversal (Friends of Friends).
*   Personalized PageRank.

### 20. What is the "Pagination" strategy?
**Answer:**
*   Cursor-based pagination (Post ID).
*   Avoid Offset-based (slow).
