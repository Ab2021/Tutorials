# Day 33 Deep Dive: Ranking & Sharding

## 1. Feed Ranking (Algorithmic Feed)
*   **Chronological:** Easy. Sort by time.
*   **Ranked:** Show "Relevant" posts first.
*   **Signals:**
    *   **Affinity:** How much you interact with this user.
    *   **Weight:** Is it a video or photo?
    *   **Decay:** How old is the post?
*   **Architecture:**
    *   **Candidate Generation:** Get top 1000 recent posts from followers.
    *   **Scoring:** ML Model (Logistic Regression / Neural Net) predicts $P(Click)$.
    *   **Re-ranking:** Sort by Score. Inject Ads.

## 2. Sharding Strategy
*   **Sharding by UserID:**
    *   All data for User A is on Shard 1.
    *   **Pros:** Fast to fetch User Profile.
    *   **Cons:** Hot Shard (Celebrity).
*   **Sharding by PhotoID:**
    *   Photos distributed randomly.
    *   **Pros:** Even distribution.
    *   **Cons:** To fetch User A's photos, must query ALL shards (Scatter-Gather).
*   **Solution:** Shard by UserID, but handle celebrities separately.

## 3. Global ID Generation
*   Instagram uses the **PL/PGSQL Snowflake** approach (discussed in Day 17).
*   ID contains Shard ID, ensuring we know exactly where data lives.
