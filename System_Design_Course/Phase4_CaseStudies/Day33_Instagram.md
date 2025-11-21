# Day 33: Design Instagram Feed

## 1. Requirements
*   **Functional:** Post Photo, Follow User, View News Feed.
*   **Non-Functional:** Low latency (Feed loads < 200ms), High availability.
*   **Scale:** 1 Billion users. Read-heavy (Feed) vs Write-heavy (Posting).

## 2. Feed Generation Models
### Pull Model (Fan-out on Load)
*   **Concept:** User opens app. Server queries: `SELECT * FROM posts WHERE user_id IN (following_ids) ORDER BY time DESC`.
*   **Pros:** Simple. Real-time.
*   **Cons:** Slow for users following 1000+ people. DB load is huge.

### Push Model (Fan-out on Write)
*   **Concept:** User A posts. Server pushes Post ID to the "Feed Cache" of all followers.
*   **Read:** User B opens app. Reads from their pre-computed Feed Cache (Redis).
*   **Pros:** Extremely fast reads ($O(1)$).
*   **Cons:** Slow writes. If Justin Bieber posts (100M followers), we must update 100M caches. (Thundering Herd).

## 3. Hybrid Approach (The Winner)
*   **Normal Users:** Use Push Model.
*   **Celebrities:** Use Pull Model.
*   **Feed Construction:** Merge results from Push Cache + Pull (Celebrity) DB query.

## 4. Storage
*   **User/Graph:** SQL (MySQL/Postgres) or Graph DB (Neo4j).
*   **Posts:** NoSQL (Cassandra) or Sharded SQL.
*   **Images:** S3 + CDN.
