# Day 35: Design Twitter Search

## 1. Requirements
*   **Functional:** Search tweets by keyword, hashtag, user. Real-time results.
*   **Non-Functional:** Low latency, High ingestion rate (New tweets searchable instantly).
*   **Scale:** 500M tweets/day.

## 2. Architecture
*   **Ingestion:**
    *   User posts Tweet.
    *   Write to Tweet DB (Cassandra).
    *   Push to Kafka ("NewTweet").
*   **Indexing:**
    *   Search Worker reads Kafka.
    *   Tokenizes tweet.
    *   Updates Inverted Index (Elasticsearch/Earlybird).
*   **Query:**
    *   User searches "System Design".
    *   Search Service queries Index.
    *   Merges results, ranks, returns.

## 3. Earlybird (Twitter's Engine)
*   **Lucene based:** Modified version of Lucene.
*   **In-Memory:** Holds recent tweets in RAM for fast access.
*   **Segments:**
    *   **Active Segment:** Writable. In-memory.
    *   **Read-Only Segment:** Immutable. On disk.
*   **Optimization:** Tweets are small. Index is optimized for short text.

## 4. Scatter-Gather
*   Index is sharded (by TweetID or Time).
*   Query must go to ALL shards.
*   **Blender:** Service that aggregates results from all shards and sorts them.
