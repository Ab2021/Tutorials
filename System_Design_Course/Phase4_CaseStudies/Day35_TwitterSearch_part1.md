# Day 35 Deep Dive: Real-time Indexing

## 1. The Challenge
*   Standard Elasticsearch refreshes every 1s (Near Real-Time).
*   Twitter needs it faster.
*   **Volume:** 6000 tweets/sec (average), 100k+ (peak).

## 2. Ingestion Pipeline
*   **Tokenizer:** Splits text. Handles hashtags (#), mentions (@), URLs.
*   **Stemming:** `running` -> `run`.
*   **Stop Words:** Remove `the`, `a`. (Maybe keep them for phrase search).
*   **Posting List:** `term -> [doc1, doc2, ...]`.
    *   **Compression:** Use Delta Encoding / PForDelta to compress DocIDs.

## 3. Ranking (Signals)
*   **Static:** User Reputation, Tweet Length, Media Type.
*   **Dynamic:** Likes, Retweets, Replies (Updates in real-time).
*   **Rescoring:**
    *   L1: Retrieve top 1000 by simple relevance (TF-IDF).
    *   L2: Re-rank top 1000 using ML model (Engagement prediction).

## 4. Archive Search
*   **Hot:** Last 7 days. In-memory (Earlybird).
*   **Cold:** Older. Disk-based (Hadoop/Cold Lucene).
*   **Federator:** Queries both Hot and Cold clusters and merges.
