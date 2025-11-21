# Day 35 Interview Prep: Design Twitter Search

## Q1: Shard by TweetID vs Time?
**Answer:**
*   **By Time:**
    *   **Pros:** Easy to query "Last 1 hour". Only query recent shards.
    *   **Cons:** Hot spotting. All writes go to the "Current" shard.
*   **By TweetID:**
    *   **Pros:** Even load distribution.
    *   **Cons:** Query must hit ALL shards (Scatter-Gather).
*   **Twitter's Choice:** Shard by TweetID (to handle write load), but optimize Scatter-Gather.

## Q2: How to handle "Trending Topics"?
**Answer:**
*   **Stream Processing:** Use Flink/Storm.
*   **Window:** Sliding Window (Last 15 mins).
*   **Count:** Count hashtags.
*   **Heavy Hitters:** Use **Count-Min Sketch** (Probabilistic data structure) to count top K items with low memory.

## Q3: How to update "Like Count" in Search Index?
**Answer:**
*   Updating the Inverted Index for every Like is too expensive.
*   **Solution:** Store "Like Count" in a separate Forward Index (DocID -> Count).
*   During query, fetch DocIDs from Inverted Index, then look up Counts from Forward Index to compute score.

## Q4: How to handle multi-language search?
**Answer:**
*   **Language Detection:** Detect lang at ingestion.
*   **Analyzers:** Use specific analyzers (Japanese Kuromoji, English Standard) based on lang.
*   **Index:** Store in separate indices (`tweets_en`, `tweets_jp`) or use a field `lang`.
