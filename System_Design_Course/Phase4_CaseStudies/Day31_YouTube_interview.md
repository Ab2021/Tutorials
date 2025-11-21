# Day 31 Interview Prep: Design YouTube

## Q1: How to handle "The Thundering Herd" for a viral video?
**Answer:**
*   Millions of users request the same video at once.
*   **CDN:** Caches the video at the edge.
*   **Origin Shield:** A mid-tier cache to protect S3.
*   **P2P (Peer-to-Peer):** Clients share chunks with each other (used by Spotify/Facebook Live).

## Q2: Why use HLS over MP4?
**Answer:**
*   **MP4:** Single file. Hard to switch quality mid-stream.
*   **HLS:** Playlist of small chunks. Allows Adaptive Bitrate (ABR). Better for mobile networks.

## Q3: How to implement "Watch Later" or "History"?
**Answer:**
*   **Write-heavy:** Every second of watch time generates a write.
*   **Solution:**
    *   Buffer in memory (Redis) on the client/edge.
    *   Flush to DB (Cassandra/HBase) every minute.
    *   Use a Time Series Database pattern.

## Q4: How to calculate "View Count"?
**Answer:**
*   **Approximate:** Increment a Redis counter. Flush to DB periodically.
*   **Exact:** For monetization, we need exact counts.
    *   Log every view event to Kafka.
    *   Run a Flink job to filter bots and aggregate unique views.
    *   Update DB.
