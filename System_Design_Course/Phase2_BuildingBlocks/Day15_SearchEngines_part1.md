# Day 15 Deep Dive: Elasticsearch Architecture

## 1. Architecture
*   **Node:** A server.
*   **Index:** A collection of documents (Logical DB).
*   **Shard:** A Lucene instance. An index is split into $N$ shards.
*   **Replica:** Copy of a shard.

## 2. Write Path (Indexing)
1.  Client sends JSON to Coordinator Node.
2.  Coordinator hashes ID to find Primary Shard.
3.  Primary Shard writes to buffer and Translog (WAL).
4.  Primary replicates to Replica Shards.
5.  **Refresh:** Every 1s, buffer is converted to a Segment (Searchable). "Near Real-Time".

## 3. Read Path (Searching)
1.  Client sends query to Coordinator.
2.  **Scatter:** Coordinator sends query to *all* shards (Primary or Replica).
3.  **Gather:** Each shard returns top $K$ results (IDs + Scores).
4.  **Merge:** Coordinator merges results, sorts, fetches full documents, returns to client.

## 4. Case Study: Uber Marketplace Search
*   **Challenge:** Search for drivers/restaurants nearby. Geo-spatial + Real-time.
*   **Solution:**
    *   **Geohash:** Encode Lat/Lon into string (`u4pruyd`).
    *   **Indexing:** Store Geohash in Inverted Index.
    *   **Query:** "Find drivers in `u4pru`".
    *   **Updates:** Driver location updates every 4s. High write throughput.
    *   **Optimization:** Use specialized Geo-index (Quadtree/Google S2) in memory (Redis) for live drivers, Elasticsearch for static places.
