# Day 6 Interview Prep: Caching

## Q1: What is the difference between Write-Through and Write-Back?
**Answer:**
*   **Write-Through:** Write to Cache AND DB synchronously. Safe but slower write latency.
*   **Write-Back:** Write to Cache only. Cache updates DB asynchronously. Fast write latency but risk of data loss.

## Q2: How do you solve the Thundering Herd problem?
**Answer:**
*   **Mutex Lock:** The first process to get a cache miss acquires a lock to update the cache. Others wait.
*   **Pre-warming:** A background job refreshes the cache before it expires.

## Q3: Explain LRU eviction.
**Answer:**
*   Least Recently Used.
*   Keep a list. When an item is accessed, move it to the front.
*   When full, remove the item at the back.
*   Efficiently implemented with a Hash Map + Doubly Linked List ($O(1)$).

## Q4: Where can you place a cache?
**Answer:**
*   **Browser:** Local storage, HTTP Cache.
*   **CDN:** Edge locations (Static assets).
*   **Load Balancer:** Reverse Proxy cache (Nginx).
*   **App Server:** Local memory (fastest, but not shared).
*   **Distributed Cache:** Redis/Memcached (shared, scalable).
