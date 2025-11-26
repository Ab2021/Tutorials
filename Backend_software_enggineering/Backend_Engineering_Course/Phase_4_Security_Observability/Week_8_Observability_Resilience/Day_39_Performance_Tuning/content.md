# Day 39: Performance Tuning & Caching

## 1. Where is the time going?

Latency = Network + App + DB.
*   **Network**: Distance (Speed of light). Fix: CDN.
*   **App**: CPU/GC. Fix: Profiling.
*   **DB**: I/O. Fix: Indexing/Caching.

---

## 2. Caching Layers

### 2.1 Browser Cache (The Fastest)
*   **Header**: `Cache-Control: max-age=3600`.
*   **Effect**: Browser doesn't even talk to the server for 1 hour.
*   **Use Case**: Images, CSS, JS.

### 2.2 CDN (Content Delivery Network)
*   **Concept**: Servers close to the user (Edge).
*   **Effect**: User in London hits London server, not your Origin in Virginia.
*   **Use Case**: Static Assets, Public API responses.

### 2.3 Application Cache (Redis)
*   **Concept**: Store expensive DB queries in RAM.
*   **Pattern**: Look-aside.

---

## 3. Cache Invalidation

"There are only two hard things in Computer Science: Cache Invalidation and Naming Things."

### 3.1 Strategies
1.  **TTL (Time To Live)**: Set `EXPIRE 60`. Data is stale for max 60s. (Easy, Eventual Consistency).
2.  **Write-Through**: Update DB and Cache at the same time. (Strong Consistency, Slower writes).
3.  **Purge (Ban)**: Explicitly delete key when data changes. (Complex, prone to bugs).

---

## 4. Profiling

Don't guess. Measure.
*   **Python**: `cProfile`, `py-spy`.
*   **Go**: `pprof`.
*   **Flame Graphs**: Visualize where CPU time is spent.

---

## 5. Summary

Today we made it fast.
*   **Browser**: `Cache-Control`.
*   **CDN**: Move data closer.
*   **Redis**: Save the DB.
*   **Profile**: Find the bottleneck.

**Tomorrow (Day 40)**: We wrap up Phase 4 with **Resilience**. Circuit Breakers, Retries, and Rate Limiting.
