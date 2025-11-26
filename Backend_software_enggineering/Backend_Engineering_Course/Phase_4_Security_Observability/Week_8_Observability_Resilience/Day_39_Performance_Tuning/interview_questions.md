# Day 39: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the difference between `Cache-Control: no-cache` and `no-store`?
**Answer:**
*   **no-store**: **Never** cache this. Do not save it to disk. (For banking data).
*   **no-cache**: You *can* cache it, but you must **revalidate** with the server (ETag) before using it. (Checks if it changed).

### Q2: How does a CDN work?
**Answer:**
*   **Anycast DNS**: Routes user to the nearest Edge Location.
*   **Proxy**: Edge checks its cache.
    *   *Hit*: Returns immediately.
    *   *Miss*: Fetches from Origin, caches it, returns it.

### Q3: Explain "Cache Penetration".
**Answer:**
*   **Problem**: Malicious user requests keys that *don't exist* (`id=-1`).
*   **Flow**: Cache Miss -> DB Miss (returns null).
*   **Result**: Every request hits the DB.
*   **Fix**:
    1.  **Cache Nulls**: Store `key=-1, value=null` with short TTL.
    2.  **Bloom Filter**: Check if ID exists in Bloom Filter before hitting Cache/DB.

---

## Scenario-Based Questions

### Q4: Your API is slow. You added Redis, but it's still slow. Why?
**Answer:**
*   **Serialization**: Maybe JSON dumping 1MB objects is taking CPU time.
*   **Network**: Maybe the link to Redis is saturated.
*   **Big Keys**: Fetching a 10MB value from Redis blocks the single thread.
*   **Fix**: Profile the app. Check Redis `SLOWLOG`.

### Q5: You updated the CSS file, but users still see the old version. How do you force an update?
**Answer:**
*   **Cache Busting (Fingerprinting)**.
*   Rename the file: `style.css` -> `style.v2.css` or `style.a1b2c3.css`.
*   Update the HTML to point to the new file.
*   The browser treats it as a completely new resource.

---

## Behavioral / Role-Specific Questions

### Q6: A developer wants to cache *everything* in Redis to make it fast. Good idea?
**Answer:**
*   **No**.
*   **Memory Cost**: RAM is expensive.
*   **Complexity**: Invalidation logic becomes a nightmare.
*   **Staleness**: Users might see wrong data.
*   **Advice**: Cache only the "Hot" data (Pareto Principle: 20% of data gets 80% of traffic).
