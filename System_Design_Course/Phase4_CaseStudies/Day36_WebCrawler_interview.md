# Day 36 Interview Prep: Design Web Crawler

## Q1: How to handle DNS resolution at scale?
**Answer:**
*   DNS is slow (UDP/TCP).
*   **Solution:** Build a custom DNS Resolver.
*   **Cache:** Aggressive caching.
*   **Prefetch:** If we see a link, resolve DNS before fetching.

## Q2: How to check if a URL is visited?
**Answer:**
*   **Volume:** Billions of URLs. RAM is not enough for HashSet.
*   **Bloom Filter:** Probabilistic. Space efficient.
    *   **False Positive:** Might say "Visited" when not. (Okay, we skip one page).
    *   **False Negative:** Impossible.
*   **Disk:** Store full list on disk (RocksDB) to verify Bloom Filter positives.

## Q3: How to render JavaScript (SPA)?
**Answer:**
*   Traditional crawler only sees HTML.
*   **Headless Browser:** Use Chrome Headless / Puppeteer.
*   **Cost:** Expensive (CPU/RAM).
*   **Strategy:** Only render for high-value domains.

## Q4: How to distribute the crawl?
**Answer:**
*   **Consistent Hashing:** Hash(Hostname) -> Worker Node.
*   **Benefit:** All URLs from `cnn.com` go to same worker. Easier to enforce politeness (Rate Limit).
