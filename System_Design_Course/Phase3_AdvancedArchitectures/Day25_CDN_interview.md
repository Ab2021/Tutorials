# Day 25 Interview Prep: CDNs

## Q1: How does DNS resolution work with CDN?
**Answer:**
1.  User queries `www.example.com`.
2.  CNAME record points to `example.cdn.com`.
3.  CDN's DNS server sees User IP.
4.  Returns IP of the nearest Edge Node.

## Q2: How to serve dynamic content via CDN?
**Answer:**
*   **Edge Side Includes (ESI):** Assemble page at edge (Header + Body + Footer).
*   **Dynamic Acceleration:** Use CDN as a fast tunnel (optimized TCP/TLS) to Origin.
*   **Edge Compute:** Run logic at edge.

## Q3: What is the Thundering Herd problem in CDNs?
**Answer:**
*   If a hot file expires, 1000 edge nodes might hit Origin simultaneously.
*   **Solution:** **Origin Shield (Collapsed Forwarding)**.
    *   Edge Nodes -> Regional Edge -> Origin.
    *   Only the Regional Edge hits Origin once.

## Q4: Push vs Pull CDN?
**Answer:**
*   **Pull:** Lazy. Good for websites.
*   **Push:** Proactive. Good for 10GB game patches (Pre-warm the cache).
