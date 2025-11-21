# Day 36: Design Web Crawler

## 1. Requirements
*   **Goal:** Crawl the web (Billions of pages) to build a search index.
*   **Politeness:** Don't DDoS servers. Respect `robots.txt`.
*   **Freshness:** Re-crawl updated pages.
*   **Scale:** 1 Billion pages/month.

## 2. Architecture
*   **Seed URLs:** Starting point (e.g., CNN, Wikipedia).
*   **URL Frontier:** Priority Queue of URLs to crawl.
*   **Fetcher:** Downloads HTML.
*   **DNS Resolver:** Caches IP addresses.
*   **Content Parser:** Extracts text and links.
*   **Duplicate Detector:** Checks if page is already seen.
*   **Link Extractor:** Finds new URLs and adds to Frontier.

## 3. URL Frontier (The Brain)
*   **Politeness:** Ensure we don't hit `example.com` 100 times/sec.
*   **Priority:** Crawl high-rank pages (PageRank) more often.
*   **Implementation:**
    *   **Front Queue:** Prioritizer.
    *   **Back Queue:** Politeness enforcer. One queue per domain.
    *   **Heap:** Manage release times for queues.

## 4. Storage
*   **Content:** Bigtable / HBase (Compressed).
*   **Metadata:** SQL/NoSQL.
