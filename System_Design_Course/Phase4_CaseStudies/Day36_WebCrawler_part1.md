# Day 36 Deep Dive: Deduplication & PageRank

## 1. Duplicate Detection
*   **Problem:** 30% of web is duplicates.
*   **Exact Match:** Hash (MD5/SHA).
*   **Near Duplicate:** SimHash.
    *   **SimHash:** Generates a fingerprint.
    *   **Hamming Distance:** If distance between two fingerprints is small (e.g., < 3 bits flip), they are near duplicates.
    *   **Benefit:** Detects copy-pasted content with minor changes.

## 2. PageRank (Link Analysis)
*   **Concept:** A page is important if important pages link to it.
*   **Formula:** $PR(A) = (1-d) + d \sum (PR(T_i) / C(T_i))$
    *   $d$: Damping factor (0.85).
    *   $C(T_i)$: Outbound links on page $T_i$.
*   **Computation:**
    *   Iterative algorithm.
    *   MapReduce / Spark GraphX.
    *   Run until convergence.

## 3. Handling Traps
*   **Spider Traps:** Infinite loops (`calendar.php?year=2020`, `year=2021`...).
*   **Solution:**
    *   Max URL length.
    *   Max depth.
    *   Bloom Filter to detect visited URLs.

## 4. Robots.txt
*   Standard for exclusion.
*   Fetcher must check `example.com/robots.txt` before crawling.
*   Cache this file to avoid re-fetching.
