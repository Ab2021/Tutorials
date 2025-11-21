# Day 26: Geo-Spatial Indexes

## 1. The Problem
*   "Find all drivers within 5km of me."
*   **Naive:** `SELECT * FROM drivers WHERE sqrt((x-x1)^2 + (y-y1)^2) < 5`.
*   **Issue:** Full table scan. Math on every row. Slow.

## 2. Quadtrees
*   **Concept:** Recursively divide 2D space into 4 quadrants.
*   **Structure:** Tree.
    *   Root: World.
    *   Children: NW, NE, SW, SE.
*   **Search:** Traverse tree. Ignore quadrants outside range.
*   **Pros:** Variable resolution (dense areas have deeper trees).
*   **Cons:** Rebalancing is hard (if drivers move).

## 3. Geohash
*   **Concept:** Encode Lat/Lon into a string.
*   **Mechanism:** Interleave bits of Lat and Lon. Base32 encode.
*   **Property:** Shared prefix = Proximity.
    *   `u4pru` is inside `u4pr`.
*   **Search:** `SELECT * FROM drivers WHERE geohash LIKE 'u4pru%'`.
*   **Pros:** Single string column. Easy to index (B-Tree).
*   **Cons:** Edge cases (two close points might have different hashes at boundary).

## 4. Google S2
*   **Concept:** Project Earth onto a Cube. Use Hilbert Curve to fill space.
*   **Pros:** Mathematical perfection. Handles poles better. Used by Uber/Pokemon Go.
