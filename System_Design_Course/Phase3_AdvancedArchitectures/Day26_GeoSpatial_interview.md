# Day 26 Interview Prep: Geo-Spatial

## Q1: Quadtree vs Geohash?
**Answer:**
*   **Quadtree:** Good for in-memory. Dynamic depth. Hard to persist/shard.
*   **Geohash:** Good for DB (String index). Static grid. Edge cases (Boundary problem).

## Q2: How to handle the "Boundary Problem" in Geohash?
**Answer:**
*   **Problem:** You are at the right edge of cell A. Your neighbor is in cell B. `LIKE 'A%'` won't find B.
*   **Solution:** Query the center cell AND its 8 neighbors.
*   `WHERE hash IN (Center, N, NE, E, SE, S, SW, W, NW)`.

## Q3: Design "Yelp Nearby".
**Answer:**
*   **Data:** Restaurants (Static).
*   **Read/Write:** Read-heavy.
*   **Solution:** PostGIS or Elasticsearch.
*   **Optimization:** Cache results by Geohash. (Users in same block see same list).

## Q4: Design "Uber Live Location".
**Answer:**
*   **Data:** Drivers (Dynamic).
*   **Read/Write:** Write-heavy.
*   **Solution:** Redis (Geo) or In-memory Grid.
*   **Optimization:** Don't write to DB. Ephemeral state.
