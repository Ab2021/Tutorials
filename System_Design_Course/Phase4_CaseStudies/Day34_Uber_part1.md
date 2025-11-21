# Day 34 Deep Dive: Hexagonal Grid & State Machine

## 1. Why Hexagons (H3)?
*   **Grid:** We need to divide the world into cells to index drivers.
*   **Square:** Diagonals are longer than sides. Neighbors have different distances.
*   **Hexagon:**
    *   All 6 neighbors are equidistant.
    *   Approximates a circle better.
    *   Easy to aggregate (7 small hexes = 1 big hex).
*   **Uber H3:** Open source library for hexagonal hierarchical spatial index.

## 2. Trip State Machine
*   **States:** `REQUESTED`, `MATCHING`, `MATCHED`, `ARRIVED`, `IN_PROGRESS`, `COMPLETED`, `CANCELLED`.
*   **Implementation:**
    *   **Database:** ACID is required. (Postgres).
    *   **Locking:** When matching, lock the Driver row to prevent double assignment.
    *   **Idempotency:** If client retries "Request Ride", return existing Trip ID.

## 3. Surge Pricing
*   **Goal:** Balance Supply (Drivers) and Demand (Riders).
*   **Mechanism:**
    *   Calculate `Demand / Supply` ratio per Hexagon.
    *   If Ratio > 1, increase price multiplier.
    *   **Propagation:** Update price in Redis. Push to clients via WebSocket.
    *   **Consistency:** Price is valid for X minutes (Lock the price at request time).
