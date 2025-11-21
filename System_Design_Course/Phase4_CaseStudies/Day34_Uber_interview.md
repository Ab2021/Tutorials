# Day 34 Interview Prep: Design Uber

## Q1: How to handle high write throughput of location updates?
**Answer:**
*   **Volume:** 10M drivers * 1 update/4s = 2.5M QPS.
*   **DB:** Postgres can't handle this.
*   **Solution:**
    *   **Redis:** Update in-memory only.
    *   **Snapshots:** Only persist to disk (Cassandra) every 30s or on trip end.
    *   **Sharding:** Shard Redis by DriverID.

## Q2: What if the Matching Service crashes?
**Answer:**
*   **Stateless:** Matching service should be stateless.
*   **State:** State lives in DB/Redis.
*   **Recovery:** Spin up new instance. It reads pending requests from Queue/DB and resumes matching.

## Q3: How to handle "Driver declines ride"?
**Answer:**
*   **Flow:**
    1.  Match Driver A.
    2.  Send Offer.
    3.  Driver A declines (or timeout).
    4.  **Negative Ack:** Mark Driver A as "Ignored" for this Trip.
    5.  Rematch (Find next nearest driver excluding A).

## Q4: How to prevent double booking?
**Answer:**
*   **Distributed Lock:** Acquire lock on DriverID in Redis.
*   **Database Constraint:** `UPDATE drivers SET status='BUSY' WHERE id=1 AND status='AVAILABLE'`. Check row count.
