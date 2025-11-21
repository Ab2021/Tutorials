# Day 5 Interview Prep: CAP & Consistency

## Q1: Explain CAP Theorem with an example.
**Answer:**
*   Imagine a distributed database with 2 nodes (A and B).
*   **Partition:** The network cable between A and B is cut.
*   **Scenario:** User writes to A.
*   **Choice:**
    *   **CP:** A refuses the write because it can't replicate to B. (System Unavailable).
    *   **AP:** A accepts the write. B doesn't know about it. (System Inconsistent).

## Q2: When to use SQL vs NoSQL?
**Answer:**
*   **SQL:** Structured data, complex joins, transactions (Banking, E-commerce orders).
*   **NoSQL:** Unstructured data, high throughput, simple queries, massive scale (User logs, Social feed, IoT data).

## Q3: What is Eventual Consistency?
**Answer:**
*   A consistency model used in distributed systems to achieve high availability.
*   It guarantees that if no new updates are made to a given data item, eventually all accesses to that item will return the last updated value.
*   **Trade-off:** Users might see stale data for a few milliseconds.

## Q4: How does Cassandra handle conflicts?
**Answer:**
*   **Last Write Wins (LWW):** Uses timestamp. The write with the latest timestamp overwrites others.
*   **Vector Clocks:** Detects concurrent writes and asks the client to resolve (rarely used in default config).
