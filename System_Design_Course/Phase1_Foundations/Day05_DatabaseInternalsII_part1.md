# Day 5 Deep Dive: Consistency Models

## 1. Consistency Spectrum
*   **Strong Consistency:** Once a write is acknowledged, all subsequent reads see it. (SQL).
*   **Weak Consistency:** No guarantee. (Video chat).
*   **Eventual Consistency:** If no new updates are made, eventually all accesses will return the last updated value. (DNS, Cassandra).
*   **Causal Consistency:** Operations that are causally related are seen by everyone in the same order.

## 2. Quorums (N, R, W)
How to tune consistency in distributed DBs (Cassandra/Dynamo).
*   **N:** Number of replicas.
*   **R:** Number of replicas that must agree for a Read to succeed.
*   **W:** Number of replicas that must acknowledge a Write to succeed.

### Formulas
*   **Strong Consistency:** $R + W > N$. (Overlap ensures you always read at least one node that has the latest write).
*   **High Availability (Fast Write):** $W = 1$. (Risk of data loss).
*   **High Availability (Fast Read):** $R = 1$. (Risk of stale data).

### Example (N=3)
*   **Strong:** $W=2, R=2$. ($2+2 > 3$).
*   **Fast Write:** $W=1, R=3$. (Slow read).

## 3. Vector Clocks
*   **Problem:** In AP systems, two nodes might accept writes for the same key at the same time (Conflict).
*   **Solution:** Attach a version vector `[NodeA: 1, NodeB: 2]`.
*   **Resolution:**
    *   If V1 > V2, V1 wins.
    *   If concurrent, Client must resolve (e.g., "Merge shopping cart").
