# Day 30 Interview Prep: Project Defense

## Q1: Why Consistent Hashing?
**Answer:**
*   If we used `hash(key) % N`, adding a node would change the mapping for almost ALL keys.
*   With Consistent Hashing, adding a node only affects $1/N$ of the keys (neighbors).
*   Minimizes data movement during scaling.

## Q2: How to handle concurrent writes?
**Answer:**
*   **Last Write Wins (LWW):** Use system timestamp. Simple but clock drift can cause data loss.
*   **Vector Clocks:** `[A:1, B:2]`. Detects conflicts. Client must resolve them.

## Q3: What is Gossip Protocol?
**Answer:**
*   Nodes randomly ping other nodes: "Are you alive?".
*   Information propagates like a virus ($O(\log N)$).
*   Used for **Membership** (Who is in the cluster?) and **Failure Detection**.

## Q4: Tunable Consistency?
**Answer:**
*   **Strong:** $R + W > N$. (e.g., $2+2 > 3$). Guaranteed to read latest write.
*   **Eventual:** $R + W \le N$. (e.g., $1+1 \le 3$). Fast, but might read stale data.
