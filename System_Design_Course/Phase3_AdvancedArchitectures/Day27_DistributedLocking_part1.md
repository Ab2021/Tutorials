# Day 27 Deep Dive: Redlock Algorithm

## 1. The Problem with Single Redis
*   If Redis Master crashes after accepting a lock but *before* replicating to Slave, the Slave promotes to Master.
*   The lock is lost. Another client can grab it.
*   **Result:** Violation of Mutual Exclusion.

## 2. Redlock (Redis Distributed Lock)
*   **Setup:** $N$ Redis masters (e.g., 5). Totally independent (no replication).
*   **Algorithm:**
    1.  Client gets current time $T_1$.
    2.  Client tries to acquire lock in all $N$ instances sequentially.
    3.  Client calculates elapsed time $T_{elapsed} = T_{now} - T_1$.
    4.  **Success:** If Client acquired lock in majority ($N/2 + 1$, i.e., 3 nodes) AND $T_{elapsed} < TTL$.
    5.  **Failure:** Unlock all instances.

## 3. Controversy (Martin Kleppmann vs Antirez)
*   **Kleppmann:** Redlock is dangerous because of **Clock Drift**. If one server's clock jumps forward, it might expire the lock too early.
*   **Antirez (Redis Creator):** Clock drift is rare and manageable.
*   **Conclusion:** Use Redlock for efficiency (preventing duplicate work). Use Zookeeper/Etcd for correctness (financial transactions).

## 4. Fencing Token
*   **Scenario:** Client A pauses (GC pause) for minutes. Lock expires. Client B grabs lock. Client A wakes up and writes to DB.
*   **Fix:** Fencing Token.
    *   Lock Service returns a monotonic token (1, 2, 3...).
    *   DB checks token. If `Token < LastSeenToken`, reject write.
