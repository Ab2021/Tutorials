# Day 27 Interview Prep: Distributed Locking

## Q1: Optimistic vs Pessimistic Locking?
**Answer:**
*   **Pessimistic:** "Lock the resource before using it." (Redis Lock, DB `FOR UPDATE`). Good for high contention.
*   **Optimistic:** "Read, Modify, Verify." (Version number).
    *   `UPDATE table SET val=new, ver=2 WHERE id=1 AND ver=1`.
    *   If rows affected = 0, retry.
    *   Good for low contention.

## Q2: How to handle Deadlocks?
**Answer:**
*   **TTL:** Always set an expiration on the lock.
*   **Ordering:** Always acquire locks in the same order (Resource A then B).
*   **Detection:** Build a "Wait-for graph" and find cycles.

## Q3: What is a Fencing Token?
**Answer:**
*   A number that increases every time a lock is acquired.
*   Used to prevent "Zombie Clients" (clients that think they hold the lock but it expired) from corrupting data.
*   The storage layer (DB) must check the token.

## Q4: When to use Zookeeper over Redis for locking?
**Answer:**
*   Use **Zookeeper** when correctness is critical (e.g., Leader Election, Money Transfer). It guarantees CP (Consistency).
*   Use **Redis** when performance is critical and occasional duplicate work is acceptable (e.g., Processing a background job).
