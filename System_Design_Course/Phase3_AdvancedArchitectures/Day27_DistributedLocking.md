# Day 27: Distributed Locking

## 1. The Problem
*   **Local Lock:** `synchronized` or `Mutex` works for one process.
*   **Distributed:** Multiple processes on different servers trying to access a shared resource (e.g., "Booking the last seat").
*   **Race Condition:**
    *   Server A reads Seat 1 (Available).
    *   Server B reads Seat 1 (Available).
    *   Server A books it.
    *   Server B books it.
    *   **Result:** Double booking.

## 2. Requirements
*   **Mutual Exclusion:** Only one client holds the lock.
*   **Deadlock Free:** If client crashes, lock must be released (TTL).
*   **Fault Tolerance:** If Lock Manager crashes, system should survive.

## 3. Solutions
### Database Row Lock
*   `SELECT * FROM seats WHERE id=1 FOR UPDATE;`
*   **Pros:** Simple, Strong consistency.
*   **Cons:** Slow. Holds DB connection.

### Redis (SetNX)
*   `SET resource_name my_random_value NX PX 30000`
*   **NX:** Only set if not exists.
*   **PX:** Expire in 30s (TTL).
*   **Release:** Lua script to check if value matches `my_random_value` (to prevent deleting someone else's lock) and then DEL.

### Zookeeper / Etcd
*   **Mechanism:** Create an Ephemeral Node.
*   **Pros:** Strongest consistency. Session based (if client disconnects, lock is gone).
*   **Cons:** Complex setup.
