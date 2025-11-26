# Day 26: Advanced Relational Design & Performance

## 1. It's Not the Database, It's You

Most "slow database" problems are actually "bad query" problems.
Today we stop guessing and start measuring.

### 1.1 EXPLAIN ANALYZE
The most important command in SQL.
*   **EXPLAIN**: Shows the *plan* (what the DB *thinks* it will do).
*   **ANALYZE**: Runs the query and shows *actual* execution time.

**Key Metrics**:
*   **Cost**: Arbitrary units (disk I/O + CPU). Lower is better.
*   **Rows**: Estimated vs Actual rows. (Big difference = Outdated Statistics).
*   **Loops**: How many times a node was executed (Nested Loops).

---

## 2. Advanced Indexing

We know B-Trees. Let's go deeper.

### 2.1 Composite Indexes (The Order Matters)
Index `(A, B)` is NOT the same as `(B, A)`.
*   Query `WHERE A=1 AND B=2`: Uses `(A, B)`.
*   Query `WHERE A=1`: Uses `(A, B)`.
*   Query `WHERE B=2`: **Cannot** use `(A, B)`. (Like finding a name in a phonebook by last letter).

### 2.2 Covering Index (Index Only Scan)
If the index contains *all* the columns you need, the DB doesn't touch the Heap (Table).
*   Query: `SELECT email FROM users WHERE id = 5`
*   Index: `(id, email)`
*   Result: **Blazing fast**.

### 2.3 Partial Index
Index only a subset of rows.
*   Scenario: You have 1M users, but only 1000 are "active". You mostly query active users.
*   Index: `CREATE INDEX idx_active_users ON users(id) WHERE status = 'active';`
*   Result: Tiny index, fast updates.

---

## 3. Partitioning

Splitting one giant table into smaller physical tables (transparent to the app).

### 3.1 Strategies
1.  **Range**: `orders_2023`, `orders_2024`. (Good for Time-Series).
2.  **List**: `users_us`, `users_eu`. (Good for Data Residency).
3.  **Hash**: `users_0` ... `users_3`. (Good for load distribution).

### 3.2 Benefits
*   **Performance**: Scans only relevant partitions (`Partition Pruning`).
*   **Maintenance**: Drop old data instantly (`DROP TABLE orders_2020`) instead of `DELETE WHERE year=2020` (which causes vacuum/fragmentation).

---

## 4. Connection Pooling

Opening a TCP connection to Postgres takes time (SSL handshake, Auth).
*   **Anti-Pattern**: Open/Close connection for every HTTP request.
*   **Pattern**: Keep a pool of open connections (e.g., 20). Reuse them.
*   **Tools**:
    *   **App Side**: HikariCP (Java), SQLAlchemy Pool (Python).
    *   **Server Side**: PgBouncer (Middleware).

---

## 5. Summary

Today we tuned the engine.
*   **Analyze**: Don't guess.
*   **Index**: Cover your queries.
*   **Partition**: Divide and conquer.
*   **Pool**: Reuse connections.

**Tomorrow (Day 27)**: We leave SQL behind. We will look at **NoSQL Patterns** and when to use Redis/Mongo for specific problems like Leaderboards and Sessions.
