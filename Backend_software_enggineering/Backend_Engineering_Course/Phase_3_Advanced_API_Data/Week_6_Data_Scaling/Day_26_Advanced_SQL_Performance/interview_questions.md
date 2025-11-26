# Day 26: Interview Questions & Answers

## Conceptual Questions

### Q1: What is a "Table Scan" (Seq Scan) and when is it actually better than an Index Scan?
**Answer:**
*   **Definition**: Reading every row in the table.
*   **When it's bad**: Searching for 1 row in 1 Million (`WHERE id = 5`).
*   **When it's good**: Fetching a large % of the table (e.g., > 20%).
    *   Random I/O (Index) is slower than Sequential I/O (Table Scan).
    *   If you need 90% of the rows, reading the whole file sequentially is faster than jumping around the index.

### Q2: Explain "Optimistic Locking" vs "Pessimistic Locking".
**Answer:**
*   **Pessimistic**: Lock the row immediately. `SELECT * FROM items WHERE id=1 FOR UPDATE`. No one else can touch it until I commit. (Safe but slow).
*   **Optimistic**: Don't lock. Add a `version` column.
    *   Read: `v=1`.
    *   Update: `UPDATE items SET price=10, version=2 WHERE id=1 AND version=1`.
    *   Check: If 0 rows updated, someone else changed it. Retry. (Fast but complex).

### Q3: What is the difference between Clustered and Non-Clustered Indexes?
**Answer:**
*   **Clustered**: The data *is* the index. The rows are physically stored in the order of the index. (Only 1 per table, usually PK).
*   **Non-Clustered**: A separate structure pointing to the data rows. (Can have many).
*   *Postgres Note*: Postgres uses Heap tables (no Clustered Index by default), but `CLUSTER` command can reorder the table to match an index (one-time operation). MySQL (InnoDB) always uses Clustered Index for PK.

---

## Scenario-Based Questions

### Q4: Your database CPU is at 100%. You see thousands of "Idle in Transaction" connections. What is happening?
**Answer:**
*   **Cause**: The application opens a transaction, does some slow work (HTTP call, heavy computation), and *then* commits. The DB connection is held open, holding locks and consuming resources.
*   **Fix**: Keep transactions short. Do the slow work *before* opening the transaction or *after* committing. Use a Connection Pooler (PgBouncer) to multiplex connections.

### Q5: You have a table with 1 Billion rows. You need to delete 100 Million old rows. `DELETE FROM table WHERE date < '2023-01-01'` is timing out.
**Answer:**
*   **Problem**: Huge transaction. Fills up Transaction Log (WAL). Locks the table.
*   **Solution 1 (Batches)**: Delete in loops of 10k rows. `DELETE ... LIMIT 10000`.
*   **Solution 2 (Partitioning)**: If the table was partitioned by date, you could just `DROP TABLE partition_2022`. (Instant).

---

## Behavioral / Role-Specific Questions

### Q6: A developer suggests adding an index on a boolean column `is_active`. Is this useful?
**Answer:**
*   **Usually No**.
*   **Selectivity**: An index is useful if it filters out most rows.
*   **Scenario**: If 50% are active and 50% inactive, the index is useless (DB will likely scan).
*   **Exception**: If 99% are inactive and you query `WHERE is_active = true` (the 1%), then a **Partial Index** (`WHERE is_active = true`) is very useful.
